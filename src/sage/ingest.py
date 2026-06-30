"""Ingestion pipeline.

    files -> load (any format) -> chunk -> enrich metadata -> embed
          -> persist Chroma (dense)  +  persist BM25 chunk store (sparse)

Two ways to use it:

    # CLI — (re)build from the documents/ folder (and legacy 'all pdf/')
    python -m sage.ingest            # incremental
    python -m sage.ingest --rebuild  # wipe and rebuild from scratch

    # In-app — add freshly uploaded files to the live index
    from sage.ingest import ingest_paths
    summary = ingest_paths([Path("documents/report.xlsx"), ...])

The index is built ONCE and persisted, then merely loaded by the app, so cold
start stays fast. Re-ingesting a file with the same name REPLACES its old
chunks (clean update, no duplicates). Both stores are written atomically so an
interrupted ingest can never corrupt the index.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from sage.config import settings
from sage.exceptions import FileTooLargeError
from sage.io_utils import atomic_write_pickle, atomic_write_text
from sage.loaders import is_supported, load_documents
from sage.logging_config import get_logger

if TYPE_CHECKING:  # heavy ML deps are imported lazily inside functions
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _section_from_filename(filename: str) -> str:
    """A clean, human-friendly title used in metadata + citations.

    'ACTUALISAT-LLM _ 03 Core Principles-090725-070353.pdf' -> 'Core Principles'
    'Q3 Financials.xlsx'                                     -> 'Q3 Financials'
    """
    stem = Path(filename).stem
    tail = stem.split("_")[-1].strip()
    parts = tail.split("-")
    keep = [p for p in parts if not p.strip().isdigit()]
    section = "-".join(keep).strip()
    tokens = section.split()
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    return " ".join(tokens).strip() or stem


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _splitter() -> RecursiveCharacterTextSplitter:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def _embeddings() -> HuggingFaceEmbeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def check_upload_size(path: Path) -> None:
    """Raise :class:`FileTooLargeError` if a file exceeds the configured limit."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise FileTooLargeError(
            f"{path.name} is {size_mb:.1f} MB (limit {settings.max_upload_mb} MB)"
        )


def _chunk_file(
    path: Path, splitter: RecursiveCharacterTextSplitter, origin: str
) -> list[Document]:
    """Load + chunk a single file into enriched Documents (or [] if no text).

    ``origin`` is "upload" (user-uploaded) or "static" (default ``all pdf``
    corpus) — it drives query-time routing (uploaded docs answer on their own;
    otherwise we fall back to the static knowledge base).
    """
    section = _section_from_filename(path.name)
    pages = load_documents(path)
    if not pages:
        return []

    chunks = splitter.split_documents(pages)
    for i, ch in enumerate(chunks):
        ch.metadata.update(
            {
                "source": path.name,
                "section": section,
                "origin": origin,
                "page": int(ch.metadata.get("page", 1)),
                "chunk_id": f"{section}-{i}",
                "content_hash": _content_hash(ch.page_content),
            }
        )
    return chunks


# --------------------------------------------------------------------------- #
# Persistence layer (Chroma + BM25 pickle kept in sync, written atomically)
# --------------------------------------------------------------------------- #
def _bump_version() -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    cur = 0
    if settings.version_path.exists():
        try:
            cur = int(settings.version_path.read_text().strip() or 0)
        except ValueError:
            cur = 0
    atomic_write_text(settings.version_path, str(cur + 1))


def _load_bm25() -> list[Document]:
    if settings.bm25_path.exists():
        try:
            with open(settings.bm25_path, "rb") as fh:
                return pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            log.error("BM25 store unreadable (%s); treating as empty", exc)
    return []


def _save_bm25(chunks: list[Document]) -> None:
    atomic_write_pickle(settings.bm25_path, chunks)


def _open_chroma(embeddings: HuggingFaceEmbeddings) -> Chroma:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
    )


def indexed_sources(origin: str | None = None) -> dict[str, int]:
    """Map of ``{source filename -> chunk count}`` in the index, optionally
    filtered by origin ("upload" or "static")."""
    counts: dict[str, int] = {}
    for d in _load_bm25():
        if origin is not None and d.metadata.get("origin") != origin:
            continue
        src = d.metadata.get("source", "?")
        counts[src] = counts.get(src, 0) + 1
    return counts


def uploaded_sources() -> dict[str, int]:
    """User-uploaded documents currently in the index."""
    return indexed_sources(origin="upload")


def static_sources() -> dict[str, int]:
    """Default ``all pdf`` knowledge-base documents currently in the index."""
    return indexed_sources(origin="static")


def ingest_paths(
    paths: list[Path],
    embeddings: HuggingFaceEmbeddings | None = None,
    origin: str = "upload",
) -> dict[str, object]:
    """Add (or replace) the given files in the live index. Returns a summary.

    Re-ingesting a file with the same name first removes its previous chunks,
    so updates are clean and never duplicated. Oversized files are recorded in
    ``summary["too_large"]`` and skipped rather than aborting the batch.
    ``origin`` tags the chunks for query-time routing (defaults to "upload").
    """
    paths = [Path(p) for p in paths if is_supported(p)]
    summary: dict[str, object] = {"added": [], "empty": [], "too_large": [], "chunks": 0}
    if not paths:
        return summary

    embeddings = embeddings or _embeddings()
    splitter = _splitter()
    vectordb = _open_chroma(embeddings)
    bm25_chunks = _load_bm25()

    # Validate sizes up front; drop offenders from the working set.
    safe_paths: list[Path] = []
    for p in paths:
        try:
            check_upload_size(p)
            safe_paths.append(p)
        except FileTooLargeError as exc:
            log.warning("%s", exc)
            summary["too_large"].append(p.name)  # type: ignore[union-attr]
    paths = safe_paths

    replaced = {p.name for p in paths}
    # Drop any prior versions of these files from both stores.
    bm25_chunks = [c for c in bm25_chunks if c.metadata.get("source") not in replaced]
    for name in replaced:
        try:
            vectordb.delete(where={"source": name})
        except Exception as exc:  # noqa: BLE001
            log.debug("chroma delete(%s) failed: %s", name, exc)

    new_chunks: list[Document] = []
    for path in paths:
        chunks = _chunk_file(path, splitter, origin)
        if not chunks:
            summary["empty"].append(path.name)  # type: ignore[union-attr]
            continue
        new_chunks.extend(chunks)
        summary["added"].append(path.name)  # type: ignore[union-attr]

    if new_chunks:
        vectordb.add_documents(
            new_chunks,
            ids=[f"{c.metadata['content_hash']}-{i}" for i, c in enumerate(new_chunks)],
        )
        bm25_chunks.extend(new_chunks)

    _save_bm25(bm25_chunks)
    _bump_version()
    summary["chunks"] = len(new_chunks)
    log.info(
        "ingest: +%d chunks from %d file(s) (empty=%d, too_large=%d)",
        len(new_chunks), len(summary["added"]), len(summary["empty"]), len(summary["too_large"]),
    )
    return summary


def remove_source(name: str, embeddings: HuggingFaceEmbeddings | None = None) -> None:
    """Delete one document (all its chunks) from both stores + the docs folder."""
    embeddings = embeddings or _embeddings()
    vectordb = _open_chroma(embeddings)
    try:
        vectordb.delete(where={"source": name})
    except Exception as exc:  # noqa: BLE001
        log.debug("chroma delete(%s) failed: %s", name, exc)

    bm25_chunks = [c for c in _load_bm25() if c.metadata.get("source") != name]
    _save_bm25(bm25_chunks)

    # best-effort: remove the saved upload file too
    f = settings.docs_dir / name
    try:
        f.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("could not unlink %s: %s", f, exc)
    _bump_version()
    log.info("removed source %s", name)


# --------------------------------------------------------------------------- #
# Full (re)build from the documents folders — CLI
# --------------------------------------------------------------------------- #
def _source_files_by_origin() -> list[tuple[Path, str]]:
    """(file, origin) pairs: ``documents/`` -> upload, legacy ``all pdf/`` -> static."""
    pairs: list[tuple[Path, str]] = []
    for folder, origin in ((settings.docs_dir, "upload"), (settings.pdf_dir, "static")):
        if folder.exists():
            pairs += [
                (p, origin)
                for p in sorted(folder.glob("*"))
                if p.is_file() and is_supported(p)
            ]
    return pairs


def rebuild(wipe: bool = True) -> dict[str, object]:
    """Wipe (optional) and rebuild the whole index from the source folders.

    ``documents/`` files are tagged origin="upload" and ``all pdf/``
    origin="static", so query-time routing works after a fresh rebuild.
    """
    if wipe:
        if settings.chroma_dir.exists():
            shutil.rmtree(settings.chroma_dir)
        if settings.index_dir.exists():
            shutil.rmtree(settings.index_dir)

    files = _source_files_by_origin()
    if not files:
        sys.exit(f"[ingest] No supported files found in {settings.docs_dir} or {settings.pdf_dir}")

    from langchain_chroma import Chroma

    embeddings = _embeddings()
    splitter = _splitter()

    all_chunks: list[Document] = []
    empty: list[str] = []
    for path, origin in files:
        chunks = _chunk_file(path, splitter, origin)
        if not chunks:
            log.warning("NO TEXT extracted from %s (image-only/corrupt) -> skipped", path.name)
            empty.append(path.name)
            continue
        all_chunks.extend(chunks)
        log.info("+ %-55s -> %3d chunks", path.name, len(chunks))

    if not all_chunks:
        sys.exit("[ingest] No content extracted from any file. Aborting.")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    log.info("embedding + writing Chroma ...")
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        ids=[f"{c.metadata['content_hash']}-{i}" for i, c in enumerate(all_chunks)],
    )
    log.info("Chroma persisted -> %s (%d vectors)", settings.chroma_dir, vectordb._collection.count())
    _save_bm25(all_chunks)
    _bump_version()
    log.info("BM25 chunk store -> %s", settings.bm25_path)
    log.info("total chunks: %d from %d/%d files", len(all_chunks), len(files) - len(empty), len(files))
    if empty:
        log.warning("these files gave no text and were skipped: %s", ", ".join(empty))
    return {"chunks": len(all_chunks), "empty": empty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG indexes.")
    parser.add_argument("--rebuild", action="store_true", help="wipe and rebuild")
    args = parser.parse_args()
    rebuild(wipe=args.rebuild)
    log.info("done -> run:  streamlit run app.py")


if __name__ == "__main__":
    main()
