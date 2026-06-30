"""Multi-format document loaders.

One entry point — :func:`load_documents` — turns any supported file
(PDF / Word / Excel / CSV / text / markdown) into a list of LangChain
``Document``s with consistent metadata: ``{source, page}``. The ingest pipeline
adds the rest (section, chunk_id, content_hash).

Each loader is deliberately dependency-light so the app stays fast and portable,
and is imported lazily so a missing optional dependency only fails the one format
that needs it.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from sage.logging_config import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Individual format loaders
# --------------------------------------------------------------------------- #
def _load_pdf(path: Path) -> list[Document]:
    """Layout-aware PDF parsing (PyMuPDF), with a pypdf fallback."""
    from langchain_community.document_loaders import PyMuPDFLoader

    try:
        pages = PyMuPDFLoader(str(path)).load()
        if any((p.page_content or "").strip() for p in pages):
            return pages
    except Exception:  # noqa: BLE001 - fall through to the alternate backend
        pages = []

    # Fallback: pypdf (sometimes succeeds where PyMuPDF chokes, and vice-versa)
    try:
        from pypdf import PdfReader

        docs: list[Document] = []
        for i, page in enumerate(PdfReader(str(path)).pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i}))
        return docs
    except Exception:  # noqa: BLE001
        return pages


def _load_docx(path: Path) -> list[Document]:
    """Word documents via docx2txt (plain text, no formatting)."""
    import docx2txt

    text = docx2txt.process(str(path)) or ""
    return [Document(page_content=text, metadata={"page": 0})] if text.strip() else []


def _load_excel(path: Path) -> list[Document]:
    """Each worksheet becomes its own Document (better retrieval granularity)."""
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, dtype=str)  # dict[name -> DataFrame]
    docs: list[Document] = []
    for i, (name, df) in enumerate(sheets.items()):
        df = df.fillna("")
        if df.empty:
            continue
        body = df.to_csv(index=False)  # keeps column context for the LLM
        text = f"Sheet: {name}\n{body}"
        docs.append(Document(page_content=text, metadata={"page": i, "sheet": name}))
    return docs


def _load_csv(path: Path) -> list[Document]:
    import pandas as pd

    df = pd.read_csv(path, dtype=str).fillna("")
    if df.empty:
        return []
    return [Document(page_content=df.to_csv(index=False), metadata={"page": 0})]


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"page": 0})] if text.strip() else []


_DISPATCH = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,
    ".xlsx": _load_excel,
    ".xls": _load_excel,
    ".csv": _load_csv,
    ".txt": _load_text,
    ".md": _load_text,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _DISPATCH


def load_documents(path: str | Path) -> list[Document]:
    """Load any supported file into Documents tagged with source + page.

    Returns ``[]`` (never raises) when a file is unsupported, empty, or
    unreadable — the caller decides how loudly to flag it.
    """
    path = Path(path)
    loader = _DISPATCH.get(path.suffix.lower())
    if loader is None:
        log.warning("unsupported file type, skipping: %s", path.name)
        return []

    try:
        docs = loader(path)
    except Exception as exc:  # noqa: BLE001 - keep one bad file from killing a batch
        log.error("could not read %s: %s", path.name, exc)
        return []

    # 1-based page numbers for humans + always stamp the source filename.
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
    return docs
