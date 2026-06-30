"""Hybrid retriever (ONLINE).

Pipeline::

    query
      ├─ dense  : Chroma similarity / MMR  (semantic)   ← "relevant kwargs" live here
      ├─ sparse : BM25                      (keyword)
      ├─ fuse   : EnsembleRetriever (Reciprocal Rank Fusion, weighted)
      └─ rerank : bge-reranker cross-encoder -> top_n   (precision)

This single retriever object is the ONE source of truth: the same docs that
the LLM reads are the ones shown to the user as citations.

Heavy models (embeddings, cross-encoder) can be passed in so the app loads
them ONCE and reuses them across re-indexing — see ``sage.ui`` caching.
"""
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from sage.config import settings
from sage.logging_config import get_logger

if TYPE_CHECKING:  # heavy ML deps are imported lazily inside functions
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_huggingface import HuggingFaceEmbeddings

log = get_logger(__name__)


class _EmptyRetriever(BaseRetriever):
    """Stand-in used before any document has been indexed."""

    def _get_relevant_documents(self, query, *, run_manager=None):  # noqa: D401, ANN001
        return []


def load_embeddings() -> HuggingFaceEmbeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_cross_encoder() -> HuggingFaceCrossEncoder:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=settings.reranker_model)


def _load_chunks() -> list[Document]:
    if not settings.bm25_path.exists():
        return []
    try:
        with open(settings.bm25_path, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        log.error("BM25 store unreadable (%s); treating as empty", exc)
        return []


def _dense_retriever(
    embeddings: HuggingFaceEmbeddings, origin: str | None = None
) -> BaseRetriever:
    from langchain_chroma import Chroma

    vectordb = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
    )
    if settings.dense_search_type == "mmr":
        search_kwargs = {
            "k": settings.dense_k,
            "fetch_k": settings.dense_fetch_k,
            "lambda_mult": settings.mmr_lambda,
        }
    elif settings.dense_search_type == "similarity_score_threshold":
        search_kwargs = {"k": settings.dense_k, "score_threshold": settings.score_threshold}
    else:
        search_kwargs = {"k": settings.dense_k}

    if origin is not None:  # scope dense search to one corpus
        search_kwargs["filter"] = {"origin": origin}

    return vectordb.as_retriever(
        search_type=settings.dense_search_type,
        search_kwargs=search_kwargs,
    )


def has_index() -> bool:
    """True once at least one document has been ingested."""
    return bool(_load_chunks())


def build_retriever(
    embeddings: HuggingFaceEmbeddings | None = None,
    cross_encoder: HuggingFaceCrossEncoder | None = None,
    origin: str | None = None,
) -> BaseRetriever:
    """Assemble dense + sparse -> RRF fusion -> cross-encoder rerank.

    ``origin`` scopes retrieval to one corpus ("upload" or "static"); None
    searches everything. Returns an empty retriever (answers from no context)
    when nothing matches, so the app can boot before any upload.
    """
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.retrievers import BM25Retriever

    chunks = _load_chunks()
    if origin is not None:  # scope BM25 to the same corpus as the dense side
        chunks = [c for c in chunks if c.metadata.get("origin") == origin]
    if not chunks:
        return _EmptyRetriever()

    embeddings = embeddings or load_embeddings()
    cross_encoder = cross_encoder or load_cross_encoder()

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.bm25_k

    ensemble = EnsembleRetriever(
        retrievers=[bm25, _dense_retriever(embeddings, origin)],
        weights=list(settings.hybrid_weights),  # [sparse, dense]
    )

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )


if __name__ == "__main__":  # quick smoke test
    r = build_retriever()
    q = "What is this about?"
    docs = r.invoke(q)
    print(f"\nQuery: {q}\nTop {len(docs)} reranked chunks:")
    for i, d in enumerate(docs, 1):
        print(f" {i}. [{d.metadata.get('section')} p.{d.metadata.get('page')}] {d.page_content[:90]}...")
