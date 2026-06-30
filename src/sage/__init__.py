"""Sage — a personal, multi-format Retrieval-Augmented-Generation assistant.

Public package surface. Importing :data:`settings` from here is the canonical
way to read configuration; everything else is reachable through the submodules
(:mod:`sage.ingest`, :mod:`sage.retriever`, :mod:`sage.rag_chain`, ...).
"""
from __future__ import annotations

from sage.config import settings

__all__ = ["settings", "__version__"]

__version__ = "1.0.0"
