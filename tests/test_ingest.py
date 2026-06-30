"""Ingest helpers — section parsing, hashing, size limits, BM25 store IO.

The heavy embed/Chroma path needs model downloads, so it's out of scope here;
we cover the pure logic and the atomic BM25 persistence that backs retrieval.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from sage import ingest
from sage.config import settings
from sage.exceptions import FileTooLargeError


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("ACTUALISAT-LLM _ 03 Core Principles-090725-070353.pdf", "Core Principles"),
        ("ACTUALISAT-LLM _ 01_WhatIsActualisation-090725-070257.pdf", "WhatIsActualisation"),
        ("Q3 Financials.xlsx", "Q3 Financials"),
        ("simple.txt", "simple"),
    ],
)
def test_section_from_filename(filename, expected):
    assert ingest._section_from_filename(filename) == expected


def test_content_hash_is_stable_and_short():
    h1 = ingest._content_hash("hello")
    h2 = ingest._content_hash("hello")
    assert h1 == h2 and len(h1) == 16
    assert ingest._content_hash("hello") != ingest._content_hash("world")


def test_check_upload_size_ok(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    p = tmp_path / "small.txt"
    p.write_bytes(b"x" * 1024)  # 1 KB
    ingest.check_upload_size(p)  # should not raise


def test_check_upload_size_too_large(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
    with pytest.raises(FileTooLargeError):
        ingest.check_upload_size(p)


def test_bm25_store_roundtrip(data_dir: Path):
    docs = [Document(page_content="a", metadata={"source": "f.txt", "origin": "upload"})]
    ingest._save_bm25(docs)
    loaded = ingest._load_bm25()
    assert len(loaded) == 1 and loaded[0].page_content == "a"


def test_load_bm25_missing_is_empty(data_dir: Path):
    assert ingest._load_bm25() == []


def test_indexed_sources_counts_by_origin(data_dir: Path):
    docs = [
        Document(page_content="a", metadata={"source": "up.txt", "origin": "upload"}),
        Document(page_content="b", metadata={"source": "up.txt", "origin": "upload"}),
        Document(page_content="c", metadata={"source": "kb.pdf", "origin": "static"}),
    ]
    ingest._save_bm25(docs)
    assert ingest.uploaded_sources() == {"up.txt": 2}
    assert ingest.static_sources() == {"kb.pdf": 1}


def test_bump_version_increments(data_dir: Path):
    ingest._bump_version()
    first = int(settings.version_path.read_text())
    ingest._bump_version()
    assert int(settings.version_path.read_text()) == first + 1
