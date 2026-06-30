"""Multi-format loader behaviour (text-based formats only — no model downloads)."""
from __future__ import annotations

from pathlib import Path

from sage.loaders import is_supported, load_documents


def test_is_supported():
    assert is_supported("a.pdf")
    assert is_supported(Path("b.DOCX"))   # case-insensitive
    assert is_supported("c.md")
    assert not is_supported("d.exe")
    assert not is_supported("noext")


def test_load_txt(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    docs = load_documents(p)
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "note.txt"
    assert docs[0].metadata["page"] == 1   # 1-based


def test_load_markdown(tmp_path: Path):
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\nbody", encoding="utf-8")
    docs = load_documents(p)
    assert docs and "Title" in docs[0].page_content


def test_load_csv(tmp_path: Path):
    p = tmp_path / "rows.csv"
    p.write_text("name,age\nAda,36\nLin,29\n", encoding="utf-8")
    docs = load_documents(p)
    assert len(docs) == 1
    assert "Ada" in docs[0].page_content
    assert docs[0].metadata["source"] == "rows.csv"


def test_empty_text_returns_nothing(tmp_path: Path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n  ", encoding="utf-8")
    assert load_documents(p) == []


def test_unsupported_returns_empty(tmp_path: Path):
    p = tmp_path / "thing.exe"
    p.write_bytes(b"\x00\x01")
    assert load_documents(p) == []


def test_loader_never_raises_on_bad_file(tmp_path: Path):
    # A .csv that is actually garbage should be swallowed, not crash.
    p = tmp_path / "broken.csv"
    p.write_bytes(b"\xff\xfe\x00not,valid")
    # Should return either [] or some docs, but never raise.
    load_documents(p)
