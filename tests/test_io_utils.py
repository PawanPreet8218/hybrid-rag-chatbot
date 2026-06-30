"""Filename safety + atomic-write helpers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from sage.exceptions import UnsafeFilenameError
from sage.io_utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_pickle,
    safe_filename,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\evil.dll", "evil.dll"),
        ("my doc (final).docx", "my doc (final).docx"),
        ("weird:name?.txt", "weird_name_.txt"),
        ("/abs/path/file.csv", "file.csv"),
    ],
)
def test_safe_filename_strips_paths_and_unsafe_chars(raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "..", ".", "/", "\\", "..."])
def test_safe_filename_rejects_empty_or_traversal(bad):
    with pytest.raises(UnsafeFilenameError):
        safe_filename(bad)


def test_atomic_write_bytes_roundtrip(tmp_path: Path):
    p = tmp_path / "nested" / "blob.bin"
    atomic_write_bytes(p, b"hello")
    assert p.read_bytes() == b"hello"
    # No leftover temp files in the directory.
    assert [x.name for x in p.parent.iterdir()] == ["blob.bin"]


def test_atomic_write_json_roundtrip(tmp_path: Path):
    p = tmp_path / "data.json"
    atomic_write_json(p, {"a": 1, "b": ["x", "y"]})
    assert json.loads(p.read_text()) == {"a": 1, "b": ["x", "y"]}


def test_atomic_write_pickle_roundtrip(tmp_path: Path):
    p = tmp_path / "data.pkl"
    atomic_write_pickle(p, {"k": [1, 2, 3]})
    with open(p, "rb") as fh:
        assert pickle.load(fh) == {"k": [1, 2, 3]}


def test_atomic_write_overwrites_existing(tmp_path: Path):
    p = tmp_path / "f.txt"
    atomic_write_bytes(p, b"first")
    atomic_write_bytes(p, b"second")
    assert p.read_bytes() == b"second"
