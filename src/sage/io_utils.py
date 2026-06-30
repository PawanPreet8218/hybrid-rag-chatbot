"""Small, well-tested IO helpers shared across the package.

These exist so persistence is *crash-safe* (atomic writes — a power cut never
leaves a half-written index or chat) and *secure* (uploaded filenames can never
escape the documents directory).
"""
from __future__ import annotations

import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

from sage.exceptions import UnsafeFilenameError

# Characters that are unsafe or awkward in filenames across OSes.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str) -> str:
    """Reduce an arbitrary upload name to a safe, basename-only filename.

    Strips any directory components (defeating ``../../etc/passwd`` style path
    traversal) and replaces unsafe characters. Raises
    :class:`UnsafeFilenameError` if nothing usable remains.
    """
    # Take the basename only — kills path traversal regardless of separator.
    base = os.path.basename(name.replace("\\", "/")).strip()
    base = _UNSAFE.sub("_", base).strip(". ")
    if not base or base in {".", ".."}:
        raise UnsafeFilenameError(f"unsafe or empty filename: {name!r}")
    return base


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` atomically (temp file in same dir + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX and Windows
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, obj: Any) -> None:
    """Serialise ``obj`` to JSON and write atomically."""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def atomic_write_pickle(path: Path, obj: Any) -> None:
    """Pickle ``obj`` and write atomically."""
    atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
