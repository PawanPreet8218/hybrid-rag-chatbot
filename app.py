"""Streamlit entry point for Sage.

Run from the repo root::

    streamlit run app.py

This file is intentionally thin: it makes the ``src/`` layout importable (so the
app works even without an editable install) and hands off to ``sage.ui.main``.
All real logic lives in the :mod:`sage` package under ``src/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable so `import sage` works whether or not the package was
# installed with `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sage.ui import main  # noqa: E402  (path bootstrap must run first)

main()
