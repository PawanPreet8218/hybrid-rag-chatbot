"""Shared pytest fixtures.

These tests deliberately avoid downloading any ML model — they exercise the
pure logic (config, IO safety, loaders, chat store, ingest helpers) so the suite
runs fast and offline in CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sage.config import settings


@pytest.fixture()
def data_dir(tmp_path: Path):
    """Point all of Sage's persisted state at an isolated temp directory.

    The settings singleton is mutated for the duration of the test and restored
    afterwards, so chat_store / ingest read & write inside ``tmp_path``.
    """
    original = settings.data_dir
    settings.data_dir = tmp_path
    yield tmp_path
    settings.data_dir = original
