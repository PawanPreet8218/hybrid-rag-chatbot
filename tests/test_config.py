"""Config validation + derived-path behaviour."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sage.config import Settings, settings


def test_singleton_has_sane_defaults():
    assert settings.app_name == "Sage"
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.dense_search_type in {"mmr", "similarity", "similarity_score_threshold"}


def test_derived_paths_hang_off_data_dir(tmp_path: Path):
    s = Settings(data_dir=tmp_path)
    assert s.docs_dir == tmp_path / "documents"
    assert s.chroma_dir == tmp_path / "chroma_db"
    assert s.bm25_path == tmp_path / "indexes" / "bm25_docs.pkl"
    assert s.chats_dir == tmp_path / "chats"


def test_invalid_search_type_rejected():
    with pytest.raises(ValidationError):
        Settings(dense_search_type="banana")


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_invalid_log_level_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="LOUD")


def test_env_override(monkeypatch):
    monkeypatch.setenv("SAGE_CHUNK_SIZE", "1234")
    monkeypatch.setenv("SAGE_LLM_MODEL", "some-other-model")
    s = Settings()
    assert s.chunk_size == 1234
    assert s.llm_model == "some-other-model"


def test_groq_key_uses_unprefixed_name(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-123")
    s = Settings()
    assert s.groq_api_key == "secret-123"
