"""Central, validated configuration for Sage.

Every tunable knob lives here as a single :class:`Settings` model (powered by
``pydantic-settings``) so the rest of the code reads cleanly, values are
type-checked at startup, and anything can be overridden by an environment
variable or ``.env`` entry without touching code.

Conventions
-----------
* All app settings use the ``SAGE_`` prefix, e.g. ``SAGE_LLM_MODEL``,
  ``SAGE_CHUNK_SIZE``, ``SAGE_DATA_DIR``.
* ``GROQ_API_KEY`` keeps its conventional (unprefixed) name.
* Persisted state (vector store, BM25 index, chats, uploads) lives under
  :attr:`Settings.data_dir` which defaults to the repository root but can be
  pointed at a mounted volume in Docker via ``SAGE_DATA_DIR``.

Import the ready-to-use singleton:

    from sage import settings
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/sage/config.py -> src/sage -> src -> <repo root>
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings, overridable via env / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="SAGE_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----------------------------------------------------------------- #
    # Branding  (rename the whole app by setting SAGE_APP_NAME)
    # ----------------------------------------------------------------- #
    app_name: str = "Sage"
    app_tagline: str = "Your personal document assistant"
    app_icon: str = "📚"
    user: str = "Gurri"          # who the profile pill greets  (SAGE_USER)
    user_plan: str = "Free"

    # ----------------------------------------------------------------- #
    # Storage root  (all persisted state hangs off here -> portable)
    # ----------------------------------------------------------------- #
    data_dir: Path = PROJECT_ROOT

    # ----------------------------------------------------------------- #
    # Models
    # ----------------------------------------------------------------- #
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # 768-dim, strong CPU model
    reranker_model: str = "BAAI/bge-reranker-base"                    # local cross-encoder
    llm_model: str = "llama-3.3-70b-versatile"                        # Groq, fast + free
    llm_temperature: float = 0.1                                      # low -> grounded, factual
    device: str = "cpu"
    collection_name: str = "actualisation_docs"

    # ----------------------------------------------------------------- #
    # Chunking
    # ----------------------------------------------------------------- #
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ----------------------------------------------------------------- #
    # Retrieval
    # ----------------------------------------------------------------- #
    # "mmr" (diversity) | "similarity" | "similarity_score_threshold"
    dense_search_type: str = "mmr"
    dense_k: int = 10            # final dense candidates
    dense_fetch_k: int = 25      # MMR pool before diversity selection
    mmr_lambda: float = 0.5      # 0 = max diversity, 1 = max relevance
    score_threshold: float = 0.2  # only for similarity_score_threshold
    bm25_k: int = 10             # sparse candidates
    # Hybrid fusion weights for EnsembleRetriever (RRF): [sparse, dense]
    hybrid_weights: tuple[float, float] = (0.4, 0.6)
    rerank_top_n: int = 5        # docs passed to the LLM after rerank
    rerank_min_score: float = -4.0  # below this logit -> "no good context"

    # ----------------------------------------------------------------- #
    # Upload safety limits
    # ----------------------------------------------------------------- #
    max_upload_mb: int = 50      # reject single uploads larger than this

    # ----------------------------------------------------------------- #
    # Observability
    # ----------------------------------------------------------------- #
    log_level: str = "INFO"
    log_json: bool = False       # set True for structured JSON logs (prod)

    # ----------------------------------------------------------------- #
    # Secrets  (conventional unprefixed name)
    # ----------------------------------------------------------------- #
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    # ----------------------------------------------------------------- #
    # Supported upload formats  (extension -> human label)
    # ----------------------------------------------------------------- #
    supported_exts: dict[str, str] = {
        ".pdf": "PDF",
        ".docx": "Word",
        ".doc": "Word",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".csv": "CSV",
        ".txt": "Text",
        ".md": "Markdown",
    }

    # ----------------------------------------------------------------- #
    # Validators
    # ----------------------------------------------------------------- #
    @field_validator("dense_search_type")
    @classmethod
    def _check_search_type(cls, v: str) -> str:
        allowed = {"mmr", "similarity", "similarity_score_threshold"}
        if v not in allowed:
            raise ValueError(f"dense_search_type must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, v: int, info) -> int:  # noqa: ANN001
        size = info.data.get("chunk_size", 0)
        if size and v >= size:
            raise ValueError(f"chunk_overlap ({v}) must be smaller than chunk_size ({size})")
        return v

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log_level: {v}")
        return v

    # ----------------------------------------------------------------- #
    # Derived paths  (computed from data_dir -> never machine-specific)
    # ----------------------------------------------------------------- #
    @computed_field  # type: ignore[prop-decorator]
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "documents"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pdf_dir(self) -> Path:
        """Legacy static knowledge-base folder (still ingested if present)."""
        return self.data_dir / "all pdf"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25_docs.pkl"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version_path(self) -> Path:
        return self.index_dir / "version.txt"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chats_dir(self) -> Path:
        return self.data_dir / "chats"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()


# Eager singleton for convenient `from sage import settings`.
settings = get_settings()
