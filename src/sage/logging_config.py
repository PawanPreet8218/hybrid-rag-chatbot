"""Centralised logging setup.

Call :func:`configure_logging` once at process start (entry points do this) and
grab a module logger anywhere with :func:`get_logger`. Supports human-readable
console logs (default) or single-line JSON logs (``SAGE_LOG_JSON=true``) for
ingestion into a log aggregator in production.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from sage.config import settings

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Minimal, dependency-free structured formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, *, json_logs: bool | None = None) -> None:
    """Idempotently configure the root logger for the whole app."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = (level or settings.log_level).upper()
    json_logs = settings.log_json if json_logs is None else json_logs

    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party libraries.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "chromadb", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
