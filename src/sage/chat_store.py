"""Persistent chat sessions — the "history sidebar" backend (Claude-style).

Each conversation is one JSON file in :attr:`settings.chats_dir`::

    {
      "id": "<uuid>",
      "title": "New chat",
      "created": 1700000000.0,
      "updated": 1700000000.0,
      "messages": [ {role, content, citations}, ... ],   # for display
      "history":  [ ["human", "..."], ["ai", "..."] ]     # LLM-facing memory
    }

Chats survive app restarts and are written atomically (no half-written file on a
crash). Functions never raise on a missing/corrupt file — they just skip it — so
one bad chat can't break the sidebar.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from sage.config import settings
from sage.io_utils import atomic_write_json
from sage.logging_config import get_logger

log = get_logger(__name__)


def _dir() -> Path:
    settings.chats_dir.mkdir(parents=True, exist_ok=True)
    return settings.chats_dir


def _path(chat_id: str) -> Path:
    # chat_id is generated internally (uuid hex) but guard anyway: basename only.
    safe_id = Path(chat_id).name
    return _dir() / f"{safe_id}.json"


def new_chat(title: str = "New chat") -> dict:
    now = time.time()
    chat = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created": now,
        "updated": now,
        "messages": [],
        "history": [],
    }
    save_chat(chat)
    return chat


def save_chat(chat: dict) -> None:
    chat["updated"] = time.time()
    atomic_write_json(_path(chat["id"]), chat)


def load_chat(chat_id: str) -> dict | None:
    p = _path(chat_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("skipping unreadable chat %s: %s", chat_id, exc)
        return None


def list_chats() -> list[dict]:
    """Lightweight metadata for every chat, newest first."""
    out: list[dict] = []
    for p in _dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append(
                {
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "updated": data.get("updated", 0),
                    "n_messages": len(data.get("messages", [])),
                }
            )
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return sorted(out, key=lambda c: c["updated"], reverse=True)


def prune_empty(keep_id: str | None = None) -> None:
    """Delete every chat that has no messages, except ``keep_id``.

    Keeps the sidebar clean: a blank chat only lives on disk while it's the
    active one — once you move on without sending anything, it's removed.
    """
    for meta in list_chats():
        if meta["id"] != keep_id and meta["n_messages"] == 0:
            delete_chat(meta["id"])


def rename_chat(chat_id: str, title: str) -> None:
    chat = load_chat(chat_id)
    if chat is not None:
        chat["title"] = title.strip() or "Untitled"
        save_chat(chat)


def delete_chat(chat_id: str) -> None:
    _path(chat_id).unlink(missing_ok=True)


def auto_title(text: str, limit: int = 40) -> str:
    """Derive a chat title from the first user message."""
    title = " ".join(text.strip().split())
    return (title[:limit] + "…") if len(title) > limit else (title or "New chat")
