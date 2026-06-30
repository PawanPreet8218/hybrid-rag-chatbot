"""Persistent chat store — CRUD, pruning, robustness."""
from __future__ import annotations

from pathlib import Path

from sage import chat_store


def test_new_and_load_roundtrip(data_dir: Path):
    chat = chat_store.new_chat()
    loaded = chat_store.load_chat(chat["id"])
    assert loaded is not None
    assert loaded["id"] == chat["id"]
    assert loaded["messages"] == []


def test_save_and_list(data_dir: Path):
    chat = chat_store.new_chat()
    chat["messages"].append({"role": "user", "content": "hi"})
    chat_store.save_chat(chat)

    metas = chat_store.list_chats()
    assert any(m["id"] == chat["id"] and m["n_messages"] == 1 for m in metas)


def test_rename(data_dir: Path):
    chat = chat_store.new_chat()
    chat_store.rename_chat(chat["id"], "  My Title  ")
    assert chat_store.load_chat(chat["id"])["title"] == "My Title"


def test_rename_empty_falls_back(data_dir: Path):
    chat = chat_store.new_chat()
    chat_store.rename_chat(chat["id"], "   ")
    assert chat_store.load_chat(chat["id"])["title"] == "Untitled"


def test_delete(data_dir: Path):
    chat = chat_store.new_chat()
    chat_store.delete_chat(chat["id"])
    assert chat_store.load_chat(chat["id"]) is None


def test_prune_empty_keeps_active(data_dir: Path):
    keep = chat_store.new_chat()
    drop = chat_store.new_chat()
    chat_store.prune_empty(keep_id=keep["id"])
    assert chat_store.load_chat(keep["id"]) is not None
    assert chat_store.load_chat(drop["id"]) is None


def test_prune_keeps_chats_with_messages(data_dir: Path):
    chat = chat_store.new_chat()
    chat["messages"].append({"role": "user", "content": "x"})
    chat_store.save_chat(chat)
    chat_store.prune_empty(keep_id="nonexistent")
    assert chat_store.load_chat(chat["id"]) is not None


def test_load_missing_returns_none(data_dir: Path):
    assert chat_store.load_chat("does-not-exist") is None


def test_corrupt_file_is_skipped(data_dir: Path):
    chat_store.new_chat()  # ensures dir exists
    bad = data_dir / "chats" / "corrupt.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    # Neither call should raise.
    assert chat_store.load_chat("corrupt") is None
    chat_store.list_chats()


def test_auto_title_truncates():
    long = "word " * 50
    title = chat_store.auto_title(long, limit=20)
    assert len(title) <= 21 and title.endswith("…")
    assert chat_store.auto_title("") == "New chat"
