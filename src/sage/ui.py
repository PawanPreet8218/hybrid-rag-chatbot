"""Sage — your personal RAG document assistant (Streamlit UI).

A Claude-style interface over your own files:
  • Upload via the chat box "+" (or drag-and-drop) — indexed on the fly.
  • Chat history sidebar with rename / delete, persisted across restarts.
  • Grounded answers with citations back to the exact source + page.

Routing: if you've uploaded documents, answers come from THOSE only; otherwise
Sage falls back to the default knowledge base (the ``all pdf`` folder).

Run (from the repo root)::

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from sage import chat_store, ingest
from sage.config import settings
from sage.exceptions import ConfigError, SageError
from sage.io_utils import safe_filename
from sage.logging_config import get_logger
from sage.rag_chain import build_rag_chain, format_citations, load_llm
from sage.retriever import build_retriever, has_index, load_cross_encoder, load_embeddings

log = get_logger(__name__)

_UPLOAD_TYPES = sorted({e.lstrip(".") for e in settings.supported_exts})


# --------------------------------------------------------------------------- #
# Claude-style look & feel (injected CSS)
# --------------------------------------------------------------------------- #
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

/* Inter via inheritance so it never clobbers Streamlit's Material icon font. */
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select, .stApp label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

/* Narrow centered chat column */
.block-container { max-width: 46rem; padding-top: 2.5rem; padding-bottom: 9rem; }

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] { background: transparent; padding: 0.35rem 0; gap: 0.85rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #F0EBE4; border: 1px solid #E8E0D5; border-radius: 16px; padding: 0.4rem 1.1rem;
}
[data-testid="stChatMessage"] p { line-height: 1.65; }

/* ---- Chat input: big rounded card like Claude ---- */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] {
    border-radius: 26px; border: 1px solid #E4DCCF;
    box-shadow: 0 8px 34px rgba(80, 60, 40, 0.12);
    background: #FFFFFF; padding: 0.45rem 0.5rem;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #CC785C; box-shadow: 0 8px 34px rgba(204, 120, 92, 0.18);
}
[data-testid="stChatInput"] textarea { font-size: 1.05rem; min-height: 3.2rem; padding-top: 0.7rem; }
/* "+" attach button + send button -> coral round */
[data-testid="stChatInputSubmitButton"] { background: #CC785C !important; border-radius: 50% !important; color: #fff !important; }
[data-testid="stChatInputSubmitButton"]:hover { background: #B5654A !important; }

/* When the chat is empty, lift the input toward the vertical centre (Claude). */
[data-testid="stApp"]:has(#sage-empty-marker) [data-testid="stBottom"] {
    top: 50%; bottom: auto; transform: translateY(10px); height: auto;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] { background: #F3F0E9; border-right: 1px solid #E7E1D6; }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }

/* Decorative top tabs */
.sage-tabs { display:flex; gap:4px; background:#E7E1D5; padding:4px; border-radius:12px; margin-bottom:1rem; }
.sage-tab { flex:1; text-align:center; padding:6px 4px; border-radius:9px; font-size:0.82rem; color:#7A756C; font-weight:500; }
.sage-tab.active { background:#FFFFFF; color:#1F1E1B; box-shadow:0 1px 3px rgba(0,0,0,0.06); }

/* Nav + chat-list rows -> borderless, left-aligned, hover highlight */
[class*="st-key-nav_"] button, [class*="st-key-open_"] button {
    background: transparent !important; border: none !important;
    text-align: left !important; justify-content: flex-start !important;
    color: #3C3A35 !important; font-weight: 500 !important;
    padding: 0.4rem 0.6rem !important; border-radius: 9px !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
[class*="st-key-nav_"] button:hover, [class*="st-key-open_"] button:hover {
    background: #E5DECF !important; color: #1F1E1B !important;
}
[class*="st-key-open_"] button { font-size: 0.9rem; color: #524F49 !important; }

.sage-recents-h { font-size:0.78rem; color:#938E84; font-weight:600; margin:1rem 0 0.2rem 0.4rem; }

/* The "⋯" row menu -> quiet, borderless until hovered (like Claude) */
[data-testid="stSidebar"] [data-testid="stPopover"] button {
    border: none !important; background: transparent !important;
    color: #B0AAA0 !important; box-shadow: none !important;
    padding: 0.25rem 0.3rem !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
    background: #E5DECF !important; color: #524F49 !important;
}

/* expander chips */
[data-testid="stExpander"] { border: 1px solid #EBE6DC; border-radius: 12px; background: #FCFBF8; }

/* ---- Hero greeting (empty state) ---- */
.sage-hero { display:flex; align-items:center; justify-content:center; gap:0.7rem; margin: 22vh 0 1.2rem 0; }
.sage-spark { color:#CC785C; font-size:2.1rem; line-height:1; animation: sage-spin 14s linear infinite; }
@keyframes sage-spin { to { transform: rotate(360deg); } }
.sage-greeting { font-family:'Newsreader', Georgia, serif; font-size:2.6rem; font-weight:400; color:#2B2A27; letter-spacing:-0.01em; }
@media (max-width: 640px) { .sage-greeting { font-size:1.9rem; } }
</style>
"""

_AVATARS = {"assistant": settings.app_icon, "user": "🧑"}


# --------------------------------------------------------------------------- #
# Cached resources — models load once, chain rebuilds only when index changes
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embeddings, reranker and LLM …")
def get_models():
    if not settings.groq_api_key:
        st.error("`GROQ_API_KEY` is not set. Add it to your `.env` file, then refresh.")
        st.stop()
    return load_embeddings(), load_cross_encoder(), load_llm()


def _index_version() -> int:
    if settings.version_path.exists():
        try:
            return int(settings.version_path.read_text().strip() or 0)
        except ValueError:
            return 0
    return 0


@st.cache_resource(show_spinner="Loading your knowledge base …")
def get_chain(version: int, scope: str):
    """Rebuilt whenever ``version`` (bumped on ingest) or ``scope`` changes."""
    embeddings, cross_encoder, llm = get_models()
    retriever = build_retriever(embeddings=embeddings, cross_encoder=cross_encoder, origin=scope)
    return build_rag_chain(llm=llm, retriever=retriever)


def _active_scope() -> str:
    """Uploaded docs answer on their own; otherwise use the static corpus."""
    return "upload" if ingest.uploaded_sources() else "static"


# --------------------------------------------------------------------------- #
# Session bootstrap
# --------------------------------------------------------------------------- #
def _ensure_current_chat() -> None:
    """Guarantee a current chat, reusing an existing blank one instead of
    spawning duplicates, then prune any other empty chats."""
    if "chat_id" not in st.session_state or not chat_store.load_chat(st.session_state.chat_id):
        empties = [c for c in chat_store.list_chats() if c["n_messages"] == 0]
        chat_id = empties[0]["id"] if empties else chat_store.new_chat()["id"]
        st.session_state.chat_id = chat_id
    chat_store.prune_empty(keep_id=st.session_state.chat_id)


def _select_chat(chat_id: str) -> None:
    st.session_state.chat_id = chat_id


def _start_new_chat() -> None:
    """New chat — but if the current one is already blank, just stay on it
    (matches Claude: 'New chat' does nothing when you're on an empty chat)."""
    current = chat_store.load_chat(st.session_state.chat_id)
    if not (current and current["messages"]):
        return
    chat_store.prune_empty(keep_id=st.session_state.chat_id)
    st.session_state.chat_id = chat_store.new_chat()["id"]


def _ingest_files(files) -> dict:
    """Save + index files attached in the chat box (origin='upload').

    Filenames are sanitised (no path traversal) and oversized files are
    reported back to the user instead of being indexed.
    """
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rejected: list[str] = []
    for uf in files:
        try:
            name = safe_filename(uf.name)
        except SageError:
            rejected.append(uf.name)
            continue
        dest = settings.docs_dir / name
        dest.write_bytes(uf.getbuffer())
        saved.append(dest)

    embeddings, _, _ = get_models()
    with st.spinner(f"Indexing {len(saved)} file(s)…"):
        summary = ingest.ingest_paths(saved, embeddings=embeddings, origin="upload")

    if summary["added"]:
        st.toast(f"✅ Added {len(summary['added'])} document(s) to context.")
    if summary["empty"]:
        st.warning("No readable text in: " + ", ".join(summary["empty"]))
    if summary["too_large"]:
        st.warning(
            f"Skipped (over {settings.max_upload_mb} MB): " + ", ".join(summary["too_large"])
        )
    if rejected:
        st.warning("Rejected unsafe filename(s): " + ", ".join(rejected))
    return summary


# --------------------------------------------------------------------------- #
# Sidebar — Claude layout: tab, new chat, recents, profile
# --------------------------------------------------------------------------- #
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='sage-tabs'><div class='sage-tab active'>💬 Chat</div></div>",
            unsafe_allow_html=True,
        )

        if st.button("＋  New chat", key="nav_newchat", use_container_width=True):
            _start_new_chat()
            st.rerun()

        st.markdown("<div class='sage-recents-h'>Recents</div>", unsafe_allow_html=True)
        _render_chat_list()


def _render_chat_list() -> None:
    # Only chats that actually have messages — the active blank chat is hidden
    # until you send something (like Claude), so no "New chat" clutter.
    chats = [c for c in chat_store.list_chats() if c["n_messages"] > 0]
    if not chats:
        st.caption("No chats yet.")
        return
    for c in chats:
        is_active = c["id"] == st.session_state.chat_id
        col_open, col_menu = st.columns([0.84, 0.16])
        label = ("• " if is_active else "") + c["title"]
        if col_open.button(label, key=f"open_{c['id']}", use_container_width=True):
            _select_chat(c["id"])
            st.rerun()
        with col_menu.popover("⋯", use_container_width=True):
            new_title = st.text_input("Rename", value=c["title"], key=f"menu_rename_{c['id']}")
            cr, cd = st.columns(2)
            if cr.button("Save", key=f"menu_save_{c['id']}", use_container_width=True):
                chat_store.rename_chat(c["id"], new_title)
                st.rerun()
            if cd.button("🗑 Delete", key=f"menu_del_{c['id']}", use_container_width=True):
                chat_store.delete_chat(c["id"])
                if is_active:
                    st.session_state.pop("chat_id", None)
                st.rerun()


def _render_chat() -> None:
    chat = chat_store.load_chat(st.session_state.chat_id)
    if chat is None:  # deleted underneath us
        _ensure_current_chat()
        st.rerun()

    scope = _active_scope()

    # Empty state -> centered serif greeting + marker that lifts the input.
    if not chat["messages"]:
        st.markdown(
            "<span id='sage-empty-marker'></span>"
            "<div class='sage-hero'>"
            "<span class='sage-spark'>✻</span>"
            "<span class='sage-greeting'>How can I help with your documents?</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if not has_index():
            st.caption("Attach files with the ＋ button below to get started.")
        elif scope == "upload":
            st.caption("🟢 Answering from **your uploaded documents**.")
        else:
            st.caption("📚 Answering from the **default knowledge base** — attach files to switch.")

    for msg in chat["messages"]:
        with st.chat_message(msg["role"], avatar=_AVATARS[msg["role"]]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Sources"):
                    st.markdown(msg["citations"])

    user_input = st.chat_input(
        "Ask anything about your documents…",
        accept_file="multiple",
        file_type=_UPLOAD_TYPES,
    )
    if not user_input:
        return

    # 1) index any attached files first, so the answer can use them immediately
    if user_input.files:
        _ingest_files(user_input.files)
        scope = _active_scope()

    text = (user_input.text or "").strip()
    if not text:
        st.rerun()  # files-only submit -> refresh to show chips + new scope

    chat["messages"].append({"role": "user", "content": text})
    with st.chat_message("user", avatar=_AVATARS["user"]):
        st.markdown(text)

    chain = get_chain(_index_version(), scope)
    with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
        with st.spinner("Retrieving and reasoning…"):
            try:
                out = chain.invoke({"input": text, "chat_history": chat["history"]})
                answer = out["answer"]
                citations = format_citations(out["context"])
            except ConfigError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001 - surface a clean message, log the detail
                log.exception("chain.invoke failed")
                answer = (
                    "⚠️ Something went wrong while answering. Please try again. "
                    f"(details: {type(exc).__name__})"
                )
                citations = ""
        st.markdown(answer)
        if citations:
            with st.expander("📎 Sources"):
                st.markdown(citations)

    chat["history"] += [("human", text), ("ai", answer)]
    chat["messages"].append({"role": "assistant", "content": answer, "citations": citations})
    if chat["title"] == "New chat":
        chat["title"] = chat_store.auto_title(text)
    chat_store.save_chat(chat)
    st.rerun()


# --------------------------------------------------------------------------- #
# Entry point — called by the root `app.py` that Streamlit runs.
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title=settings.app_name, page_icon=settings.app_icon, layout="centered"
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _ensure_current_chat()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
