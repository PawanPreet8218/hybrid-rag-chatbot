"""RAG chain (ONLINE) — modern LangChain LCEL, no deprecated RetrievalQA.

    chat history + question
        -> history-aware retriever (condense follow-ups to a standalone query)
        -> hybrid retrieve + rerank  (sage.retriever)
        -> grounded generation (Groq llama-3.3-70b)
        -> answer + the exact source documents (for citations)

Public API::

    chain = build_rag_chain()
    out   = chain.invoke({"input": "...", "chat_history": [...]})
    out["answer"], out["context"]   # context = list[Document] used to answer
"""
from __future__ import annotations

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from sage.config import settings
from sage.exceptions import ConfigError
from sage.logging_config import get_logger
from sage.retriever import build_retriever

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
# Rewrites a follow-up ("and their team?") into a standalone question using history.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Given the chat history and the latest user question, rewrite it as a "
         "standalone question that can be understood without the history. "
         "Do NOT answer it — only reformulate. If it is already standalone, return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Grounded answer prompt — answers ONLY from retrieved context.
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are Sage, a helpful assistant that answers questions about the user's "
         "own documents. Answer the question using ONLY the context below. Give a "
         "thorough, well-explained answer with background, significance and practical "
         "relevance when the context supports it. Explain not just what something is, "
         "but why it matters and how it works. "
         "If the context does not contain the answer, reply exactly: "
         "\"I don't have that information in your documents.\"\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def load_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise ConfigError(
            "GROQ_API_KEY not set. Copy .env.example -> .env and add your key."
        )
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def build_rag_chain(llm=None, retriever=None):
    """Build the full conversational RAG chain (memory + hybrid retrieval + citations).

    ``llm`` and ``retriever`` can be injected so the app reuses cached heavy
    models across re-indexing; both are built on demand when omitted.
    """
    llm = llm or load_llm()
    retriever = retriever if retriever is not None else build_retriever()

    history_aware = create_history_aware_retriever(llm, retriever, CONDENSE_PROMPT)
    answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    # returns {"answer": str, "context": list[Document], ...}
    return create_retrieval_chain(history_aware, answer_chain)


def format_citations(docs) -> str:
    """Deduplicated, human-readable source list for the UI."""
    seen, lines = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{d.metadata.get('section', '?')}** "
                     f"({d.metadata.get('source', '?')}, page {d.metadata.get('page', '?')})")
    return "\n".join(lines)


if __name__ == "__main__":  # CLI smoke test
    chain = build_rag_chain()
    history: list = []
    for q in ["What are these documents about?", "Summarise the key points."]:
        out = chain.invoke({"input": q, "chat_history": history})
        print(f"\nQ: {q}\nA: {out['answer']}\nSources:\n{format_citations(out['context'])}")
        history += [("human", q), ("ai", out["answer"])]
