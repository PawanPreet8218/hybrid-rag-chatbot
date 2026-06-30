# 📚 Sage — Your Personal RAG Document Assistant

A grounded, Claude-style chatbot over **your own documents**. Upload PDFs, Word,
Excel, CSV or text files and chat with them — every core RAG stage is implemented
explicitly: chunking, embedding, a persisted vector DB, similarity search with
tuned kwargs, **hybrid retrieval, reranking, conversational memory, citations,
and an evaluation harness**.

**Features**
- 📎 **Multi-format upload** — PDF, Word (`.docx`), Excel (`.xlsx`/`.xls`), CSV, `.txt`, `.md`, indexed on the fly.
- 💬 **Chat history sidebar** — multiple conversations with **rename / delete**, persisted across restarts.
- 🔍 **Hybrid retrieval** — dense + BM25 → Reciprocal Rank Fusion → cross-encoder reranking → grounded answers with citations.

> Rename the app any time with `SAGE_APP_NAME` (or edit the default in `src/sage/config.py`).

---

## Architecture

```
OFFLINE  ─ python -m sage.ingest ────────────────────────────────────────────
  files ─▶ load (PDF/Word/Excel/CSV/txt) ─▶ section-aware chunk ─▶ enrich ─▶ embed
            (documents/, all pdf/)          (800/150, sentence-aware)  (all-mpnet)
                                       │
                 ┌─────────────────────┴─────────────────────┐
            Chroma (PERSISTED, dense)              BM25 store (sparse, pickled)
                         (both written atomically — crash-safe)

ONLINE   ─ streamlit run app.py ─────────────────────────────────────────────
  question + chat history
     ├─ history-aware rewrite   (condense follow-ups -> standalone query)
     ├─ dense  : Chroma MMR  (k, fetch_k, lambda_mult)
     ├─ sparse : BM25
     ├─ fuse   : EnsembleRetriever — Reciprocal Rank Fusion (weighted)
     ├─ rerank : bge-reranker-base cross-encoder -> top-5
     ├─ generate : Groq llama-3.3-70b, grounded prompt (temp 0.1)
     └─ answer + citations (section · file · page)

QUALITY  ─ python scripts/eval.py ───────────────────────────────────────────
  golden set ─▶ RAGAS: faithfulness · answer_relevancy · context_precision · context_recall
```

## Project layout

```
app.py                     # thin Streamlit entry point (bootstraps src/ on path)
pyproject.toml             # packaging + ruff / mypy / pytest config
requirements.txt           # runtime deps (mirrors pyproject)
Dockerfile                 # multi-stage, non-root production image
docker-compose.yml         # one-command local deployment + persistent volume
Makefile                   # dev task runner (install / lint / test / run / ingest)
.github/workflows/ci.yml   # CI: ruff + mypy + pytest

src/sage/                  # the installable package
  config.py                # pydantic-settings — every knob, env-overridable
  logging_config.py        # structured (console or JSON) logging
  exceptions.py            # typed errors
  io_utils.py              # atomic writes + filename-safety (anti path-traversal)
  loaders.py               # multi-format document loaders
  ingest.py                # build + persist Chroma + BM25  (CLI + in-app)
  retriever.py             # hybrid + RRF + reranker
  rag_chain.py             # LCEL chain: memory -> retrieve -> grounded answer -> cite
  chat_store.py            # persistent chat sessions (atomic JSON)
  ui.py                    # Streamlit UI

scripts/eval.py            # RAGAS evaluation
data/golden_set.json       # eval questions + ground truth
tests/                     # pytest suite (model-free, runs offline)
```

## RAG features covered

| Stage | Implementation |
|---|---|
| **Chunking** | `RecursiveCharacterTextSplitter`, sentence-aware separators, overlap, `start_index` |
| **Metadata** | source file, section title (parsed from filename), page, chunk_id, content hash |
| **Embedding** | `all-mpnet-base-v2`, L2-normalized (cosine) |
| **Vector DB** | **Chroma, persisted to disk** (built once, not on every start) |
| **Similarity / kwargs** | MMR (`k`, `fetch_k`, `lambda_mult`) or score-threshold — all in config |
| **Hybrid search** | dense + BM25 fused with Reciprocal Rank Fusion (`EnsembleRetriever`) |
| **Reranking** | `bge-reranker-base` cross-encoder (`ContextualCompressionRetriever`) |
| **Memory** | history-aware retriever condenses follow-ups |
| **Grounding** | answers only from context; refuses with "I don't have that information." |
| **Citations** | exact source docs shown under each answer |
| **Evaluation** | RAGAS metrics over a golden set (`scripts/eval.py`) |

## Setup & run

```bash
# 1. install (Windows)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # runtime
#   …or for development:  pip install -e ".[dev,eval]"

# 2. add your key
copy .env.example .env                    # then put your GROQ_API_KEY in .env

# 3. build the index ONCE (or after changing the source files)
python -m sage.ingest --rebuild

# 4. run the chatbot
streamlit run app.py

# 5. (optional) measure quality
python scripts/eval.py
```

## Run with Docker

```bash
cp .env.example .env        # add your GROQ_API_KEY
docker compose up --build   # -> http://localhost:8501
```

All persisted state (vector store, indexes, chats, uploads, model cache) lives in
the named `sage-data` volume, so it survives container restarts.

## Development

```bash
make dev        # editable install with dev + eval extras
make lint       # ruff
make type       # mypy
make test       # pytest (model-free, fast, offline)
make cov        # tests + coverage report
```

The test suite is intentionally **model-free** — it exercises config validation,
filename safety, atomic IO, loaders, the chat store and ingest helpers without
downloading any ML model, so it runs in seconds in CI.

## Configuration

Everything is a field on `Settings` in `src/sage/config.py` and can be overridden
by an env var (or `.env`) using the `SAGE_` prefix:

| Setting | Env var | Default |
|---|---|---|
| Dense search type | `SAGE_DENSE_SEARCH_TYPE` | `mmr` |
| Hybrid weights `[sparse, dense]` | `SAGE_HYBRID_WEIGHTS` | `(0.4, 0.6)` |
| Rerank top-N | `SAGE_RERANK_TOP_N` | `5` |
| Chunk size / overlap | `SAGE_CHUNK_SIZE` / `SAGE_CHUNK_OVERLAP` | `800` / `150` |
| Max upload size (MB) | `SAGE_MAX_UPLOAD_MB` | `50` |
| Data directory | `SAGE_DATA_DIR` | repo root |
| Log level / JSON logs | `SAGE_LOG_LEVEL` / `SAGE_LOG_JSON` | `INFO` / `false` |

## Production hardening included

- **Validated config** (pydantic-settings) — bad values fail fast at startup.
- **Atomic persistence** — index, BM25 store and chats are written via temp-file
  + rename, so an interrupted write can never corrupt them.
- **Upload safety** — filenames are sanitised (no path traversal) and oversized
  files are rejected with a clear message.
- **Structured logging** — console or JSON, with noisy libraries quietened.
- **Graceful errors** — the UI never shows a raw traceback; failures are logged
  and surfaced as a friendly message.
- **Lazy ML imports** — listing documents / booting the UI doesn't load torch.
- **Tests + CI + Docker** — pytest suite, GitHub Actions, non-root container.

## Roadmap (next steps)

- FastAPI service (API-first deployment) alongside the Streamlit UI
- Langfuse tracing (latency / token cost per query)
- Parent-document retrieval (retrieve small, answer from full section)
- Streaming responses
```
