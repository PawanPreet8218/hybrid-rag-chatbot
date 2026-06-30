# syntax=docker/dockerfile:1
#
# Sage — production image.
# Multi-stage: install deps into a venv, then copy only what's needed into a
# slim runtime image that runs as a non-root user.

# --------------------------------------------------------------------------- #
# Stage 1 — builder: install Python dependencies into an isolated venv
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Build tools needed by some wheels (e.g. tokenizers / rank-bm25 builds).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app
COPY requirements.txt ./
# Install CPU-only torch wheels to keep the image small (no CUDA).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: slim image, non-root, just the venv + source
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGE_DATA_DIR=/data \
    SAGE_LOG_JSON=true \
    HF_HOME=/data/.hf_cache

# Runtime libs PyMuPDF / pandas need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 sage

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY app.py ./
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# Persisted state (vector store, indexes, chats, uploads, model cache) lives here.
RUN mkdir -p /data && chown -R sage:sage /data /app
USER sage

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true", "--browser.gatherUsageStats=false"]
