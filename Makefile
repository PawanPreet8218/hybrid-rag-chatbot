# Sage — developer task runner.
# Usage: `make <target>`. On Windows, run these under Git Bash or WSL.

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install dev lint format type test cov ingest run eval docker-build docker-up clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

dev:  ## Editable install with dev + eval extras
	$(PY) -m pip install -e ".[dev,eval]"

lint:  ## Ruff lint
	ruff check src tests

format:  ## Ruff auto-format + fix
	ruff format src tests
	ruff check --fix src tests

type:  ## Mypy type-check
	mypy src/sage

test:  ## Run the test suite
	pytest

cov:  ## Run tests with coverage report
	pytest --cov=sage --cov-report=term-missing

ingest:  ## Build/rebuild the index from documents/ and 'all pdf/'
	$(PY) -m sage.ingest --rebuild

run:  ## Launch the Streamlit app
	streamlit run app.py

eval:  ## Score the pipeline with RAGAS over the golden set
	$(PY) scripts/eval.py

docker-build:  ## Build the production Docker image
	docker compose build

docker-up:  ## Run the app via Docker Compose
	docker compose up

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info
