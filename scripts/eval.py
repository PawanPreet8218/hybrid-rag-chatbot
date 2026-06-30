"""Evaluation harness — proves the RAG system actually works (the "strong
foundation" differentiator).

For each question in ``data/golden_set.json`` we run the real pipeline and score
it with RAGAS:

    faithfulness        -> is the answer grounded in retrieved context? (anti-hallucination)
    answer_relevancy    -> does the answer address the question?
    context_precision   -> are the retrieved chunks relevant? (retriever quality)
    context_recall      -> did we retrieve everything needed? (coverage)

Run from the repo root::

    python scripts/eval.py

NOTE: RAGAS needs a judge LLM + embeddings. By default it uses OpenAI; here we
wire it to use the same Groq LLM and local HF embeddings so it stays free.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src/` importable when run as a plain script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datasets import Dataset  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from sage.config import settings  # noqa: E402
from sage.logging_config import get_logger  # noqa: E402
from sage.rag_chain import build_rag_chain  # noqa: E402

log = get_logger("sage.eval")


def collect_predictions() -> Dataset:
    chain = build_rag_chain()
    golden = json.loads((settings.project_root / "data" / "golden_set.json").read_text())

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden:
        out = chain.invoke({"input": item["question"], "chat_history": []})
        rows["question"].append(item["question"])
        rows["answer"].append(out["answer"])
        rows["contexts"].append([d.page_content for d in out["context"]])
        rows["ground_truth"].append(item["ground_truth"])
        log.info("scored question: %s", item["question"])
    return Dataset.from_dict(rows)


def main() -> None:
    log.info("running pipeline over golden set …")
    dataset = collect_predictions()

    judge_llm = LangchainLLMWrapper(
        ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model, temperature=0)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    log.info("scoring with RAGAS …")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    df = result.to_pandas()
    out_path = settings.project_root / "eval_results.csv"
    df.to_csv(out_path, index=False)
    print("\n=== RAGAS scores (mean) ===")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
          .mean().round(3).to_string())
    print(f"\nSaved per-question detail -> {out_path}")


if __name__ == "__main__":
    main()
