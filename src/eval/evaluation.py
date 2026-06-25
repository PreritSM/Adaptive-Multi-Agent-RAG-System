from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.api.schemas import EvalResponse


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def build_ragas_dataset(samples: list[EvalSample]) -> Dataset:
    """Convert EvalSample list to a Hugging Face Dataset expected by RAGAS."""
    # TODO: map fields to RAGAS schema (question, answer, contexts, ground_truth)
    raise NotImplementedError


async def run_ragas_evaluation(
    samples: list[EvalSample],
    prompt_variant: str = "default",
) -> EvalResponse:
    """
    Run all four core RAGAS metrics and return a typed EvalResponse.

    Metrics:
      - faithfulness        (hallucination guard)
      - answer_relevancy    (response quality)
      - context_precision   (retrieval precision)
      - context_recall      (retrieval recall)
    """
    dataset = build_ragas_dataset(samples)

    # TODO: call ragas.evaluate() with the four metrics above
    # result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                     context_precision, context_recall])
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Prompt ablation script
# ---------------------------------------------------------------------------

def load_prompt_variants(prompts_yaml_path: str) -> dict[str, Any]:
    with open(prompts_yaml_path) as f:
        return yaml.safe_load(f)


async def run_ablation_study(
    dataset_name: str,
    prompts_yaml_path: str,
    k: int = 5,
) -> list[EvalResponse]:
    """
    Iterate over all prompt variants defined in prompts.yaml, run RAGAS for
    each, and return a ranked list of EvalResponse objects sorted by
    faithfulness descending.
    """
    variants = load_prompt_variants(prompts_yaml_path)

    # TODO: for each variant key, load the dataset, run the RAG pipeline with
    #       that variant, collect samples, call run_ragas_evaluation()
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI entry-point (python -m src.eval.evaluation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS ablation study")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument(
        "--prompts",
        default="src/agents/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument("--k", type=int, default=5, help="Docs to retrieve per query")
    args = parser.parse_args()

    results = asyncio.run(
        run_ablation_study(args.dataset, args.prompts, k=args.k)
    )
    for r in results:
        print(r.model_dump_json(indent=2))
