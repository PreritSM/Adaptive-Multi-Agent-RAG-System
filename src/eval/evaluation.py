from __future__ import annotations

import asyncio
import statistics
import uuid
from pathlib import Path
from typing import Callable

from datasets import Dataset  # type: ignore[import-untyped]
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from ragas import evaluate  # type: ignore[import-untyped]
from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
from ragas.metrics import (  # type: ignore[import-untyped]
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.agents.prompt_loader import load_prompts
from src.agents.state import build_initial_state
from src.api.schemas import EvalGoldenSample, EvalResponse, EvalSample
from src.retrieval.core import EmbeddingFn

# ---------------------------------------------------------------------------
# Golden dataset loading
# ---------------------------------------------------------------------------

def load_golden_dataset(dataset_name: str, datasets_dir: str) -> list[EvalGoldenSample]:
    """Load a golden Q&A dataset from `{datasets_dir}/{dataset_name}.jsonl`."""
    path = Path(datasets_dir) / f"{dataset_name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Eval dataset {dataset_name!r} not found at {path}")

    samples: list[EvalGoldenSample] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        samples.append(EvalGoldenSample.model_validate_json(line))
    return samples


# ---------------------------------------------------------------------------
# Embeddings adapter — bridges the app's bare EmbeddingFn to langchain_core
# ---------------------------------------------------------------------------

class _RagasEmbeddingsAdapter(Embeddings):
    """Wraps an EmbeddingFn callable as a langchain_core Embeddings, since
    ragas expects a LangChain-shaped embeddings object rather than a bare
    callable."""

    def __init__(self, embedding_fn: EmbeddingFn) -> None:
        self._embedding_fn = embedding_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedding_fn(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_fn(text)


# ---------------------------------------------------------------------------
# Sample generation — run the real agent graph over each golden question
# ---------------------------------------------------------------------------

async def generate_eval_samples(
    golden: list[EvalGoldenSample],
    compiled_graph: CompiledStateGraph,
    k: int,
) -> list[EvalSample]:
    """Run the compiled agent graph over each golden question and collect
    the generated answer plus retrieved contexts for RAGAS scoring."""
    samples: list[EvalSample] = []
    for row in golden:
        initial_state = build_initial_state(
            query=row.question,
            query_id=str(uuid.uuid4()),
            k=k,
            retrieval_mode="hybrid",
            uncertainty_threshold=0.75,
        )
        result = await compiled_graph.ainvoke(initial_state)
        samples.append(
            EvalSample(
                question=row.question,
                ground_truth=row.ground_truth,
                answer=result["answer"],
                contexts=[doc.page_content for doc in result["retrieved_docs"]],
            )
        )
    return samples


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def build_ragas_dataset(samples: list[EvalSample]) -> Dataset:
    """Convert EvalSample list to the HF Dataset shape ragas 0.2.x expects."""
    return Dataset.from_list(
        [
            {
                "user_input": sample.question,
                "response": sample.answer,
                "retrieved_contexts": sample.contexts,
                "reference": sample.ground_truth,
            }
            for sample in samples
        ]
    )


async def run_ragas_evaluation(
    samples: list[EvalSample],
    llm: BaseChatModel,
    embeddings: EmbeddingFn,
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

    result = await asyncio.to_thread(
        evaluate,
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(_RagasEmbeddingsAdapter(embeddings)),
    )

    return EvalResponse(
        prompt_variant=prompt_variant,
        faithfulness=statistics.fmean(result["faithfulness"]),
        answer_relevancy=statistics.fmean(result["answer_relevancy"]),
        context_precision=statistics.fmean(result["context_precision"]),
        context_recall=statistics.fmean(result["context_recall"]),
    )


# ---------------------------------------------------------------------------
# Prompt ablation script
# ---------------------------------------------------------------------------

def load_prompt_variants(prompts_yaml_path: str) -> dict[str, dict[str, str]]:
    return load_prompts(prompts_yaml_path)


async def run_ablation_study(
    dataset_name: str,
    prompts_yaml_path: str,
    graph_factory: Callable[[str], CompiledStateGraph],
    llm: BaseChatModel,
    embeddings: EmbeddingFn,
    k: int = 5,
    datasets_dir: str = "data/eval_datasets",
) -> list[EvalResponse]:
    """
    Iterate over all prompt variants defined in prompts.yaml, run RAGAS for
    each, and return a ranked list of EvalResponse objects sorted by
    faithfulness descending.
    """
    variants = load_prompt_variants(prompts_yaml_path)
    golden = load_golden_dataset(dataset_name, datasets_dir)

    results: list[EvalResponse] = []
    for variant in variants:
        graph = graph_factory(variant)
        samples = await generate_eval_samples(golden, graph, k)
        results.append(await run_ragas_evaluation(samples, llm, embeddings, variant))

    results.sort(key=lambda r: r.faithfulness, reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI entry-point (python -m src.eval.evaluation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import dataclasses

    from src.agents.graph import compile_graph
    from src.api.config import get_settings
    from src.api.main import build_deps

    parser = argparse.ArgumentParser(description="Run RAGAS ablation study")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--prompts",
        default="src/agents/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument("--k", type=int, default=5, help="Docs to retrieve per query")
    args = parser.parse_args()

    async def _main() -> None:
        settings = get_settings()
        _runtime, base_deps = build_deps(settings)

        def _graph_factory(variant: str) -> CompiledStateGraph:
            return compile_graph(dataclasses.replace(base_deps, prompt_variant=variant))

        results = await run_ablation_study(
            args.dataset,
            args.prompts,
            _graph_factory,
            base_deps.llm,
            base_deps.embedding_fn,
            k=args.k,
            datasets_dir=settings.eval_datasets_dir,
        )
        for r in results:
            print(r.model_dump_json(indent=2))

    asyncio.run(_main())
