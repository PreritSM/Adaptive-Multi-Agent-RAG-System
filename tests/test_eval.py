from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.dependencies import GraphDependencies
from src.agents.graph import compile_graph
from src.agents.prompt_loader import load_prompts
from src.api.schemas import EvalGoldenSample, EvalSample
from src.eval.evaluation import build_ragas_dataset, generate_eval_samples, load_golden_dataset
from src.retrieval.core import Retriever

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_FAKE_DOCS: list[Document] = [
    Document(page_content="Fake context chunk one.", metadata={"chunk_id": "aaaa"}),
    Document(page_content="Fake context chunk two.", metadata={"chunk_id": "bbbb"}),
]
_FAKE_SCORES: list[float] = [1.0, 0.01]


class _FakeRetriever:
    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        return _FAKE_DOCS, _FAKE_SCORES


def _fake_retriever_factory(mode: str) -> Retriever:
    return _FakeRetriever()


def _constant_embedding_fn(text: str) -> list[float]:
    return [1.0, 0.0, 0.0]


def _build_fake_deps() -> GraphDependencies:
    # Grading scores are deliberately skewed (not equal): node_grade_documents
    # overwrites retrieval_scores with these graded scores, and assess_uncertainty
    # scores the *distribution* of retrieval_scores — equal scores flatten the
    # distribution into high (unwanted) uncertainty and trigger an expand-query
    # retry loop instead of proceeding straight to generate.
    responses = [
        '{"relevant": true, "score": 0.99, "reason": "ok"}',
        '{"relevant": true, "score": 0.01, "reason": "ok"}',
        "This is a fake answer grounded in context.",
        '{"grounded": true, "unsupported_claims": []}',
    ]
    return GraphDependencies(
        llm=FakeListChatModel(responses=responses),
        retriever_factory=_fake_retriever_factory,
        embedding_fn=_constant_embedding_fn,
        prompts=load_prompts("src/agents/prompts.yaml"),
        prompt_variant="default",
        max_expand_attempts=2,
        max_generation_attempts=2,
    )


# ---------------------------------------------------------------------------
# load_golden_dataset
# ---------------------------------------------------------------------------

class TestLoadGoldenDataset:
    def test_loads_smoke_fixture(self) -> None:
        samples = load_golden_dataset("smoke", "data/eval_datasets")
        assert len(samples) == 3
        assert all(isinstance(s, EvalGoldenSample) for s in samples)
        assert all(s.question and s.ground_truth for s in samples)

    def test_missing_dataset_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_golden_dataset("does-not-exist", "data/eval_datasets")


# ---------------------------------------------------------------------------
# build_ragas_dataset
# ---------------------------------------------------------------------------

class TestBuildRagasDataset:
    def test_column_names_and_row_count(self) -> None:
        samples = [
            EvalSample(
                question="Q1",
                ground_truth="GT1",
                answer="A1",
                contexts=["C1"],
            ),
            EvalSample(
                question="Q2",
                ground_truth="GT2",
                answer="A2",
                contexts=["C2a", "C2b"],
            ),
        ]
        dataset = build_ragas_dataset(samples)
        assert len(dataset) == 2
        assert set(dataset.column_names) == {
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        }
        assert dataset[0]["user_input"] == "Q1"
        assert dataset[1]["retrieved_contexts"] == ["C2a", "C2b"]


# ---------------------------------------------------------------------------
# generate_eval_samples
# ---------------------------------------------------------------------------

class TestGenerateEvalSamples:
    @pytest.mark.asyncio
    async def test_runs_graph_over_golden_questions(self) -> None:
        golden = [
            EvalGoldenSample(question="What is RAG?", ground_truth="Retrieval-augmented generation."),
            EvalGoldenSample(question="What is LangGraph?", ground_truth="A graph orchestration library."),
        ]
        graph = compile_graph(_build_fake_deps())

        samples = await generate_eval_samples(golden, graph, k=2)

        assert len(samples) == len(golden)
        for golden_row, sample in zip(golden, samples):
            assert sample.question == golden_row.question
            assert sample.ground_truth == golden_row.ground_truth
            assert sample.answer == "This is a fake answer grounded in context."
            assert sample.contexts == [doc.page_content for doc in _FAKE_DOCS]
