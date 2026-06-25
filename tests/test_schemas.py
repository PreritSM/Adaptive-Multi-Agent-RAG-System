from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    DocumentChunk,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalMode,
)


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------

class TestQueryRequest:
    def test_defaults(self) -> None:
        req = QueryRequest(query="What is RAG?")
        assert req.k == 5
        assert req.retrieval_mode == RetrievalMode.hybrid
        assert req.uncertainty_threshold == 0.75
        assert req.collection == "default"

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_k_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="x", k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="x", k=21)
        req = QueryRequest(query="x", k=20)
        assert req.k == 20

    def test_retrieval_mode_enum(self) -> None:
        req = QueryRequest(query="x", retrieval_mode="dense")
        assert req.retrieval_mode == RetrievalMode.dense

    def test_invalid_retrieval_mode(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="x", retrieval_mode="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DocumentChunk
# ---------------------------------------------------------------------------

class TestDocumentChunk:
    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DocumentChunk(id="x", content="y", score=1.1)
        with pytest.raises(ValidationError):
            DocumentChunk(id="x", content="y", score=-0.1)

    def test_metadata_defaults_empty(self) -> None:
        chunk = DocumentChunk(id="a", content="b", score=0.5)
        assert chunk.metadata == {}


# ---------------------------------------------------------------------------
# QueryResponse — citation_map and confidence_scores
# ---------------------------------------------------------------------------

class TestQueryResponse:
    def _make_chunk(self, chunk_id: str, score: float = 0.8) -> DocumentChunk:
        return DocumentChunk(id=chunk_id, content="text", score=score)

    def _make_response(self, chunks: list[DocumentChunk]) -> QueryResponse:
        return QueryResponse(
            query_id="qid-1",
            answer="The answer.",
            sources=chunks,
            uncertainty_score=0.2,
            agent_trace=["retrieve", "grade", "generate"],
            latency_ms=120.5,
            prompt_tokens=50,
            completion_tokens=30,
        )

    def test_citation_map_auto_populated(self) -> None:
        chunks = [self._make_chunk("doc-a"), self._make_chunk("doc-b")]
        resp = self._make_response(chunks)
        assert resp.citation_map == {1: "doc-a", 2: "doc-b"}

    def test_confidence_scores_auto_populated(self) -> None:
        chunks = [self._make_chunk("x", 0.9), self._make_chunk("y", 0.6)]
        resp = self._make_response(chunks)
        assert resp.confidence_scores == [0.9, 0.6]

    def test_empty_sources(self) -> None:
        resp = self._make_response([])
        assert resp.citation_map == {}
        assert resp.confidence_scores == []

    def test_explicit_citation_map_preserved(self) -> None:
        chunk = self._make_chunk("doc-z", 0.7)
        resp = QueryResponse(
            query_id="q",
            answer="a",
            sources=[chunk],
            citation_map={1: "doc-z"},
            confidence_scores=[0.7],
            uncertainty_score=0.3,
            agent_trace=[],
            latency_ms=10.0,
            prompt_tokens=5,
            completion_tokens=5,
        )
        assert resp.citation_map == {1: "doc-z"}

    def test_mismatched_confidence_scores_rejected(self) -> None:
        chunk = self._make_chunk("doc-1")
        with pytest.raises(ValidationError):
            QueryResponse(
                query_id="q",
                answer="a",
                sources=[chunk],
                citation_map={1: "doc-1"},
                confidence_scores=[0.5, 0.6],  # length mismatch
                uncertainty_score=0.1,
                agent_trace=[],
                latency_ms=5.0,
                prompt_tokens=1,
                completion_tokens=1,
            )

    def test_invalid_citation_map_keys_rejected(self) -> None:
        chunk = self._make_chunk("doc-1")
        with pytest.raises(ValidationError):
            QueryResponse(
                query_id="q",
                answer="a",
                sources=[chunk],
                citation_map={2: "doc-1"},  # should be key 1, not 2
                confidence_scores=[0.5],
                uncertainty_score=0.1,
                agent_trace=[],
                latency_ms=5.0,
                prompt_tokens=1,
                completion_tokens=1,
            )

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryResponse(
                query_id="q",
                answer="a",
                sources=[],
                uncertainty_score=0.1,
                agent_trace=[],
                latency_ms=-1.0,
                prompt_tokens=1,
                completion_tokens=1,
            )


# ---------------------------------------------------------------------------
# IngestRequest
# ---------------------------------------------------------------------------

class TestIngestRequest:
    def test_empty_documents_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(documents=[])

    def test_metadata_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(
                documents=["doc1", "doc2"],
                metadata=[{"source": "a"}],  # only 1 item for 2 docs
            )

    def test_metadata_length_match_accepted(self) -> None:
        req = IngestRequest(
            documents=["doc1", "doc2"],
            metadata=[{"source": "a"}, {"source": "b"}],
        )
        assert len(req.metadata) == 2

    def test_empty_metadata_always_valid(self) -> None:
        req = IngestRequest(documents=["doc1", "doc2"])
        assert req.metadata == []


# ---------------------------------------------------------------------------
# EvalRequest / EvalResponse
# ---------------------------------------------------------------------------

class TestEvalSchemas:
    def test_eval_request_defaults(self) -> None:
        req = EvalRequest(dataset_name="my-dataset")
        assert req.prompt_variant == "default"
        assert req.k == 5

    def test_eval_response_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EvalResponse(
                prompt_variant="v1",
                faithfulness=1.1,  # out of range
                answer_relevancy=0.9,
                context_precision=0.8,
                context_recall=0.7,
            )

    def test_eval_response_valid(self) -> None:
        resp = EvalResponse(
            prompt_variant="v1",
            faithfulness=0.95,
            answer_relevancy=0.88,
            context_precision=0.82,
            context_recall=0.79,
        )
        assert resp.faithfulness == 0.95


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------

class TestHealthResponse:
    def test_chroma_reachable_defaults_false(self) -> None:
        h = HealthResponse(status="ok", version="0.1.0", qdrant_reachable=True)
        assert h.chroma_reachable is False

    def test_all_fields(self) -> None:
        h = HealthResponse(
            status="ok", version="0.1.0", qdrant_reachable=True, chroma_reachable=True
        )
        assert h.qdrant_reachable is True
        assert h.chroma_reachable is True
