from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class RetrievalMode(str, Enum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    uncertainty_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    collection: str = Field(default="default")


class DocumentChunk(BaseModel):
    id: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[DocumentChunk]
    # 1-based citation index → chunk ID; auto-populated from sources if omitted
    citation_map: dict[int, str] = Field(default_factory=dict)
    # Per-source confidence scores, parallel to sources; auto-populated if omitted
    confidence_scores: list[float] = Field(default_factory=list)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    agent_trace: list[str]
    latency_ms: float = Field(..., ge=0.0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)

    @model_validator(mode="after")
    def populate_citation_fields(self) -> Self:
        if not self.citation_map:
            self.citation_map = {i + 1: chunk.id for i, chunk in enumerate(self.sources)}
        if not self.confidence_scores:
            self.confidence_scores = [chunk.score for chunk in self.sources]
        if len(self.confidence_scores) != len(self.sources):
            raise ValueError(
                f"confidence_scores length ({len(self.confidence_scores)}) "
                f"must equal sources length ({len(self.sources)})"
            )
        if set(self.citation_map.keys()) != set(range(1, len(self.sources) + 1)):
            raise ValueError("citation_map keys must be 1-based indices covering all sources")
        return self


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1)
    collection: str = Field(default="default")
    metadata: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("metadata", mode="after")
    @classmethod
    def metadata_length_matches_documents(
        cls, metadata: list[dict[str, Any]], info: Any
    ) -> list[dict[str, Any]]:
        documents: list[str] | None = (info.data or {}).get("documents")
        if documents and metadata and len(metadata) != len(documents):
            raise ValueError(
                f"metadata length ({len(metadata)}) must match documents length ({len(documents)})"
            )
        return metadata


class IngestResponse(BaseModel):
    collection: str
    inserted: int = Field(..., ge=0)
    skipped: int = Field(..., ge=0)


class EvalRequest(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    prompt_variant: str = Field(default="default")
    k: int = Field(default=5, ge=1, le=20)


class EvalResponse(BaseModel):
    prompt_variant: str
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(..., ge=0.0, le=1.0)


class EvalGoldenSample(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)


class EvalSample(BaseModel):
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_reachable: bool
    chroma_reachable: bool = False
