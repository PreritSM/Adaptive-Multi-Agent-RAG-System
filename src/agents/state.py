from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared mutable state passed between every node in the LangGraph."""

    # Core conversation context
    query: str
    query_id: str
    messages: Annotated[list[Any], add_messages]

    # Retrieval artefacts
    k: int
    retrieved_docs: list[Document]
    retrieval_scores: list[float]
    retrieval_mode: str  # "dense" | "sparse" | "hybrid"

    # Uncertainty gating
    uncertainty_score: float
    uncertainty_threshold: float
    requires_fallback: bool
    expand_attempts: int
    generation_attempts: int

    # Generation artefacts
    answer: str
    agent_trace: list[str]

    # Observability
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def build_initial_state(
    query: str,
    query_id: str,
    k: int,
    retrieval_mode: str,
    uncertainty_threshold: float,
) -> AgentState:
    """Build the zeroed-out initial state every graph run starts from."""
    return AgentState(
        query=query,
        query_id=query_id,
        messages=[],
        k=k,
        retrieved_docs=[],
        retrieval_scores=[],
        retrieval_mode=retrieval_mode,
        uncertainty_score=0.0,
        uncertainty_threshold=uncertainty_threshold,
        requires_fallback=False,
        expand_attempts=0,
        generation_attempts=0,
        answer="",
        agent_trace=[],
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
    )
