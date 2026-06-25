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
    retrieved_docs: list[Document]
    retrieval_scores: list[float]
    retrieval_mode: str  # "dense" | "sparse" | "hybrid"

    # Uncertainty gating
    uncertainty_score: float
    uncertainty_threshold: float
    requires_fallback: bool

    # Generation artefacts
    answer: str
    agent_trace: list[str]

    # Observability
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
