from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents.state import AgentState


# ---------------------------------------------------------------------------
# Node stubs
# (Each node is a pure function: AgentState -> dict[str, ...] patch)
# ---------------------------------------------------------------------------

async def node_retrieve(state: AgentState) -> dict[str, object]:
    """Run the configured retriever and populate retrieved_docs + retrieval_scores."""
    # TODO: call build_retriever() from retrieval/core.py
    raise NotImplementedError


async def node_grade_documents(state: AgentState) -> dict[str, object]:
    """LLM-as-judge: score each retrieved doc for relevance; filter low-scorers."""
    # TODO: call grade_document tool for each doc, update retrieved_docs
    raise NotImplementedError


async def node_assess_uncertainty(state: AgentState) -> dict[str, object]:
    """Compute uncertainty_score; set requires_fallback flag."""
    # TODO: call score_uncertainty() from retrieval/uncertainty.py
    raise NotImplementedError


async def node_expand_query(state: AgentState) -> dict[str, object]:
    """HyDE / step-back expansion when confidence is low; re-run retrieval."""
    # TODO: call expand_query tool, re-populate retrieved_docs
    raise NotImplementedError


async def node_generate(state: AgentState) -> dict[str, object]:
    """Generate final answer via explicit LCEL chain (prompt | llm | parser)."""
    # TODO: load prompt variant from prompts.yaml, build LCEL chain, invoke
    raise NotImplementedError


async def node_hallucination_check(state: AgentState) -> dict[str, object]:
    """LLM-as-judge: verify the generated answer is grounded in retrieved docs."""
    # TODO: call hallucination_judge prompt, set requires_fallback if ungrounded
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Routing functions
# (Named, unit-testable — no inline lambdas allowed per CLAUDE.md)
# ---------------------------------------------------------------------------

def route_after_uncertainty(
    state: AgentState,
) -> Literal["expand_query", "generate"]:
    """Branch: low confidence → expand query; sufficient confidence → generate."""
    if state["requires_fallback"]:
        return "expand_query"
    return "generate"


def route_after_hallucination_check(
    state: AgentState,
) -> Literal["generate", "__end__"]:
    """Branch: hallucination detected → regenerate; clean answer → end."""
    if state["requires_fallback"]:
        return "generate"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_documents", node_grade_documents)
    graph.add_node("assess_uncertainty", node_assess_uncertainty)
    graph.add_node("expand_query", node_expand_query)
    graph.add_node("generate", node_generate)
    graph.add_node("hallucination_check", node_hallucination_check)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "assess_uncertainty")

    graph.add_conditional_edges(
        "assess_uncertainty",
        route_after_uncertainty,
        {"expand_query": "expand_query", "generate": "generate"},
    )

    graph.add_edge("expand_query", "retrieve")
    graph.add_edge("generate", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {"generate": "generate", "__end__": END},
    )

    return graph


# Compiled graph — import this in api/main.py
compiled_graph = build_graph().compile()
