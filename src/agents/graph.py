from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.dependencies import GraphDependencies
from src.agents.prompt_loader import get_prompt
from src.agents.state import AgentState
from src.agents.tools import make_expand_query_tool, make_grade_document_tool
from src.retrieval.core import QueryClassifier
from src.retrieval.uncertainty import is_uncertain, score_uncertainty


def _format_docs(docs: list[Document]) -> str:
    lines: list[str] = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        lines.append(f"[{chunk_id}] {doc.page_content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Node factory
# (Each node is a pure function of AgentState -> state patch, built by a
#  closure that captures runtime collaborators (LLM, retrievers, prompts).
#  compiled_graph cannot be built at import time because those collaborators
#  require a running event loop / network access — see api/main.py lifespan.)
# ---------------------------------------------------------------------------

def build_graph(deps: GraphDependencies) -> StateGraph:
    grade_document_tool = make_grade_document_tool(deps.llm, deps.prompts, deps.prompt_variant)
    expand_query_tool = make_expand_query_tool(deps.llm, deps.prompts, deps.prompt_variant)

    async def node_retrieve(state: AgentState) -> dict[str, object]:
        """Run the configured retriever and populate retrieved_docs + retrieval_scores."""
        mode = QueryClassifier().classify(state["query"])
        retriever = deps.retriever_factory(mode.value)
        docs, scores = await retriever.retrieve(state["query"], state["k"])
        return {
            "retrieved_docs": docs,
            "retrieval_scores": scores,
            "retrieval_mode": mode.value,
        }

    async def node_grade_documents(state: AgentState) -> dict[str, object]:
        """LLM-as-judge: score each retrieved doc for relevance; filter low-scorers."""
        kept_docs: list[Document] = []
        kept_scores: list[float] = []
        for doc in state["retrieved_docs"]:
            result: dict[str, Any] = await grade_document_tool.ainvoke(
                {"document_content": doc.page_content, "query": state["query"]}
            )  # tool result crosses an LLM-as-judge boundary — Any is correct here
            if bool(result.get("relevant", False)):
                kept_docs.append(doc)
                kept_scores.append(float(result.get("score", 0.0)))
        return {"retrieved_docs": kept_docs, "retrieval_scores": kept_scores}

    async def node_assess_uncertainty(state: AgentState) -> dict[str, object]:
        """Compute uncertainty_score; set requires_fallback flag."""
        query_embedding = np.asarray(deps.embedding_fn(state["query"]), dtype=np.float32)
        if state["retrieved_docs"]:
            doc_embeddings = np.asarray(
                [deps.embedding_fn(doc.page_content) for doc in state["retrieved_docs"]],
                dtype=np.float32,
            )
        else:
            doc_embeddings = np.empty((0, query_embedding.shape[0]), dtype=np.float32)

        uncertainty = score_uncertainty(query_embedding, doc_embeddings, state["retrieval_scores"])

        if state["expand_attempts"] >= deps.max_expand_attempts:
            requires_fallback = False
        else:
            requires_fallback = is_uncertain(uncertainty, state["uncertainty_threshold"])

        return {"uncertainty_score": uncertainty, "requires_fallback": requires_fallback}

    async def node_expand_query(state: AgentState) -> dict[str, object]:
        """HyDE / step-back expansion when confidence is low; re-run retrieval."""
        expansions: list[str] = await expand_query_tool.ainvoke(
            {"original_query": state["query"], "strategy": "hypothetical_document"}
        )
        expanded_query = expansions[-1] if len(expansions) > 1 else state["query"]
        return {
            "query": expanded_query,
            "expand_attempts": state["expand_attempts"] + 1,
        }

    async def node_generate(state: AgentState) -> dict[str, object]:
        """Generate final answer via explicit LCEL chain (prompt | llm | parser)."""
        template = get_prompt(deps.prompts, deps.prompt_variant, "generator")
        chain = PromptTemplate.from_template(template) | deps.llm | StrOutputParser()
        answer = await chain.ainvoke(
            {"context": _format_docs(state["retrieved_docs"]), "question": state["query"]}
        )
        return {
            "answer": answer,
            "generation_attempts": state["generation_attempts"] + 1,
        }

    async def node_hallucination_check(state: AgentState) -> dict[str, object]:
        """LLM-as-judge: verify the generated answer is grounded in retrieved docs."""
        if state["generation_attempts"] >= deps.max_generation_attempts:
            return {"requires_fallback": False}

        template = get_prompt(deps.prompts, deps.prompt_variant, "hallucination_judge")
        chain = PromptTemplate.from_template(template) | deps.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {"context": _format_docs(state["retrieved_docs"]), "answer": state["answer"]}
        )
        parsed: dict[str, Any] = json.loads(raw)  # raw LLM JSON output — Any is correct at this boundary
        grounded = bool(parsed.get("grounded", True))
        return {"requires_fallback": not grounded}

    # -----------------------------------------------------------------------
    # Routing functions
    # (Named, unit-testable — no inline lambdas allowed per CLAUDE.md)
    # -----------------------------------------------------------------------

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


def compile_graph(deps: GraphDependencies) -> CompiledStateGraph:
    """Build and compile the agent graph for the given runtime dependencies."""
    return build_graph(deps).compile()
