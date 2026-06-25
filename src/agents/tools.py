from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------

@tool
def retrieve_documents(
    query: Annotated[str, "The search query to retrieve relevant documents for"],
    k: Annotated[int, "Number of documents to retrieve (1-20)"] = 5,
    mode: Annotated[str, "Retrieval mode: 'dense', 'sparse', or 'hybrid'"] = "hybrid",
) -> list[dict[str, str]]:
    """Search the vector store and return the top-k relevant document chunks."""
    # TODO: delegate to the retriever built in core.py
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Web search fallback tool
# ---------------------------------------------------------------------------

@tool
def web_search(
    query: Annotated[str, "The query to search the web for"],
    num_results: Annotated[int, "Number of web results to return (1-10)"] = 5,
) -> list[dict[str, str]]:
    """Perform a live web search when the vector store lacks sufficient context."""
    # TODO: integrate a search API (e.g. Tavily or SerpAPI)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Uncertainty re-retrieval tool
# ---------------------------------------------------------------------------

@tool
def expand_query(
    original_query: Annotated[str, "The original user query that returned uncertain results"],
    strategy: Annotated[str, "Expansion strategy: 'hypothetical_document' or 'step_back'"] = "hypothetical_document",
) -> list[str]:
    """
    Generate alternative phrasings of a query to improve recall when initial
    retrieval confidence is below the uncertainty threshold.
    """
    # TODO: use an LLM call (HyDE or step-back prompting) to produce expansions
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Grounding tool
# ---------------------------------------------------------------------------

@tool
def grade_document(
    document_content: Annotated[str, "The document chunk text to grade"],
    query: Annotated[str, "The user query the document should be relevant to"],
) -> dict[str, str | float]:
    """Grade whether a retrieved document is relevant to the query (0.0 – 1.0)."""
    # TODO: LLM-as-judge call returning {"relevant": bool, "score": float, "reason": str}
    raise NotImplementedError
