from __future__ import annotations

import json
from typing import Annotated, Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool

from src.agents.dependencies import RetrieverFactory
from src.agents.prompt_loader import get_prompt

# ---------------------------------------------------------------------------
# Tool factories.
#
# Each tool needs a runtime collaborator (retriever, LLM, HTTP client) that
# only exists once the application has started, so tools are built by a
# factory closing over that collaborator rather than as bare module-level
# @tool functions.
# ---------------------------------------------------------------------------


def make_retrieve_documents_tool(retriever_factory: RetrieverFactory) -> BaseTool:
    @tool
    async def retrieve_documents(
        query: Annotated[str, "The search query to retrieve relevant documents for"],
        k: Annotated[int, "Number of documents to retrieve (1-20)"] = 5,
        mode: Annotated[str, "Retrieval mode: 'dense', 'sparse', or 'hybrid'"] = "hybrid",
    ) -> list[dict[str, str]]:
        """Search the vector store and return the top-k relevant document chunks."""
        retriever = retriever_factory(mode)
        docs, scores = await retriever.retrieve(query, k)
        return [
            {
                "content": doc.page_content,
                "chunk_id": str(doc.metadata.get("chunk_id", "")),
                "score": str(score),
            }
            for doc, score in zip(docs, scores)
        ]

    return retrieve_documents


def make_web_search_tool(tavily_api_key: str) -> BaseTool:
    @tool
    async def web_search(
        query: Annotated[str, "The query to search the web for"],
        num_results: Annotated[int, "Number of web results to return (1-10)"] = 5,
    ) -> list[dict[str, str]]:
        """Perform a live web search when the vector store lacks sufficient context."""
        if not tavily_api_key:
            return []
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "max_results": num_results,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()  # raw Tavily JSON — Any is correct at this boundary

        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", "")),
            }
            for item in payload.get("results", [])
        ]

    return web_search


def make_expand_query_tool(
    llm: BaseChatModel,
    prompts: dict[str, dict[str, str]],
    prompt_variant: str,
) -> BaseTool:
    @tool
    async def expand_query(
        original_query: Annotated[str, "The original user query that returned uncertain results"],
        strategy: Annotated[
            str, "Expansion strategy: 'hypothetical_document' or 'step_back'"
        ] = "hypothetical_document",
    ) -> list[str]:
        """
        Generate alternative phrasings of a query to improve recall when initial
        retrieval confidence is below the uncertainty threshold.
        """
        template = get_prompt(prompts, prompt_variant, strategy)
        rendered = template.format(question=original_query)
        response = await llm.ainvoke([HumanMessage(content=rendered)])
        expansion = str(response.content).strip()
        return [original_query, expansion]

    return expand_query


def make_grade_document_tool(
    llm: BaseChatModel,
    prompts: dict[str, dict[str, str]],
    prompt_variant: str,
) -> BaseTool:
    @tool
    async def grade_document(
        document_content: Annotated[str, "The document chunk text to grade"],
        query: Annotated[str, "The user query the document should be relevant to"],
    ) -> dict[str, bool | float | str]:
        """Grade whether a retrieved document is relevant to the query (0.0 - 1.0)."""
        template = get_prompt(prompts, prompt_variant, "retriever_grader")
        rendered = template.format(question=query, document=document_content)
        response = await llm.ainvoke([HumanMessage(content=rendered)])
        parsed: dict[str, Any] = json.loads(str(response.content))  # raw LLM JSON output — Any is correct at this boundary
        return {
            "relevant": bool(parsed.get("relevant", False)),
            "score": float(parsed.get("score", 0.0)),
            "reason": str(parsed.get("reason", "")),
        }

    return grade_document
