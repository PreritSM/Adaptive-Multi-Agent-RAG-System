from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from src.retrieval.core import EmbeddingFn, Retriever


class RetrieverFactory(Protocol):
    def __call__(self, mode: str) -> Retriever: ...


@dataclass(frozen=True)
class GraphDependencies:
    """Runtime collaborators injected into every LangGraph node closure.

    Built once at application startup (see api/main.py lifespan) and captured
    by the node closures returned from agents/graph.py:build_graph — the
    module-level compiled_graph is never constructed at import time because
    these collaborators (LLM client, Qdrant client, embeddings) require a
    running event loop / network access to initialise.
    """

    llm: BaseChatModel
    retriever_factory: RetrieverFactory
    embedding_fn: EmbeddingFn
    prompts: dict[str, dict[str, str]]
    prompt_variant: str
    max_expand_attempts: int
    max_generation_attempts: int
