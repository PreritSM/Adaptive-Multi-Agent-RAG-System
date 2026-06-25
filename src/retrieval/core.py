from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from qdrant_client import AsyncQdrantClient


# ---------------------------------------------------------------------------
# Shared protocol
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        ...


# ---------------------------------------------------------------------------
# Dense retriever (Qdrant approximate nearest-neighbour)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedding_fn,  # type: ignore[type-arg]
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedding_fn = embedding_fn

    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        # TODO: embed query, run ANN search, map hits to Document objects
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sparse retriever (BM25 over an in-memory corpus)
# ---------------------------------------------------------------------------

class SparseRetriever:
    def __init__(self, corpus: list[Document]) -> None:
        self._corpus = corpus
        tokenised = [doc.page_content.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenised)

    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        # TODO: score corpus, return top-k documents and normalised BM25 scores
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hybrid retriever (reciprocal rank fusion of dense + sparse)
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._alpha = alpha
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        # TODO: run dense and sparse in parallel, apply RRF, return merged list
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retriever(
    mode: str,
    client: AsyncQdrantClient,
    collection: str,
    embedding_fn,  # type: ignore[type-arg]
    corpus: list[Document],
) -> DenseRetriever | SparseRetriever | HybridRetriever:
    if mode == "dense":
        return DenseRetriever(client, collection, embedding_fn)
    if mode == "sparse":
        return SparseRetriever(corpus)
    if mode == "hybrid":
        dense = DenseRetriever(client, collection, embedding_fn)
        sparse = SparseRetriever(corpus)
        return HybridRetriever(dense, sparse)
    raise ValueError(f"Unknown retrieval mode: {mode!r}")
