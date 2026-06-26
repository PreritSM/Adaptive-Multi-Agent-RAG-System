from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any, Protocol

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]  # no py.typed marker

from src.api.schemas import RetrievalMode


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EmbeddingFn(Protocol):
    def __call__(self, text: str) -> list[float]: ...


class Retriever(Protocol):
    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
        "so", "yet", "both", "either", "neither", "just", "also", "very",
        "that", "this", "these", "those", "it", "its", "what", "which", "who",
        "how", "when", "where", "why", "i", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    }
)

# Lines that look like section headers: markdown headings or SHORT ALL-CAPS titles
_HEADER_RE: re.Pattern[str] = re.compile(
    r"^(#{1,6}\s+\S.{0,80}|[A-Z][A-Z\d\s\-:]{3,60}[A-Z\d])$",
    re.MULTILINE,
)

# Routing thresholds
_SHORT_TOKEN_LIMIT: int = 7   # ≤ this → candidate for sparse
_LONG_TOKEN_LIMIT: int = 12   # ≥ this → route to dense
# Fraction of tokens that are NOT stopwords to be considered keyword-dense
_KEYWORD_DENSITY_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Adaptive query classifier
# ---------------------------------------------------------------------------


class QueryClassifier:
    """
    Route a query to the most appropriate retrieval mode using lightweight
    lexical heuristics — no model inference required.

    Rules (evaluated in order):
      1. Short (≤ _SHORT_TOKEN_LIMIT tokens) + keyword-dense → sparse / BM25
         Rationale: short keyword queries benefit from exact-term matching.
      2. Long (≥ _LONG_TOKEN_LIMIT tokens) → dense / ANN
         Rationale: longer, conversational queries carry semantic structure
         that vector search captures better than bag-of-words.
      3. Everything else → hybrid (RRF of both)
    """

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _keyword_density(tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        content = [t for t in tokens if t not in _STOPWORDS]
        return len(content) / len(tokens)

    def classify(self, query: str) -> RetrievalMode:
        tokens = self._tokenize(query)
        n = len(tokens)
        density = self._keyword_density(tokens)

        if n <= _SHORT_TOKEN_LIMIT and density >= _KEYWORD_DENSITY_THRESHOLD:
            return RetrievalMode.sparse
        if n >= _LONG_TOKEN_LIMIT:
            return RetrievalMode.dense
        return RetrievalMode.hybrid


# ---------------------------------------------------------------------------
# Metadata-aware chunker
# ---------------------------------------------------------------------------


def _extract_section_header(text: str) -> str | None:
    """Return the first header-like line found in *text*, stripped of markdown."""
    match = _HEADER_RE.search(text)
    if match:
        return re.sub(r"^#+\s*", "", match.group(0)).strip()
    return None


def chunk_documents(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Split *texts* into overlapping chunks and enrich each chunk's metadata.

    Each output Document carries:
      - source_file   : originating file path or synthetic doc_N label
      - page_number   : page index from caller metadata (None if absent)
      - section_header: first detected heading within the chunk (None if absent)
      - chunk_index   : 0-based position within the parent document
      - doc_index     : 0-based index of the parent document in *texts*
      - chunk_id      : deterministic 16-char hex digest for deduplication
    """
    if metadatas and len(metadatas) != len(texts):
        raise ValueError(
            f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    base_metas: list[dict[str, Any]] = (
        metadatas if metadatas else [{} for _ in texts]
    )

    chunks: list[Document] = []
    for doc_idx, (text, base_meta) in enumerate(zip(texts, base_metas)):
        raw_chunks: list[str] = splitter.split_text(text)

        source_file: str = str(
            base_meta.get("source_file") or base_meta.get("source", f"doc_{doc_idx}")
        )
        page_number: int | None = base_meta.get("page_number") or base_meta.get("page")

        for chunk_idx, chunk_text in enumerate(raw_chunks):
            section = _extract_section_header(chunk_text)
            chunk_id = hashlib.sha256(
                f"{source_file}:{page_number}:{chunk_idx}:{chunk_text[:64]}".encode()
            ).hexdigest()[:16]

            enriched: dict[str, Any] = {
                **base_meta,
                "source_file": source_file,
                "page_number": page_number,
                "section_header": section,
                "chunk_index": chunk_idx,
                "doc_index": doc_idx,
                "chunk_id": chunk_id,
            }
            chunks.append(Document(page_content=chunk_text, metadata=enriched))

    return chunks


# ---------------------------------------------------------------------------
# Shared RRF helper
# ---------------------------------------------------------------------------


def _rrf_merge(
    ranked_lists: list[list[Document]],
    weights: list[float],
    rrf_k: int,
    top_k: int,
) -> tuple[list[Document], list[float]]:
    """
    Weighted Reciprocal Rank Fusion across multiple ranked document lists.

    score(d) = Σ weight_i / (rrf_k + rank_i(d))

    Scores are normalised to [0, 1] before returning.
    Documents absent from a list simply contribute zero from that list's term.
    """
    fused: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked):
            cid: str = str(doc.metadata.get("chunk_id", doc.page_content[:64]))
            fused[cid] = fused.get(cid, 0.0) + weight / (rrf_k + rank + 1)
            doc_map[cid] = doc

    sorted_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:top_k]
    docs = [doc_map[cid] for cid in sorted_ids]
    raw = [fused[cid] for cid in sorted_ids]
    max_score = max(raw) if raw else 1.0
    norm = [s / max_score for s in raw]
    return docs, norm


# ---------------------------------------------------------------------------
# Dense retriever (Qdrant approximate nearest-neighbour)
# ---------------------------------------------------------------------------


class DenseRetriever:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedding_fn: EmbeddingFn,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedding_fn = embedding_fn

    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        vector: list[float] = self._embedding_fn(query)
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        docs: list[Document] = []
        scores: list[float] = []
        for hit in hits:
            payload: dict[str, Any] = hit.payload or {}  # raw Qdrant JSON — Any is correct at this boundary
            docs.append(
                Document(
                    page_content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )
            # Qdrant cosine similarity is in [-1, 1]; clamp to [0, 1]
            scores.append(max(0.0, min(1.0, float(hit.score))))
        return docs, scores


# ---------------------------------------------------------------------------
# Sparse retriever (BM25 over an in-memory corpus)
# ---------------------------------------------------------------------------


class SparseRetriever:
    def __init__(self, corpus: list[Document]) -> None:
        self._corpus = corpus
        tokenised: list[list[str]] = [
            doc.page_content.lower().split() for doc in corpus
        ]
        self._bm25: BM25Okapi = BM25Okapi(tokenised)

    async def retrieve(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        query_tokens: list[str] = query.lower().split()
        raw: NDArray[np.float64] = self._bm25.get_scores(query_tokens)

        top_k = min(k, len(self._corpus))
        top_indices: NDArray[np.intp] = np.argsort(raw)[::-1][:top_k]

        max_score = float(raw[top_indices[0]]) if top_k > 0 else 1.0
        norm_factor = max_score if max_score > 0.0 else 1.0

        docs: list[Document] = []
        scores: list[float] = []
        for idx in top_indices:
            docs.append(self._corpus[int(idx)])
            scores.append(float(raw[idx]) / norm_factor)
        return docs, scores


# ---------------------------------------------------------------------------
# Hybrid retriever (weighted RRF of dense + sparse)
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
        (dense_docs, _), (sparse_docs, _) = await asyncio.gather(
            self._dense.retrieve(query, k),
            self._sparse.retrieve(query, k),
        )
        return _rrf_merge(
            ranked_lists=[dense_docs, sparse_docs],
            weights=[self._alpha, 1.0 - self._alpha],
            rrf_k=self._rrf_k,
            top_k=k,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_retriever(
    mode: str,
    client: AsyncQdrantClient,
    collection: str,
    embedding_fn: EmbeddingFn,
    corpus: list[Document],
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> DenseRetriever | SparseRetriever | HybridRetriever:
    if mode == "dense":
        return DenseRetriever(client, collection, embedding_fn)
    if mode == "sparse":
        return SparseRetriever(corpus)
    if mode == "hybrid":
        dense = DenseRetriever(client, collection, embedding_fn)
        sparse = SparseRetriever(corpus)
        return HybridRetriever(dense, sparse, alpha=alpha, rrf_k=rrf_k)
    raise ValueError(f"Unknown retrieval mode: {mode!r}")
