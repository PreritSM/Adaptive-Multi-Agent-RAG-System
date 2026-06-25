from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Cosine similarity utilities
# ---------------------------------------------------------------------------

def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Return cosine similarity in [0, 1] between two 1-D vectors."""
    # TODO: implement dot(a, b) / (norm(a) * norm(b)) with zero-division guard
    raise NotImplementedError


def pairwise_cosine_matrix(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return (N, N) pairwise cosine similarity matrix for a batch of embeddings."""
    # TODO: normalise rows, compute dot product matrix
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def score_uncertainty(
    query_embedding: NDArray[np.float32],
    doc_embeddings: NDArray[np.float32],
    retrieval_scores: list[float],
) -> float:
    """
    Combine retrieval score distribution and embedding-space agreement into a
    single uncertainty value in [0, 1].

    Higher value = higher confidence (lower uncertainty).
    """
    # TODO: compute score entropy + mean cosine similarity, combine via weighted sum
    raise NotImplementedError


def is_uncertain(uncertainty_score: float, threshold: float) -> bool:
    """Return True when the system lacks sufficient confidence to answer."""
    return uncertainty_score < threshold
