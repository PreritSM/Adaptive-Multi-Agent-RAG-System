from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Weight given to score-distribution certainty vs. embedding-space agreement
# in score_uncertainty(); the remainder (1 - _SCORE_WEIGHT) goes to cosine agreement.
_SCORE_WEIGHT: float = 0.5


# ---------------------------------------------------------------------------
# Cosine similarity utilities
# ---------------------------------------------------------------------------

def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Return cosine similarity in [0, 1] between two 1-D vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(a, b)) / (norm_a * norm_b)
    return max(0.0, min(1.0, raw))


def pairwise_cosine_matrix(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return (N, N) pairwise cosine similarity matrix for a batch of embeddings."""
    norms: NDArray[np.float32] = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalised: NDArray[np.float32] = embeddings / norms
    matrix: NDArray[np.float32] = np.clip(normalised @ normalised.T, 0.0, 1.0)
    return matrix


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def _score_certainty(retrieval_scores: list[float]) -> float:
    """
    Certainty derived from the retrieval score distribution: a sharply peaked
    distribution (low entropy) means the top result stands out → high certainty.
    Returns a value in [0, 1], where 1 means minimal entropy (maximal certainty).
    """
    if not retrieval_scores:
        return 0.0
    scores: NDArray[np.float64] = np.asarray(retrieval_scores, dtype=np.float64)
    total = float(scores.sum())
    if total <= 0.0:
        return 0.0
    probs = scores / total
    nonzero = probs[probs > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = float(np.log(len(scores))) if len(scores) > 1 else 1.0
    if max_entropy == 0.0:
        return 1.0
    normalised_entropy = entropy / max_entropy
    return 1.0 - normalised_entropy


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
    score_certainty = _score_certainty(retrieval_scores)

    if doc_embeddings.shape[0] == 0:
        mean_similarity = 0.0
    else:
        similarities = [
            cosine_similarity(query_embedding, doc_embeddings[i])
            for i in range(doc_embeddings.shape[0])
        ]
        mean_similarity = sum(similarities) / len(similarities)

    combined = _SCORE_WEIGHT * score_certainty + (1.0 - _SCORE_WEIGHT) * mean_similarity
    return max(0.0, min(1.0, combined))


def is_uncertain(uncertainty_score: float, threshold: float) -> bool:
    """Return True when the system lacks sufficient confidence to answer."""
    return uncertainty_score < threshold
