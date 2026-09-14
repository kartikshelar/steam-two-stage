from __future__ import annotations

import numpy as np


def rerank(candidates: list[str], scores: np.ndarray) -> list[str]:
    """Stable descending sort of `candidates` by `scores`. Same set, new order."""
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must be the same length")
    if not candidates:
        return []
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    return [candidates[int(i)] for i in order]


def inject_positive(
    ranked: list[tuple[str, float]],
    item: str,
    score: float,
) -> list[tuple[str, float]]:
    """Guarantee the labeled item is among candidates. Retrieval order otherwise unchanged."""
    if any(c == item for c, _ in ranked):
        return list(ranked)
    return list(ranked) + [(item, score)]
