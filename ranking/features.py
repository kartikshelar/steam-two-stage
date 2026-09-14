"""Ranking feature column lists. Values come from features/point_in_time.py."""

from features.point_in_time import FEATURE_COLS

# Two-stage glue. Not a PIT interaction feature; it is the frozen retriever's score.
RANKING_COLS = list(FEATURE_COLS) + ["retrieval_score"]

__all__ = ["FEATURE_COLS", "RANKING_COLS"]
