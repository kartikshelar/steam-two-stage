from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def recall_at_k(ranked: Sequence, ground_truth: Iterable, k: int) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in gt)
    return hits / len(gt)


def ndcg_at_k(ranked: Sequence, ground_truth: Iterable, k: int) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in gt:
            dcg += 1.0 / np.log2(i + 2)
    n_ideal = min(k, len(gt))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def user_ground_truth(test_df: pd.DataFrame, label_col: str | None = None) -> dict:
    frame = test_df
    if label_col is not None:
        frame = test_df.loc[test_df[label_col]].copy()
    out: dict = defaultdict(set)
    for user, item in zip(frame["user_id"].tolist(), frame["item_id"].tolist()):
        out[user].add(item)
    return dict(out)


def user_seen_items(train_df: pd.DataFrame) -> dict:
    out: dict = defaultdict(set)
    for user, item in zip(train_df["user_id"].tolist(), train_df["item_id"].tolist()):
        out[user].add(item)
    return dict(out)


def popularity_deciles(train_df: pd.DataFrame, label_col: str | None = None) -> pd.Series:
    frame = train_df if label_col is None else train_df.loc[train_df[label_col]]
    counts = frame.groupby("item_id").size().astype(float)
    # qcut fails on ties; rank first so every item gets a decile.
    ranks = counts.rank(method="first")
    n_bins = int(min(10, max(1, len(counts))))
    try:
        deciles = pd.qcut(ranks, n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(1, index=counts.index, dtype=int)
    return deciles.astype(int)


def summarize_ranking(
    user_ranked: dict,
    truth: dict,
    ks: Sequence[int],
    item_decile: pd.Series | None = None,
) -> list[dict]:
    """One row per metric. `user_ranked` maps user_id -> list of item ids (best first)."""
    rows: list[dict] = []
    users = [u for u in truth if u in user_ranked and truth[u]]
    if not users:
        return rows

    for k in ks:
        recs = [recall_at_k(user_ranked[u], truth[u], k) for u in users]
        ndcgs = [ndcg_at_k(user_ranked[u], truth[u], k) for u in users]
        rows.append(
            {
                "metric": "recall",
                "k": int(k),
                "value": float(np.mean(recs)),
                "n_users": int(len(users)),
                "slice": "overall",
            }
        )
        rows.append(
            {
                "metric": "ndcg",
                "k": int(k),
                "value": float(np.mean(ndcgs)),
                "n_users": int(len(users)),
                "slice": "overall",
            }
        )

    if item_decile is None:
        return rows

    max_k = max(ks)
    for decile in sorted(item_decile.unique()):
        slice_truth = {}
        for u in users:
            items = {i for i in truth[u] if item_decile.get(i) == decile}
            if items:
                slice_truth[u] = items
        if not slice_truth:
            continue
        for k in ks:
            recs = [recall_at_k(user_ranked[u], slice_truth[u], k) for u in slice_truth]
            rows.append(
                {
                    "metric": "recall",
                    "k": int(k),
                    "value": float(np.mean(recs)),
                    "n_users": int(len(slice_truth)),
                    "slice": f"popularity_decile_{int(decile)}",
                }
            )
        _ = max_k
    return rows
