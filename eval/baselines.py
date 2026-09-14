from __future__ import annotations

import numpy as np
import pandas as pd


def most_popular_ranking(train_df: pd.DataFrame, label_col: str | None = None) -> list:
    frame = train_df if label_col is None else train_df.loc[train_df[label_col]]
    return frame.groupby("item_id").size().sort_values(ascending=False).index.tolist()


def random_ranking(item_ids: list, seed: int = 42) -> list:
    rng = np.random.RandomState(seed)
    items = list(item_ids)
    rng.shuffle(items)
    return items


def rank_for_users(
    global_ranked: list,
    users: list,
    seen: dict,
    k: int,
) -> dict:
    """Same global order for every user, skipping items already seen in train."""
    out = {}
    for user in users:
        banned = seen.get(user, set())
        picked = []
        for item in global_ranked:
            if item in banned:
                continue
            picked.append(item)
            if len(picked) >= k:
                break
        out[user] = picked
    return out


def recommended_game_length(
    user_ranked: dict,
    item_length: pd.Series,
    k: int = 10,
) -> tuple[float, int]:
    """Mean of (median train hours of top-k recs) over users. The label-experiment qualitative metric."""
    values = []
    for ranked in user_ranked.values():
        lengths = [item_length[i] for i in ranked[:k] if i in item_length.index]
        if lengths:
            values.append(float(np.median(lengths)))
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), int(len(values))
