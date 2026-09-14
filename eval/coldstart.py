from __future__ import annotations

import pandas as pd


def train_counts(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return train_df.groupby("user_id").size(), train_df.groupby("item_id").size()


def cold_slices(
    test_df: pd.DataFrame,
    keep_users: set,
    keep_items: set,
) -> dict[str, pd.DataFrame]:
    """Cold start = under 5 train interactions, i.e. missing from the train k-core.

    `keep_users` / `keep_items` are the k-core train sets.
    """
    users = test_df["user_id"].astype(str)
    items = test_df["item_id"].astype(str)
    user_warm = users.isin(keep_users)
    item_warm = items.isin(keep_items)
    return {
        "warm": test_df.loc[user_warm & item_warm].copy(),
        "cold_user": test_df.loc[~user_warm].copy(),
        "cold_item": test_df.loc[~item_warm].copy(),
        "cold_user_or_item": test_df.loc[~user_warm | ~item_warm].copy(),
    }
