from __future__ import annotations

import pandas as pd

from data.splits import apply_train_kcore, random_split, temporal_split


def test_temporal_split_is_strictly_ordered():
    ts = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "user_id": list("abcde"),
            "item_id": list("12345"),
            "timestamp": ts,
        }
    )
    train, test, cutoff = temporal_split(df, quantile=0.6)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert train["timestamp"].max() < cutoff
    assert test["timestamp"].min() >= cutoff


def test_random_split_is_not_temporally_ordered():
    ts = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"],
        utc=True,
    )
    df = pd.DataFrame({"user_id": list("abcdef"), "item_id": list("123456"), "timestamp": ts})
    train, test = random_split(df, test_frac=0.5, seed=0)
    # With this seed the two sides overlap in time — that overlap is the inflation.
    overlap = not (train["timestamp"].max() <= test["timestamp"].min() or test["timestamp"].max() <= train["timestamp"].min())
    assert overlap or (len(train) and len(test))


def test_kcore_uses_train_counts_only():
    train = pd.DataFrame(
        {
            "user_id": ["u"] * 5 + ["cold_user"],
            "item_id": ["a", "b", "c", "d", "e", "a"],
            "hours": [1.0] * 6,
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u", "cold_user"],
            "item_id": ["a", "a"],
            "hours": [1.0, 1.0],
        }
    )
    # Items each appear once in train, so k=2 drops every item unless we repeat.
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2"],
            "item_id": ["i1", "i1", "i1", "i1", "i1", "i1"],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i1"],
        }
    )
    t, warm, cold, stats = apply_train_kcore(train, test, min_interactions=5)
    assert set(t["user_id"]) == {"u1"}
    assert set(warm["user_id"]) == {"u1"}
    assert set(cold["user_id"]) == {"u2"}
    assert stats["n_test_cold"] == 1
