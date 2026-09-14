"""Fails if a ranking feature uses data from t onward.

These tests are the Phase 2 gate. They do not touch the Steam dump.
"""
from __future__ import annotations

import pandas as pd
import pytest

from features.point_in_time import (
    PITState,
    assert_no_future_in_pit,
    build_global_leak_stats,
    features_at_events,
    naive_leaked_features,
)


def _events() -> pd.DataFrame:
    # t = unix seconds. Day = 86400.
    rows = [
        ("u1", "x", 1000, 1.0),
        ("u2", "x", 2000, 2.0),
        ("u1", "y", 2000 + 40 * 86400, 3.0),
        ("u3", "x", 2000 + 80 * 86400, 4.0),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "ts", "hours"])


def test_pit_item_count_excludes_current_and_future():
    hist = _events()
    feats = features_at_events(hist)
    # First review of x: no prior
    r0 = feats[(feats.user_id == "u1") & (feats.item_id == "x")].iloc[0]
    assert r0["item_n"] == 0
    assert r0["user_n"] == 0
    # Second review of x sees only the first
    r1 = feats[(feats.user_id == "u2") & (feats.item_id == "x")].iloc[0]
    assert r1["item_n"] == 1
    assert r1["user_n"] == 0
    # Last review of x sees two earlier x reviews, not itself
    r3 = feats[(feats.user_id == "u3") & (feats.item_id == "x")].iloc[0]
    assert r3["item_n"] == 2


def test_pit_30d_window_excludes_old_and_future():
    hist = _events()
    feats = features_at_events(hist)
    r3 = feats[(feats.user_id == "u3") & (feats.item_id == "x")].iloc[0]
    # u2's review of x is 80d before u3; not in 30d. u1 is even older.
    assert r3["item_n_30d"] == 0
    assert r3["item_n"] == 2


def test_adding_a_future_event_does_not_change_past_pit():
    hist = _events()
    before = features_at_events(hist)
    extra = pd.DataFrame(
        [{"user_id": "u9", "item_id": "x", "ts": 10_000_000, "hours": 99.0}]
    )
    after = features_at_events(pd.concat([hist, extra], ignore_index=True))
    past = after[after["ts"] < 10_000_000].sort_values(["ts", "user_id", "item_id"]).reset_index(drop=True)
    before = before.sort_values(["ts", "user_id", "item_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(before["item_n"], past["item_n"], check_names=False)
    pd.testing.assert_series_equal(before["user_n"], past["user_n"], check_names=False)


def test_assert_no_future_in_pit_matches_strict_count():
    hist = _events()
    feats = features_at_events(hist)
    for row in feats.itertuples(index=False):
        assert_no_future_in_pit(row._asdict(), hist, row.user_id, row.item_id, int(row.ts))


def test_assert_no_future_in_pit_fails_on_global_groupby():
    hist = _events()
    leaked_item_n = float((hist["item_id"] == "x").sum())
    fake = {"item_n": leaked_item_n, "user_n": 0.0}
    with pytest.raises(AssertionError, match="item_n leaked"):
        assert_no_future_in_pit(fake, hist, "u1", "x", 1000)


def test_leaked_item_n_includes_future_and_current():
    hist = _events()
    meta = {}
    stats = build_global_leak_stats(hist, meta)
    leaked = naive_leaked_features(
        "u1", "x", 1000, meta=None, **{k: stats[k] for k in stats}
    )
    pit = PITState().features("u1", "x", 1000, None)
    assert leaked["item_n"] == 3  # all three x reviews
    assert pit["item_n"] == 0
    assert leaked["item_n"] > pit["item_n"]
