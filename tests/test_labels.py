from __future__ import annotations

import pandas as pd

from data.labels import (
    LABEL_PERCENTILE,
    LABEL_PURCHASE,
    LABEL_THRESHOLD,
    attach_labels,
    binary_purchase,
    game_percentile_cutoffs,
    thresholded_hours,
    within_game_percentile,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "item_id": ["long", "long", "short", "short"],
            "hours": [100.0, 80.0, 1.0, 3.0],
        }
    )


def test_purchase_is_all_true():
    df = _frame()
    assert binary_purchase(df).all()


def test_threshold_uses_strict_greater_than():
    df = _frame()
    out = thresholded_hours(df, threshold=2.0)
    assert list(out) == [True, True, False, True]


def test_percentile_cutoffs_ignore_held_out_rows():
    df = pd.DataFrame(
        {
            "user_id": ["a", "b", "c"],
            "item_id": ["g", "g", "g"],
            "hours": [1.0, 10.0, 1000.0],
        }
    )
    train = df.iloc[:2]
    cutoffs = game_percentile_cutoffs(train, percentile=50)
    # Train hours are 1 and 10; p50 is 5.5. The 1000h test row must not move the cutoff.
    assert float(cutoffs["g"]) == 5.5
    labeled = within_game_percentile(df, cutoffs)
    assert list(labeled) == [False, True, True]


def test_attach_labels_does_not_use_test_hours_for_cutoffs():
    train = pd.DataFrame(
        {"user_id": ["a"], "item_id": ["g"], "hours": [4.0]}
    )
    test = pd.DataFrame(
        {"user_id": ["b"], "item_id": ["g"], "hours": [400.0]}
    )
    train_l, test_l, cutoffs = attach_labels(train, test, hours_threshold=2.0, percentile=50)
    assert float(cutoffs["g"]) == 4.0
    assert bool(train_l[LABEL_PURCHASE].iloc[0]) is True
    assert bool(train_l[LABEL_THRESHOLD].iloc[0]) is True
    assert bool(test_l[LABEL_PERCENTILE].iloc[0]) is True
