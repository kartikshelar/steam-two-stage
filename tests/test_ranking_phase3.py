from __future__ import annotations

import numpy as np
import torch

from features.point_in_time import FEATURE_COLS, PITState
from ranking.features import RANKING_COLS
from ranking.rerank import inject_positive, rerank
from retrieval.index import retrieve_scored


def test_ranking_cols_are_pit_plus_retrieval_score():
    assert RANKING_COLS[-1] == "retrieval_score"
    assert FEATURE_COLS == RANKING_COLS[:-1]


def test_rerank_is_a_permutation_by_score():
    items = ["a", "b", "c"]
    scores = np.array([0.1, 0.9, 0.4])
    out = rerank(items, scores)
    assert out == ["b", "c", "a"]
    assert set(out) == set(items)


def test_rerank_stable_on_ties():
    items = ["a", "b", "c"]
    scores = np.array([0.5, 0.5, 0.1])
    assert rerank(items, scores) == ["a", "b", "c"]


def test_inject_positive_noop_when_present():
    ranked = [("x", 0.9), ("y", 0.2)]
    assert inject_positive(ranked, "x", 0.0) == ranked


def test_inject_positive_appends_when_missing():
    ranked = [("x", 0.9), ("y", 0.2)]
    out = inject_positive(ranked, "z", -0.5)
    assert out[-1] == ("z", -0.5)
    assert [c for c, _ in out[:2]] == ["x", "y"]


def test_retrieve_scored_returns_inner_products():
    users = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    items = torch.nn.functional.normalize(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]), dim=-1
    )
    idx_to_item = ["", "a", "b", "c"]
    recs = retrieve_scored(users, items, idx_to_item, [set(), set()], k=2)
    assert recs[0][0][0] == "a"
    assert recs[1][0][0] == "b"
    assert recs[0][0][1] >= recs[0][1][1]


def test_pit_eval_at_cutoff_ignores_later_events():
    import pandas as pd

    hist = pd.DataFrame(
        [
            ("u1", "x", 100, 1.0),
            ("u2", "x", 200, 1.0),
            ("u1", "y", 500, 1.0),  # after cutoff
        ],
        columns=["user_id", "item_id", "ts", "hours"],
    )
    cutoff = 300
    state = PITState()
    for rec in hist.itertuples(index=False):
        if int(rec.ts) >= cutoff:
            break
        state.observe(str(rec.user_id), str(rec.item_id), int(rec.ts), float(rec.hours), [])
    feats = state.features("u3", "x", cutoff, None)
    assert feats["item_n"] == 2.0
    # y is after cutoff and must not count
    feats_y = state.features("u3", "y", cutoff, None)
    assert feats_y["item_n"] == 0.0
