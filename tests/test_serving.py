from __future__ import annotations

import numpy as np
import pandas as pd

from features.point_in_time import PITState
from ranking.features import RANKING_COLS
from serving.service import RecsService


class _DummyRetriever:
    user_to_idx = {"u1": 1}

    def retrieve(self, users, seen, k):
        return {users[0]: [("x", 0.9), ("y", 0.1)]}


def test_feature_frame_matches_pitstate_features():
    pit = PITState()
    pit.observe("u1", "z", 100, 2.0, ["Action"])
    meta = {"x": {"genres": ["Action"], "price": 10.0, "release_ts": 0}}
    svc = RecsService(
        retriever=_DummyRetriever(),
        booster=None,
        pit=pit,
        seen={"u1": {"z"}},
        meta_by_item=meta,
        cutoff_unix=200,
        retrieve_k=2,
        default_k=2,
    )
    raw = pit.features("u1", "x", 200, meta["x"])
    raw["retrieval_score"] = 0.5
    frame = svc.feature_frame("u1", [("x", 0.5)])
    assert list(frame.columns) == RANKING_COLS
    for col in RANKING_COLS:
        assert np.isclose(float(raw[col]), float(frame.iloc[0][col]))


def test_recommend_respects_k_and_candidate_set():
    pit = PITState()

    class Booster:
        def predict(self, x):
            # prefer the second row
            return np.array([0.1, 0.9])

    svc = RecsService(
        retriever=_DummyRetriever(),
        booster=Booster(),
        pit=pit,
        seen={},
        meta_by_item={},
        cutoff_unix=1,
        retrieve_k=2,
        default_k=1,
    )
    out = svc.recommend("u1", k=1)
    assert out["k"] == 1
    assert len(out["items"]) == 1
    assert out["items"][0]["item_id"] == "y"
    assert {row["item_id"] for row in svc.recommend("u1", k=2)["items"]} == {"x", "y"}


def test_percentile_interpolation():
    from serving.loadgen import percentile

    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([5.0], 99) == 5.0
    assert percentile([], 50) != percentile([], 50)  # nan
