"""Two-stage serving: frozen two-tower retrieve 500, PIT GBDT rerank.

Not production traffic. Same feature function as training (`PITState.features`).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from data.paths import MODELS, PROCESSED, ROOT, SPLITS
from eval.metrics import user_seen_items
from features.point_in_time import PITState, _item_meta, freeze_before
from ranking.features import RANKING_COLS
from ranking.rerank import rerank
from ranking.train_gbdt import load_booster, predict_booster
from retrieval.infer import FrozenRetriever
from retrieval.train import pick_device

BUNDLE = ROOT / "serving" / "bundle"


def cutoff_unix_from_manifest(manifest_path: Path | None = None) -> int:
    path = manifest_path or (
        BUNDLE / "manifest.json" if (BUNDLE / "manifest.json").exists() else SPLITS / "manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(manifest["temporal_cutoff"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    return int(cutoff.timestamp())


class RecsService:
    def __init__(
        self,
        retriever: FrozenRetriever,
        booster,
        pit: PITState,
        seen: dict[str, set[str]],
        meta_by_item: dict,
        cutoff_unix: int,
        retrieve_k: int = 500,
        default_k: int = 20,
    ):
        self.retriever = retriever
        self.booster = booster
        self.pit = pit
        self.seen = seen
        self.meta_by_item = meta_by_item
        self.cutoff_unix = cutoff_unix
        self.retrieve_k = retrieve_k
        self.default_k = default_k

    def feature_frame(self, user_id: str, scored: list[tuple[str, float]]) -> pd.DataFrame:
        rows = []
        for item, ret_s in scored:
            feats = self.pit.features(user_id, item, self.cutoff_unix, self.meta_by_item.get(item))
            feats["retrieval_score"] = float(ret_s)
            rows.append(feats)
        if not rows:
            return pd.DataFrame(columns=RANKING_COLS)
        return pd.DataFrame(rows)[RANKING_COLS]

    def recommend(self, user_id: str, k: int | None = None) -> dict:
        k = self.default_k if k is None else int(k)
        k = max(1, min(k, self.retrieve_k))
        retrieved = self.retriever.retrieve([user_id], self.seen, self.retrieve_k)
        scored = retrieved.get(user_id) or []
        items = [c for c, _ in scored]
        if not items:
            return {"user_id": user_id, "items": [], "retrieve_k": self.retrieve_k, "k": k}
        frame = self.feature_frame(user_id, scored)
        probs = predict_booster(self.booster, frame, RANKING_COLS)
        order = rerank(items, probs)
        score_map = {item: float(p) for item, p in zip(items, probs)}
        top = [{"item_id": item, "score": score_map[item]} for item in order[:k]]
        return {"user_id": user_id, "items": top, "retrieve_k": self.retrieve_k, "k": k}


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def load_service(
    device: torch.device | None = None,
    retrieve_k: int = 500,
) -> RecsService:
    device = device or pick_device()
    games = pd.read_parquet(_first_existing(BUNDLE / "games.parquet", PROCESSED / "games.parquet"))
    ckpt = _first_existing(
        BUNDLE / "two_tower_content_temporal_purchase.pt",
        MODELS / "two_tower_content_temporal_purchase.pt",
    )
    booster_path = _first_existing(BUNDLE / "gbdt_ranker.txt", MODELS / "gbdt_ranker.txt")
    retriever = FrozenRetriever(ckpt, games, device)
    booster = load_booster(booster_path)
    pit_path = BUNDLE / "pit.pkl"
    if pit_path.exists():
        blob = pickle.loads(pit_path.read_bytes())
        pit = blob["pit"]
        seen = blob["seen"]
        meta = blob.get("meta") or _item_meta(games)
        cutoff = int(blob["cutoff_unix"])
    else:
        reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
        train = pd.read_parquet(SPLITS / "train_temporal_labeled.parquet")
        cutoff = cutoff_unix_from_manifest()
        pit = freeze_before(reviews, games, cutoff)
        seen = {str(u): {str(i) for i in items} for u, items in user_seen_items(train).items()}
        meta = _item_meta(games)
    return RecsService(
        retriever,
        booster,
        pit,
        seen,
        meta,
        cutoff,
        retrieve_k=retrieve_k,
    )
