"""Freeze PIT once and copy serving artifacts into serving/bundle (gitignored)."""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.io import utc_now, write_json
from data.paths import MODELS, PROCESSED, SPLITS
from eval.metrics import user_seen_items
from features.point_in_time import _item_meta, freeze_before
from serving.service import BUNDLE, cutoff_unix_from_manifest


def export_bundle() -> Path:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(PROCESSED / "games.parquet")
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    train = pd.read_parquet(SPLITS / "train_temporal_labeled.parquet")
    cutoff = cutoff_unix_from_manifest()
    pit = freeze_before(reviews, games, cutoff)
    seen = {str(u): {str(i) for i in items} for u, items in user_seen_items(train).items()}
    blob = {
        "pit": pit,
        "seen": seen,
        "meta": _item_meta(games),
        "cutoff_unix": cutoff,
    }
    (BUNDLE / "pit.pkl").write_bytes(pickle.dumps(blob, protocol=4))
    shutil.copy2(PROCESSED / "games.parquet", BUNDLE / "games.parquet")
    shutil.copy2(MODELS / "two_tower_content_temporal_purchase.pt", BUNDLE / "two_tower_content_temporal_purchase.pt")
    shutil.copy2(MODELS / "gbdt_ranker.txt", BUNDLE / "gbdt_ranker.txt")
    shutil.copy2(SPLITS / "manifest.json", BUNDLE / "manifest.json")
    write_json(
        BUNDLE / "bundle_manifest.json",
        {
            "created": utc_now(),
            "cutoff_unix": cutoff,
            "n_seen_users": len(seen),
            "note": "PIT freeze at temporal cutoff. Same PITState.features as training.",
        },
    )
    return BUNDLE


def main() -> int:
    path = export_bundle()
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
