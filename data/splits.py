from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.io import utc_now, write_json
from data.paths import PROCESSED, RESULTS, SPLITS, ensure_dirs

MANIFEST_NAME = "manifest.json"


def _require_reviews() -> pd.DataFrame:
    path = PROCESSED / "reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m data.prepare first.")
    return pd.read_parquet(path)


def temporal_split(reviews: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = reviews["timestamp"].quantile(quantile)
    # quantile on datetimes returns a Timestamp
    cutoff_ts = pd.Timestamp(cutoff)
    train = reviews.loc[reviews["timestamp"] < cutoff_ts].copy()
    test = reviews.loc[reviews["timestamp"] >= cutoff_ts].copy()
    return train, test, cutoff_ts


def random_split(reviews: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(reviews)) < test_frac
    test = reviews.loc[mask].copy()
    train = reviews.loc[~mask].copy()
    return train, test


def apply_train_kcore(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Keep users/items with >= k interactions in TRAIN only. Test is then filtered.

    Warm test: user and item both passed the train k-core.
    Cold test (returned separately): test rows whose user or item failed the train k-core.
    """
    if min_interactions <= 1:
        empty = train.iloc[0:0].copy()
        stats = {
            "min_train_interactions": min_interactions,
            "n_train_before": int(len(train)),
            "n_test_before": int(len(test)),
            "n_train_after": int(len(train)),
            "n_test_warm": int(len(test)),
            "n_test_cold": 0,
        }
        return train, test, empty, stats

    user_n = train.groupby("user_id").size()
    item_n = train.groupby("item_id").size()
    keep_users = set(user_n[user_n >= min_interactions].index)
    keep_items = set(item_n[item_n >= min_interactions].index)

    train_f = train[train["user_id"].isin(keep_users) & train["item_id"].isin(keep_items)].copy()
    # Recompute after the joint filter (standard iterative k-core, one extra pass is enough
    # for the common case; iterate to a fixed point).
    for _ in range(10):
        user_n = train_f.groupby("user_id").size()
        item_n = train_f.groupby("item_id").size()
        keep_users = set(user_n[user_n >= min_interactions].index)
        keep_items = set(item_n[item_n >= min_interactions].index)
        nxt = train_f[train_f["user_id"].isin(keep_users) & train_f["item_id"].isin(keep_items)]
        if len(nxt) == len(train_f) or nxt.empty:
            train_f = nxt.copy()
            break
        train_f = nxt.copy()

    warm = test[test["user_id"].isin(keep_users) & test["item_id"].isin(keep_items)].copy()
    cold = test[~(test["user_id"].isin(keep_users) & test["item_id"].isin(keep_items))].copy()
    stats = {
        "min_train_interactions": min_interactions,
        "n_train_before": int(len(train)),
        "n_test_before": int(len(test)),
        "n_train_after": int(len(train_f)),
        "n_users_train": int(train_f["user_id"].nunique()),
        "n_items_train": int(train_f["item_id"].nunique()),
        "n_test_warm": int(len(warm)),
        "n_test_cold": int(len(cold)),
        "n_cold_users_in_test": int(test.loc[~test["user_id"].isin(keep_users), "user_id"].nunique()),
        "n_cold_items_in_test": int(test.loc[~test["item_id"].isin(keep_items), "item_id"].nunique()),
    }
    return train_f.reset_index(drop=True), warm.reset_index(drop=True), cold.reset_index(drop=True), stats


def _split_stats(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_users_train": int(train["user_id"].nunique()),
        "n_users_test": int(test["user_id"].nunique()),
        "n_items_train": int(train["item_id"].nunique()),
        "n_items_test": int(test["item_id"].nunique()),
        "train_timestamp_min": str(train["timestamp"].min()) if len(train) else None,
        "train_timestamp_max": str(train["timestamp"].max()) if len(train) else None,
        "test_timestamp_min": str(test["timestamp"].min()) if len(test) else None,
        "test_timestamp_max": str(test["timestamp"].max()) if len(test) else None,
        "temporal_ordered": (
            bool(len(train) and len(test) and train["timestamp"].max() <= test["timestamp"].min())
        ),
    }


def build_splits(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_dirs()
    SPLITS.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLITS / MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists. Splits are frozen. Pass --force to rebuild."
        )

    reviews = _require_reviews()
    seed = int(config["seed"])
    q = float(config["splits"]["temporal_quantile"])
    test_frac = float(config["splits"]["random_test_frac"])
    k = int(config["splits"]["min_train_interactions"])

    t_train_raw, t_test_raw, cutoff = temporal_split(reviews, q)
    t_train, t_test_warm, t_test_cold, t_core = apply_train_kcore(t_train_raw, t_test_raw, k)

    r_train_raw, r_test_raw = random_split(reviews, test_frac, seed)
    r_train, r_test_warm, r_test_cold, r_core = apply_train_kcore(r_train_raw, r_test_raw, k)

    t_train.to_parquet(SPLITS / "train_temporal.parquet", index=False)
    t_test_warm.to_parquet(SPLITS / "test_temporal.parquet", index=False)
    t_test_cold.to_parquet(SPLITS / "test_temporal_cold.parquet", index=False)
    r_train.to_parquet(SPLITS / "train_random.parquet", index=False)
    r_test_warm.to_parquet(SPLITS / "test_random.parquet", index=False)
    r_test_cold.to_parquet(SPLITS / "test_random_cold.parquet", index=False)

    manifest = {
        "created_at": utc_now(),
        "frozen": True,
        "seed": seed,
        "temporal_quantile": q,
        "temporal_cutoff": str(cutoff),
        "random_test_frac": test_frac,
        "kcore": t_core,
        "temporal": _split_stats(t_train, t_test_warm),
        "temporal_raw_before_kcore": _split_stats(t_train_raw, t_test_raw),
        "random": _split_stats(r_train, r_test_warm),
        "random_raw_before_kcore": _split_stats(r_train_raw, r_test_raw),
        "random_core": r_core,
        "n_reviews_input": int(len(reviews)),
        "notes": [
            "Temporal split is the headline. Random split is a labeled inflation comparison only.",
            "k-core is computed on TRAIN interactions only, then applied to test (no test-count leakage).",
            "test_*_cold.parquet holds rows whose user or item missed the train k-core.",
        ],
    }
    write_json(manifest_path, manifest)
    write_json(RESULTS / "split_manifest.json", manifest)
    pd.DataFrame(
        [
            {"split": "temporal", **manifest["temporal"]},
            {"split": "random", **manifest["random"]},
        ]
    ).to_csv(RESULTS / "split_counts.csv", index=False)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze temporal + random splits.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_splits(load_config(args.config), force=args.force)
    print(json.dumps({"cutoff": manifest["temporal_cutoff"], "temporal": manifest["temporal"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
