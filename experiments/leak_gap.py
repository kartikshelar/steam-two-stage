from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.io import append_metrics, utc_now, write_json
from data.paths import FEATURE_DATA, MODELS, PROCESSED, RESULTS, SPLITS, ensure_dirs
from features.point_in_time import (
    PITState,
    _item_meta,
    build_global_leak_stats,
    naive_leaked_features,
)
from ranking.train_gbdt import (
    classification_metrics,
    feature_importance,
    ndcg_at_k_grouped,
    predict_proba,
    save_model,
    train_binary,
)


def _pack(feats: dict[str, float], user: str, item: str, t: int, label: int, split: str) -> dict[str, Any]:
    row = dict(feats)
    row["user_id"] = user
    row["item_id"] = item
    row["ts"] = t
    row["label"] = int(label)
    row["split"] = split
    return row


def build_example_tables(
    reviews: pd.DataFrame,
    games: pd.DataFrame,
    train_keys: set[tuple[str, str]],
    test_keys: set[tuple[str, str]],
    catalog: np.ndarray,
    n_neg_train: int,
    n_neg_test: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    meta_by_item = _item_meta(games)
    print("building leaked global stats (this pass sees the future on purpose)", flush=True)
    leak_stats = build_global_leak_stats(reviews, meta_by_item)
    state = PITState()
    rng = np.random.RandomState(seed)
    n_catalog = int(len(catalog))
    user_seen: dict[str, set[str]] = {}

    train_pit: list[dict] = []
    train_leak: list[dict] = []
    test_pit: list[dict] = []
    test_leak: list[dict] = []

    ordered = reviews.sort_values(["ts", "user_id", "item_id"])
    for rec in tqdm(ordered.itertuples(index=False), total=len(ordered), desc="PIT sweep"):
        user = str(rec.user_id)
        item = str(rec.item_id)
        t = int(rec.ts)
        hours = float(rec.hours) if rec.hours == rec.hours else float("nan")
        meta_i = meta_by_item.get(item)
        key = (user, item)
        emit_split = "train" if key in train_keys else ("test" if key in test_keys else None)
        if emit_split is not None:
            n_neg = n_neg_train if emit_split == "train" else n_neg_test
            pit_pos = state.features(user, item, t, meta_i)
            leak_pos = naive_leaked_features(user, item, t, meta=meta_i, **leak_stats)
            pit_bucket = train_pit if emit_split == "train" else test_pit
            leak_bucket = train_leak if emit_split == "train" else test_leak
            pit_bucket.append(_pack(pit_pos, user, item, t, 1, emit_split))
            leak_bucket.append(_pack(leak_pos, user, item, t, 1, emit_split))
            seen = user_seen.get(user, set())
            got: list[str] = []
            tries = 0
            while len(got) < n_neg and tries < n_neg * 30:
                tries += 1
                cand = str(catalog[rng.randint(0, n_catalog)])
                if cand == item or cand in seen or cand in got:
                    continue
                got.append(cand)
            for cand in got:
                m = meta_by_item.get(cand)
                pit_bucket.append(_pack(state.features(user, cand, t, m), user, cand, t, 0, emit_split))
                leak_bucket.append(
                    _pack(naive_leaked_features(user, cand, t, meta=m, **leak_stats), user, cand, t, 0, emit_split)
                )
        state.observe(user, item, t, hours, (meta_i or {}).get("genres") or [])
        user_seen.setdefault(user, set()).add(item)

    pit_train = pd.DataFrame(train_pit)
    leak_train = pd.DataFrame(train_leak)
    pit_test = pd.DataFrame(test_pit)
    leak_test = pd.DataFrame(test_leak)
    stats = {
        "n_train_pos_keys": len(train_keys),
        "n_test_pos_keys": len(test_keys),
        "n_train_rows_pit": int(len(pit_train)),
        "n_test_rows_pit": int(len(pit_test)),
        "n_train_positives_emitted": int(pit_train["label"].sum()) if len(pit_train) else 0,
        "n_test_positives_emitted": int(pit_test["label"].sum()) if len(pit_test) else 0,
    }
    return pit_train, leak_train, pit_test, leak_test, stats, leak_stats


def _eval_block(name: str, model, test: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    scores = predict_proba(model, test)
    cls = classification_metrics(test["label"].to_numpy(), scores)
    out = {f"{name}_{k}": v for k, v in cls.items()}
    for k in (10, 50):
        out[f"{name}_ndcg_{k}"] = ndcg_at_k_grouped(test, scores, k=k)
    return out, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: PIT vs leaked GBDT, report the gap.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_dirs()
    FEATURE_DATA.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    train = pd.read_parquet(SPLITS / "train_temporal_labeled.parquet")
    test = pd.read_parquet(SPLITS / "test_temporal_labeled.parquet")

    n_train = int(config["leak"]["max_train_positives"])
    n_test = int(config["leak"]["max_test_positives"])
    if len(train) > n_train:
        train = train.sample(n=n_train, random_state=int(config["seed"]))
    if len(test) > n_test:
        test = test.sample(n=n_test, random_state=int(config["seed"]))
    train_keys = set(zip(train["user_id"].astype(str), train["item_id"].astype(str)))
    test_keys = set(zip(test["user_id"].astype(str), test["item_id"].astype(str)))
    catalog = np.array(sorted(set(games["item_id"].astype(str)) | set(reviews["item_id"].astype(str))))

    pit_train, leak_train, pit_test, leak_test, built, leak_stats = build_example_tables(
        reviews,
        games,
        train_keys,
        test_keys,
        catalog,
        n_neg_train=int(config["leak"]["n_neg_train"]),
        n_neg_test=int(config["leak"]["n_neg_test"]),
        seed=int(config["seed"]),
    )
    if pit_train.empty or pit_test.empty:
        raise RuntimeError("PIT sweep emitted no rows. Check split keys against reviews.")

    pit_train.to_parquet(FEATURE_DATA / "pit_train.parquet", index=False)
    leak_train.to_parquet(FEATURE_DATA / "leaked_train.parquet", index=False)
    pit_test.to_parquet(FEATURE_DATA / "pit_test.parquet", index=False)
    leak_test.to_parquet(FEATURE_DATA / "leaked_test.parquet", index=False)

    # Real-data leak gate: 200 random PIT test positives must match a strict past count.
    pos = pit_test[pit_test["label"] == 1]
    sample = pos.sample(n=min(200, len(pos)), random_state=int(config["seed"]))
    n_checked = 0

    for rec in sample.itertuples(index=False):
        item = str(rec.item_id)
        user = str(rec.user_id)
        t = int(rec.ts)
        item_times = leak_stats["item_ts_all"].get(item, [])
        user_times = leak_stats["user_ts_all"].get(user, [])
        n_item_at_t = bisect.bisect_right(item_times, t) - bisect.bisect_left(item_times, t)
        n_user_at_t = bisect.bisect_right(user_times, t) - bisect.bisect_left(user_times, t)
        if n_item_at_t == 1:
            expect = bisect.bisect_left(item_times, t)
            if int(rec.item_n) != expect:
                raise AssertionError(f"real-data PIT item_n failed: got {rec.item_n} expected {expect} item={item} t={t}")
        if n_user_at_t == 1:
            expect_u = bisect.bisect_left(user_times, t)
            if int(rec.user_n) != expect_u:
                raise AssertionError(f"real-data PIT user_n failed: got {rec.user_n} expected {expect_u} user={user} t={t}")
        n_checked += 1

    print("training PIT GBDT", flush=True)
    pit_model = train_binary(pit_train, config.get("gbdt"))
    print("training leaked GBDT", flush=True)
    leak_model = train_binary(leak_train, config.get("gbdt"))
    save_model(pit_model, MODELS / "gbdt_pit.txt")
    save_model(leak_model, MODELS / "gbdt_leaked.txt")

    pit_metrics, pit_scores = _eval_block("pit", pit_model, pit_test)
    leak_metrics, leak_scores = _eval_block("leaked", leak_model, leak_test)

    gap = {
        "auc_gap_leaked_minus_pit": leak_metrics["leaked_auc"] - pit_metrics["pit_auc"],
        "ap_gap_leaked_minus_pit": leak_metrics["leaked_average_precision"] - pit_metrics["pit_average_precision"],
        "ndcg_10_gap_leaked_minus_pit": leak_metrics["leaked_ndcg_10"] - pit_metrics["pit_ndcg_10"],
    }

    pit_imp = feature_importance(pit_model)
    leak_imp = feature_importance(leak_model)
    pit_imp.to_csv(RESULTS / "gbdt_pit_importance.csv", index=False)
    leak_imp.to_csv(RESULTS / "gbdt_leaked_importance.csv", index=False)

    run_id = utc_now().replace(":", "").replace("-", "")
    payload = {
        "run_id": run_id,
        "phase": 2,
        "split": "temporal",
        "note": (
            "AUC/NDCG use sampled negatives at the same (user, t) as each positive. "
            "This is the leak comparison, not a retrieval number. Retrieval still "
            "evaluates against the full catalog in Phase 1."
        ),
        "n_leak_checks": n_checked,
        **built,
        **pit_metrics,
        **leak_metrics,
        **gap,
        "pit_top_feature": str(pit_imp.iloc[0]["feature"]) if len(pit_imp) else None,
        "leaked_top_feature": str(leak_imp.iloc[0]["feature"]) if len(leak_imp) else None,
    }
    write_json(RESULTS / f"leak_gap_{run_id}.json", payload)
    rows = []
    for key, value in {**pit_metrics, **leak_metrics, **gap}.items():
        rows.append(
            {
                "run_id": run_id,
                "phase": 2,
                "split": "temporal",
                "label": "purchase",
                "model": "gbdt_leaked" if key.startswith("leaked") or "gap" in key else "gbdt_pit",
                "metric": key,
                "k": 10 if "ndcg_10" in key else (50 if "ndcg_50" in key else 0),
                "value": value,
                "n_users": int(pit_test["user_id"].nunique()),
                "slice": "leak_experiment_sampled_candidates",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / f"leak_gap_{run_id}.csv", index=False)
    append_metrics(frame.to_dict(orient="records"))
    print(pd.Series(payload).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
