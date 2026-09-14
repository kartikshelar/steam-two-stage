from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.io import append_metrics, utc_now, write_json
from data.labels import LABEL_PERCENTILE, LABEL_PURCHASE, LABEL_THRESHOLD
from data.paths import MODELS, RESULTS, SPLITS, ensure_dirs
from eval.baselines import most_popular_ranking, rank_for_users, recommended_game_length
from eval.metrics import popularity_deciles, summarize_ranking, user_ground_truth, user_seen_items
from retrieval.index import retrieve_for_users
from retrieval.two_tower import TwoTower

LABEL_MAP = {
    "purchase": LABEL_PURCHASE,
    "threshold": LABEL_THRESHOLD,
    "percentile": LABEL_PERCENTILE,
}

# Pre-registered before any training run. Do not edit after the first logged result.
PRE_REGISTERED = {
    "prediction": (
        "Naive thresholding of play time (hours > 2.0) will over-recommend long games "
        "relative to binary purchase and relative to within-game percentile labels."
    ),
    "operationalization": (
        "For each model, take top-10 retrieved items per test user (temporal split, "
        "full catalog, train items masked). Map each item to its median train-set hours "
        "(a proxy for game length). Average those per-user medians. Predicted order of "
        "this statistic: thresholded > purchase > percentile."
    ),
    "written_at": "2026-09-13T before any two-tower training run",
}


def load_checkpoint(path: Path, device: torch.device) -> tuple[TwoTower, dict, dict]:
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    model = TwoTower(blob["n_users"], blob["n_items"], dim=blob["dim"])
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob["user_to_idx"], blob["item_to_idx"]


@torch.no_grad()
def model_rankings(
    model: TwoTower,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    users: list[str],
    seen: dict,
    k: int,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    idx_to_item = [""] * model.n_items
    for item, idx in item_to_idx.items():
        idx_to_item[idx] = item
    item_vectors = model.all_item_vectors(device)
    out: dict = {}
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        uidx = torch.tensor([user_to_idx[u] for u in batch_users], device=device, dtype=torch.long)
        uvec = model.encode_users(uidx)
        seen_idx = []
        for u in batch_users:
            banned = set()
            for item in seen.get(u, set()):
                if item in item_to_idx:
                    banned.add(item_to_idx[item])
            seen_idx.append(banned)
        recs = retrieve_for_users(uvec, item_vectors, idx_to_item, seen_idx, k)
        for u, ranked in zip(batch_users, recs):
            out[u] = ranked
    return out


def item_median_hours(train_df: pd.DataFrame) -> pd.Series:
    hours = pd.to_numeric(train_df["hours"], errors="coerce")
    tmp = train_df.loc[hours.notna(), ["item_id"]].copy()
    tmp["hours"] = hours[hours.notna()]
    return tmp.groupby("item_id")["hours"].median()


def evaluate_split(
    split: str,
    config: dict[str, Any],
    device: torch.device,
    run_id: str,
) -> pd.DataFrame:
    train = pd.read_parquet(SPLITS / f"train_{split}_labeled.parquet")
    test = pd.read_parquet(SPLITS / f"test_{split}_labeled.parquet")
    ks = list(config["eval"]["ks"])
    max_k = max(ks)
    max_users = int(config["eval"]["max_eval_users"])
    seen = user_seen_items(train)
    lengths = item_median_hours(train)
    rows = []

    for label_name, label_col in LABEL_MAP.items():
        truth = user_ground_truth(test, label_col)
        catalog_truth = user_ground_truth(test, LABEL_PURCHASE)
        users = [u for u in truth if u in seen]
        users = users[:max_users]
        if not users:
            continue

        pop = most_popular_ranking(train, label_col)
        pop_ranked = rank_for_users(pop, users, seen, max_k)
        deciles = popularity_deciles(train, label_col)
        pop_rows = summarize_ranking(pop_ranked, {u: truth[u] for u in users}, ks, item_decile=deciles)
        pop_len, n_len = recommended_game_length(pop_ranked, lengths, k=10)
        for rec in pop_rows:
            rows.append(
                {
                    "run_id": run_id,
                    "phase": 0,
                    "split": split,
                    "label": label_name,
                    "model": "most_popular",
                    **rec,
                }
            )
        rows.append(
            {
                "run_id": run_id,
                "phase": 0,
                "split": split,
                "label": label_name,
                "model": "most_popular",
                "metric": "recommended_game_length_top10",
                "k": 10,
                "value": pop_len,
                "n_users": n_len,
                "slice": "overall",
            }
        )

        ckpt = MODELS / f"two_tower_{split}_{label_name}.pt"
        if not ckpt.exists():
            rows.append(
                {
                    "run_id": run_id,
                    "phase": 0,
                    "split": split,
                    "label": label_name,
                    "model": "two_tower",
                    "metric": "missing_checkpoint",
                    "k": 0,
                    "value": float("nan"),
                    "n_users": 0,
                    "slice": "overall",
                }
            )
            continue

        model, user_to_idx, item_to_idx = load_checkpoint(ckpt, device)
        model_users = [u for u in users if u in user_to_idx]
        ranked = model_rankings(model, user_to_idx, item_to_idx, model_users, seen, max_k, device)
        model_rows = summarize_ranking(ranked, {u: truth[u] for u in model_users}, ks, item_decile=deciles)
        for rec in model_rows:
            rows.append(
                {
                    "run_id": run_id,
                    "phase": 0,
                    "split": split,
                    "label": label_name,
                    "model": "two_tower",
                    **rec,
                }
            )
        # Same recs scored against purchase ground truth so the three models share a label.
        purchase_rows = summarize_ranking(
            ranked,
            {u: catalog_truth[u] for u in model_users if u in catalog_truth},
            ks,
            item_decile=None,
        )
        for rec in purchase_rows:
            rec = dict(rec)
            rec["slice"] = "purchase_ground_truth"
            rows.append(
                {
                    "run_id": run_id,
                    "phase": 0,
                    "split": split,
                    "label": label_name,
                    "model": "two_tower",
                    **rec,
                }
            )
        rec_len, n_len = recommended_game_length(ranked, lengths, k=10)
        rows.append(
            {
                "run_id": run_id,
                "phase": 0,
                "split": split,
                "label": label_name,
                "model": "two_tower",
                "metric": "recommended_game_length_top10",
                "k": 10,
                "value": rec_len,
                "n_users": n_len,
                "slice": "overall",
            }
        )

    frame = pd.DataFrame(rows)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 label experiment: train + eval three labels.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_dirs()
    write_json(RESULTS / "pre_registration.json", PRE_REGISTERED)

    if not args.skip_train:
        from retrieval.train import main as train_main

        train_args = ["--split", args.split]
        if args.config:
            train_args.extend(["--config", str(args.config)])
        train_main(train_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = utc_now().replace(":", "").replace("-", "")
    frame = evaluate_split(args.split, config, device, run_id)
    out_csv = RESULTS / f"label_experiment_{args.split}_{run_id}.csv"
    frame.to_csv(out_csv, index=False)
    append_metrics(frame.to_dict(orient="records"), filename="metrics.csv")
    write_json(
        RESULTS / f"label_experiment_{args.split}_{run_id}.json",
        {
            "run_id": run_id,
            "split": args.split,
            "pre_registration": PRE_REGISTERED,
            "n_metric_rows": int(len(frame)),
            "csv": str(out_csv),
        },
    )
    print(f"wrote {out_csv} ({len(frame)} rows)")
    if "recommended_game_length_top10" in set(frame["metric"]):
        q = frame[frame["metric"] == "recommended_game_length_top10"]
        print(q[["model", "label", "value", "n_users"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
