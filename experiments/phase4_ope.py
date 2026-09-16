"""Phase 4: IPS/SNIPS, team-draft interleaving, MDE. No online experiment."""

from __future__ import annotations

import argparse
import json
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
from data.labels import LABEL_PURCHASE
from data.paths import MODELS, PROCESSED, RESULTS, ROOT as REPO_ROOT, SPLITS, ensure_dirs
from eval.baselines import most_popular_ranking, rank_for_users
from eval.metrics import recall_at_k, user_ground_truth, user_seen_items
from experiments.interleaving import compare_rankings
from experiments.ips import estimate_target_from_logs, ips_snips, simulate_bandit_logs
from experiments.mde import mde_table
from features.point_in_time import PITState, _item_meta, freeze_before
from ranking.features import RANKING_COLS
from ranking.rerank import rerank
from ranking.train_gbdt import load_booster, predict_booster
from retrieval.infer import FrozenRetriever
from retrieval.train import pick_device


def _cutoff_unix(manifest: dict[str, Any]) -> int:
    cutoff = pd.Timestamp(manifest["temporal_cutoff"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    return int(cutoff.timestamp())


def freeze_pit(reviews: pd.DataFrame, games: pd.DataFrame, cutoff_unix: int) -> PITState:
    return freeze_before(reviews, games, cutoff_unix)


def score_gbdt(
    booster,
    users: list[str],
    retrieved: dict[str, list[tuple[str, float]]],
    state: PITState,
    t: int,
    meta_by_item: dict,
    batch_users: int = 128,
) -> dict[str, list[str]]:
    ranked: dict[str, list[str]] = {}
    for start in range(0, len(users), batch_users):
        batch = users[start : start + batch_users]
        rows: list[dict] = []
        spans: list[tuple[str, list[str]]] = []
        for user in batch:
            recs = retrieved.get(user) or []
            items = [c for c, _ in recs]
            for item, ret_s in recs:
                feats = state.features(user, item, t, meta_by_item.get(item))
                feats["retrieval_score"] = float(ret_s)
                rows.append(feats)
            spans.append((user, items))
        if not rows:
            for user, items in spans:
                ranked[user] = items
            continue
        frame = pd.DataFrame(rows)
        probs = predict_booster(booster, frame, RANKING_COLS)
        offset = 0
        for user, items in spans:
            n = len(items)
            ranked[user] = rerank(items, probs[offset : offset + n]) if n else []
            offset += n
    return ranked


def _ids_only(scored: dict[str, list[tuple[str, float]]]) -> dict[str, list[str]]:
    return {u: [c for c, _ in recs] for u, recs in scored.items()}


def hit_at_k(ranked: dict[str, list[str]], truth: dict, users: list[str], k: int) -> float:
    hits = []
    for user in users:
        gt = truth[user]
        hits.append(1.0 if any(item in gt for item in ranked[user][:k]) else 0.0)
    return float(np.mean(hits)) if hits else float("nan")


def precision_at_1(ranked: dict[str, list[str]], truth: dict, users: list[str]) -> float:
    hits = []
    for user in users:
        recs = ranked[user]
        if not recs:
            hits.append(0.0)
            continue
        hits.append(1.0 if recs[0] in truth[user] else 0.0)
    return float(np.mean(hits)) if hits else float("nan")


def mean_recall(ranked: dict[str, list[str]], truth: dict, users: list[str], k: int) -> float:
    return float(np.mean([recall_at_k(ranked[u], truth[u], k) for u in users]))


def _metric_row(run_id: str, model: str, metric: str, k: int, value: float, n_users: int, slice_name: str) -> dict:
    return {
        "run_id": run_id,
        "phase": 4,
        "split": "temporal",
        "label": "purchase",
        "model": model,
        "metric": metric,
        "k": k,
        "value": value,
        "n_users": n_users,
        "slice": slice_name,
    }


def _flatten_est(run_id: str, model: str, prefix: str, est: dict, n_users: int) -> list[dict]:
    rows = []
    for key, value in est.items():
        if not isinstance(value, (int, float, np.floating)):
            continue
        rows.append(_metric_row(run_id, model, f"{prefix}_{key}", 0, float(value), n_users, "ope_simulated_logs"))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4: IPS/SNIPS, interleaving, MDE.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase4.yaml")
    parser.add_argument("--max-eval-users", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_dirs()

    seed = int(config["seed"])
    retrieve_k = int(config["retrieval"]["k"])
    max_eval = int(args.max_eval_users or config["ope"]["max_eval_users"])
    temperature = float(config["ope"]["temperature"])
    clip = config["ope"].get("clip")
    clip_v = None if clip is None else float(clip)
    n_boot = int(config["ope"]["n_bootstrap"])
    inter_k = int(config["interleave"]["k"])
    ckpt = REPO_ROOT / str(config["retrieval"]["checkpoint"])
    gbdt_path = REPO_ROOT / str(config["gbdt_model"])
    batch_size = int(config["retrieval"].get("batch_size", 512))

    print("Phase 4 — offline estimators. No online experiment, no real users.", flush=True)

    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    train = pd.read_parquet(SPLITS / "train_temporal_labeled.parquet")
    test_warm = pd.read_parquet(SPLITS / "test_temporal_labeled.parquet")
    manifest = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    cutoff_unix = _cutoff_unix(manifest)

    device = pick_device()
    print(f"loading frozen retriever + GBDT device={device}", flush=True)
    retriever = FrozenRetriever(ckpt, games, device)
    booster = load_booster(gbdt_path)

    print("freezing PIT at temporal cutoff", flush=True)
    eval_state = freeze_pit(reviews, games, cutoff_unix)
    meta_by_item = _item_meta(games)

    truth_all = user_ground_truth(test_warm, LABEL_PURCHASE)
    users = [u for u in truth_all][:max_eval]
    truth = {u: truth_all[u] for u in users}
    seen = {str(u): {str(i) for i in items} for u, items in user_seen_items(train).items()}
    pop = most_popular_ranking(train, LABEL_PURCHASE)

    print(f"retrieving {retrieve_k} for {len(users)} users", flush=True)
    retrieved = retriever.retrieve(users, seen, retrieve_k, batch_size=batch_size)
    tower = _ids_only(retrieved)
    print("scoring GBDT on retrieved candidates", flush=True)
    gbdt = score_gbdt(booster, users, retrieved, eval_state, cutoff_unix, meta_by_item)
    pop_ranked = rank_for_users(pop, users, seen, retrieve_k)

    rng = np.random.RandomState(seed)
    print("simulating two-tower bandit logs and estimating GBDT with IPS/SNIPS", flush=True)
    logs = simulate_bandit_logs(tower, gbdt, truth, users, temperature, rng)
    rng_boot = np.random.RandomState(seed + 1)
    est_gbdt = estimate_target_from_logs(logs, clip_v, rng_boot, n_boot)
    # Sanity: IPS of the logging policy on its own logs.
    est_self = ips_snips(logs["reward"], logs["p_log"], logs["p_log"], clip=clip_v)

    rng_mc = np.random.RandomState(seed + 2)
    logs_onpol = simulate_bandit_logs(gbdt, gbdt, truth, users, temperature, rng_mc)
    mc_target = float(logs_onpol["reward"].mean())
    mc_log = float(logs["reward"].mean())

    rng_iv = np.random.RandomState(seed + 3)
    iv_pop_gbdt = compare_rankings(pop_ranked, gbdt, truth, users, inter_k, rng_iv)
    rng_iv2 = np.random.RandomState(seed + 4)
    iv_tower_gbdt = compare_rankings(tower, gbdt, truth, users, inter_k, rng_iv2)

    n = len(users)
    greedy = {
        "most_popular_p_at_1": precision_at_1(pop_ranked, truth, users),
        "two_tower_p_at_1": precision_at_1(tower, truth, users),
        "gbdt_p_at_1": precision_at_1(gbdt, truth, users),
        "most_popular_hit_10": hit_at_k(pop_ranked, truth, users, 10),
        "two_tower_hit_10": hit_at_k(tower, truth, users, 10),
        "gbdt_hit_10": hit_at_k(gbdt, truth, users, 10),
        "most_popular_recall_10": mean_recall(pop_ranked, truth, users, 10),
        "two_tower_recall_10": mean_recall(tower, truth, users, 10),
        "gbdt_recall_10": mean_recall(gbdt, truth, users, 10),
    }

    mde_cfg = config["mde"]
    baseline_p = greedy["most_popular_hit_10"]
    rels = [float(x) for x in mde_cfg["relative_lifts"]]
    observed_rel = None
    if greedy["most_popular_hit_10"] > 0:
        observed_rel = greedy["gbdt_hit_10"] / greedy["most_popular_hit_10"] - 1.0
        if observed_rel > 0:
            rels = rels + [observed_rel]
    dau = [int(x) for x in mde_cfg["daily_active_users"]]
    mde_rows = mde_table(
        baseline_p,
        rels,
        dau,
        alpha=float(mde_cfg["alpha"]),
        power=float(mde_cfg["power"]),
    )

    run_id = utc_now().replace(":", "").replace("-", "")
    payload = {
        "run_id": run_id,
        "phase": 4,
        "split": "temporal",
        "disclaimer": (
            "No online experiment was run. There are no real users and no live "
            "feedback loop. IPS/SNIPS use simulated logs from a rank-softmax of "
            "the frozen two-tower over its retrieved 500. Interleaving treats "
            "held-out purchases as clicks. MDE daily_active_users values are "
            "hypothetical storefront assumptions, not this dump."
        ),
        "n_users": n,
        "retrieve_k": retrieve_k,
        "temperature": temperature,
        "clip": clip_v,
        "logging_policy": "two_tower_rank_softmax",
        "target_policy": "gbdt_rank_softmax",
        "mc_logging_reward": mc_log,
        "mc_target_reward": mc_target,
        "ips_target": est_gbdt,
        "ips_self_check": est_self,
        "interleave_pop_vs_gbdt": iv_pop_gbdt,
        "interleave_tower_vs_gbdt": iv_tower_gbdt,
        "greedy": greedy,
        "mde_baseline_metric": "most_popular_hit_at_10_bernoulli",
        "mde_observed_relative_lift_hit_10": observed_rel,
        "mde_assumptions": {
            "alpha": mde_cfg["alpha"],
            "power": mde_cfg["power"],
            "allocation": "50/50",
            "observations": "one recommendation session per user per day",
            "daily_active_users": dau,
            "note": "DAU figures are hypothetical. They are not Steam traffic and not this dataset.",
        },
        "mde": mde_rows,
    }
    write_json(RESULTS / f"phase4_ope_{run_id}.json", payload)

    rows: list[dict] = []
    for name, value in greedy.items():
        k = 1 if "p_at_1" in name else 10
        model = name.rsplit("_", 2)[0] if "p_at_1" in name else name.rsplit("_", 2)[0]
        # parse more carefully
        if name.startswith("most_popular"):
            model = "most_popular"
        elif name.startswith("two_tower"):
            model = "two_tower_content"
        else:
            model = "two_tower_then_gbdt"
        metric = "p_at_1" if "p_at_1" in name else ("hit" if "hit" in name else "recall")
        rows.append(_metric_row(run_id, model, metric, k, value, n, "ope_eval_users"))
    rows.extend(_flatten_est(run_id, "two_tower_then_gbdt", "ips_from_tower_logs", est_gbdt, n))
    rows.extend(_flatten_est(run_id, "two_tower_content", "ips_self", est_self, n))
    rows.append(_metric_row(run_id, "two_tower_content", "mc_bandit_reward", 0, mc_log, n, "ope_simulated_logs"))
    rows.append(_metric_row(run_id, "two_tower_then_gbdt", "mc_bandit_reward", 0, mc_target, n, "ope_simulated_logs"))
    for rec, tag in ((iv_pop_gbdt, "interleave_pop_vs_gbdt"), (iv_tower_gbdt, "interleave_tower_vs_gbdt")):
        for key, value in rec.items():
            rows.append(_metric_row(run_id, tag, key, inter_k, float(value), n, "interleave_gt_clicks"))
    for rec in mde_rows:
        rows.append(
            {
                "run_id": run_id,
                "phase": 4,
                "split": "temporal",
                "label": "purchase",
                "model": "mde",
                "metric": "days",
                "k": int(round(rec["relative_lift"] * 1000)),
                "value": rec["days"],
                "n_users": rec["n_total"],
                "slice": f"dau_{rec['daily_active_users']}_rel_{rec['relative_lift']:.4f}",
            }
        )

    frame = pd.DataFrame(rows)
    out_csv = RESULTS / f"phase4_ope_{run_id}.csv"
    frame.to_csv(out_csv, index=False)
    append_metrics(frame.to_dict(orient="records"))
    print(json.dumps({k: payload[k] for k in ("mc_logging_reward", "mc_target_reward", "greedy")}, indent=2), flush=True)
    print("IPS target", {k: est_gbdt[k] for k in ("ips", "snips", "ess", "clip_frac", "ips_boot_ci95_lo", "ips_boot_ci95_hi")}, flush=True)
    print("interleave pop vs gbdt", iv_pop_gbdt, flush=True)
    print(f"wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
