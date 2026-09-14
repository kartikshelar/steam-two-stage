"""Phase 3: GBDT ranker over frozen two-tower candidates, full-catalog e2e eval."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from data.labels import LABEL_PURCHASE, attach_labels
from data.paths import MODELS, PROCESSED, RESULTS, ROOT as REPO_ROOT, SPLITS, ensure_dirs
from eval.baselines import most_popular_ranking, rank_for_users
from eval.coldstart import cold_slices
from eval.metrics import popularity_deciles, summarize_ranking, user_ground_truth, user_seen_items
from features.point_in_time import FEATURE_COLS, PITState, _item_meta
from ranking.features import RANKING_COLS
from ranking.rerank import inject_positive, rerank
from ranking.train_gbdt import feature_importance, predict_proba, save_model, train_binary
from retrieval.infer import FrozenRetriever
from retrieval.train import pick_device


def _cutoff_unix(manifest: dict[str, Any]) -> int:
    cutoff = pd.Timestamp(manifest["temporal_cutoff"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    return int(cutoff.timestamp())


def _sample_train_queries(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    pos = train.loc[train[LABEL_PURCHASE], ["user_id", "item_id", "ts"]].copy()
    pos["user_id"] = pos["user_id"].astype(str)
    pos["item_id"] = pos["item_id"].astype(str)
    if len(pos) > n:
        pos = pos.sample(n=n, random_state=seed)
    return pos.reset_index(drop=True)


def _histories_for_users(reviews: pd.DataFrame, users: set[str]) -> dict[str, list[tuple[int, str]]]:
    mask = reviews["user_id"].astype(str).isin(users)
    sub = reviews.loc[mask, ["user_id", "item_id", "ts"]]
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for rec in sub.itertuples(index=False):
        out[str(rec.user_id)].append((int(rec.ts), str(rec.item_id)))
    for rows in out.values():
        rows.sort()
    return out


def _seen_before(hist: list[tuple[int, str]], t: int) -> set[str]:
    return {item for ts, item in hist if ts < t}


def _downsample_candidates(
    ranked: list[tuple[str, float]],
    positive: str,
    n_neg: int | None,
    rng: np.random.RandomState,
) -> list[tuple[str, float]]:
    if n_neg is None:
        return ranked
    pos_row = [(c, s) for c, s in ranked if c == positive]
    negs = [(c, s) for c, s in ranked if c != positive]
    if len(negs) > n_neg:
        idx = rng.choice(len(negs), size=n_neg, replace=False)
        negs = [negs[i] for i in idx]
    return pos_row + negs


def _metric_rows(
    run_id: str,
    split: str,
    slice_name: str,
    model_name: str,
    ranked: dict,
    truth: dict,
    ks: list[int],
    deciles: pd.Series | None,
) -> list[dict]:
    rows = []
    for rec in summarize_ranking(ranked, truth, ks, item_decile=deciles if slice_name == "overall" else None):
        rows.append(
            {
                "run_id": run_id,
                "phase": 3,
                "split": split,
                "label": "purchase",
                "model": model_name,
                **rec,
                "slice": rec["slice"] if rec["slice"] != "overall" else slice_name,
            }
        )
    return rows


def _score_retrieved(
    model,
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
        probs = predict_proba(model, frame, RANKING_COLS)
        offset = 0
        for user, items in spans:
            n = len(items)
            if n == 0:
                ranked[user] = []
                continue
            ranked[user] = rerank(items, probs[offset : offset + n])
            offset += n
    return ranked


def build_train_frame(
    reviews: pd.DataFrame,
    games: pd.DataFrame,
    queries: pd.DataFrame,
    candidates: list[list[tuple[str, float]]],
    cutoff_unix: int,
) -> tuple[pd.DataFrame, PITState, dict[str, Any]]:
    meta_by_item = _item_meta(games)
    want: dict[tuple[str, str], int] = {}
    for i, rec in enumerate(queries.itertuples(index=False)):
        want[(str(rec.user_id), str(rec.item_id))] = i

    n_feat = len(RANKING_COLS)
    # +1 per query in case the positive was injected
    cap = sum(len(c) for c in candidates)
    X = np.zeros((cap, n_feat), dtype=np.float32)
    y = np.zeros(cap, dtype=np.int8)
    user_ids: list[str] = []
    item_ids: list[str] = []
    ts_col: list[int] = []
    n_rows = 0
    n_emitted = 0
    n_pos = 0
    pos_of = queries["item_id"].astype(str).tolist()

    state = PITState()
    ordered = reviews.sort_values(["ts", "user_id", "item_id"])
    for rec in tqdm(ordered.itertuples(index=False), total=len(ordered), desc="PIT sweep"):
        t = int(rec.ts)
        if t >= cutoff_unix:
            break
        user = str(rec.user_id)
        item = str(rec.item_id)
        hours = float(rec.hours) if rec.hours == rec.hours else float("nan")
        key = (user, item)
        qi = want.get(key)
        if qi is not None:
            pos = pos_of[qi]
            for cand, ret_s in candidates[qi]:
                feats = state.features(user, cand, t, meta_by_item.get(cand))
                row = [float(feats[c]) for c in FEATURE_COLS] + [float(ret_s)]
                X[n_rows] = row
                y[n_rows] = 1 if cand == pos else 0
                user_ids.append(user)
                item_ids.append(cand)
                ts_col.append(t)
                n_rows += 1
                if cand == pos:
                    n_pos += 1
            n_emitted += 1
        meta_i = meta_by_item.get(item)
        state.observe(user, item, t, hours, (meta_i or {}).get("genres") or [])

    frame = pd.DataFrame(X[:n_rows], columns=RANKING_COLS)
    frame["user_id"] = user_ids
    frame["item_id"] = item_ids
    frame["ts"] = ts_col
    frame["label"] = y[:n_rows].astype(int)
    stats = {
        "n_train_queries_emitted": int(n_emitted),
        "n_train_rows": int(n_rows),
        "n_train_positives": int(n_pos),
        "n_train_queries_sampled": int(len(queries)),
    }
    return frame, state, stats


def _ids_only(scored: dict[str, list[tuple[str, float]]]) -> dict[str, list[str]]:
    return {u: [c for c, _ in recs] for u, recs in scored.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: rank retrieved candidates with PIT GBDT.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase3.yaml")
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_dirs()

    split = "temporal"
    seed = int(config["seed"])
    retrieve_k = int(config["retrieval"]["k"])
    n_train_q = int(args.max_train_queries or config["ranking"]["max_train_queries"])
    max_eval = int(args.max_eval_users or config["ranking"]["max_eval_users"])
    max_cold = int(config["ranking"].get("max_cold_users", max_eval))
    n_neg = config["ranking"].get("n_neg_train")
    n_neg_train = None if n_neg is None else int(n_neg)
    ks = list(config["eval"]["ks"])
    ckpt = REPO_ROOT / str(config["retrieval"]["checkpoint"])
    batch_size = int(config["retrieval"].get("batch_size", 512))

    print("Phase 3 — ranking on retrieved candidates (temporal split, frozen two-tower)", flush=True)

    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    train = pd.read_parquet(SPLITS / f"train_{split}_labeled.parquet")
    test_warm = pd.read_parquet(SPLITS / f"test_{split}_labeled.parquet")
    test_cold = pd.read_parquet(SPLITS / f"test_{split}_cold.parquet")
    manifest = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    cutoff_unix = _cutoff_unix(manifest)
    hours_threshold = float(config["labels"]["hours_threshold"])
    percentile = float(config["labels"]["game_percentile"])
    _, test_cold_l, _ = attach_labels(train, test_cold, hours_threshold, percentile)

    device = pick_device()
    print(f"loading frozen retriever {ckpt} device={device}", flush=True)
    retriever = FrozenRetriever(ckpt, games, device)

    queries = _sample_train_queries(train, n_train_q, seed)
    q_users = set(queries["user_id"].astype(str))
    print(f"building user histories for {len(q_users)} train-query users", flush=True)
    hist = _histories_for_users(reviews, q_users)
    seen_rows = [
        _seen_before(hist.get(str(u), []), int(t))
        for u, t in zip(queries["user_id"], queries["ts"])
    ]
    print(f"retrieving {retrieve_k} candidates for {len(queries)} train queries", flush=True)
    raw_cands = retriever.retrieve_rows(
        queries["user_id"].astype(str).tolist(),
        seen_rows,
        retrieve_k,
        batch_size=batch_size,
    )
    uvecs = retriever.user_vectors(queries["user_id"].astype(str).tolist())
    rng = np.random.RandomState(seed)
    n_injected = 0
    n_pos_in_k = 0
    train_cands: list[list[tuple[str, float]]] = []
    for i, rec in enumerate(queries.itertuples(index=False)):
        pos = str(rec.item_id)
        ranked = raw_cands[i]
        if any(c == pos for c, _ in ranked):
            n_pos_in_k += 1
        else:
            n_injected += 1
        score = retriever.item_score(uvecs[i], pos)
        ranked = inject_positive(ranked, pos, score)
        ranked = _downsample_candidates(ranked, pos, n_neg_train, rng)
        train_cands.append(ranked)

    print("PIT feature sweep for train rows + freeze at temporal cutoff", flush=True)
    train_frame, eval_state, built = build_train_frame(
        reviews, games, queries, train_cands, cutoff_unix
    )
    if train_frame.empty or int(train_frame["label"].sum()) == 0:
        raise RuntimeError("Phase 3 train frame is empty. Check queries against reviews.")

    print(f"training GBDT on {len(train_frame)} retrieved-candidate rows", flush=True)
    model = train_binary(train_frame, config.get("gbdt"), feature_cols=RANKING_COLS)
    save_model(model, MODELS / "gbdt_ranker.txt")
    imp = feature_importance(model)
    imp.to_csv(RESULTS / "gbdt_ranker_importance.csv", index=False)

    keep_users = set(train["user_id"].astype(str))
    keep_items = set(train["item_id"].astype(str))
    all_test = pd.concat([test_warm, test_cold_l], ignore_index=True)
    slices = cold_slices(all_test, keep_users, keep_items)
    slices["overall"] = test_warm
    seen = user_seen_items(train)
    seen = {str(u): {str(i) for i in items} for u, items in seen.items()}
    deciles = popularity_deciles(train, LABEL_PURCHASE)
    pop = most_popular_ranking(train, LABEL_PURCHASE)
    run_id = utc_now().replace(":", "").replace("-", "")
    meta_by_item = _item_meta(games)
    t_eval = cutoff_unix

    rows: list[dict] = []
    ceiling: dict[str, float] = {}

    for slice_name in ("overall", "cold_user", "cold_item"):
        frame = slices[slice_name]
        truth = user_ground_truth(frame, LABEL_PURCHASE)
        cap = max_eval if slice_name == "overall" else max_cold
        users = [u for u in truth][:cap]
        if not users:
            continue
        print(f"eval slice={slice_name} n_users={len(users)} retrieve_k={retrieve_k}", flush=True)
        pop_ranked = rank_for_users(pop, users, seen, max(ks))
        rows.extend(_metric_rows(run_id, split, slice_name, "most_popular", pop_ranked, {u: truth[u] for u in users}, ks, deciles))

        retrieved = retriever.retrieve(users, seen, retrieve_k, batch_size=batch_size)
        ret_only = _ids_only(retrieved)
        rows.extend(
            _metric_rows(run_id, split, slice_name, "two_tower_content", ret_only, {u: truth[u] for u in users}, ks, deciles)
        )
        rec500 = [float(len(set(ret_only[u][:retrieve_k]) & truth[u]) / len(truth[u])) for u in users]
        ceiling[slice_name] = float(np.mean(rec500))
        rows.append(
            {
                "run_id": run_id,
                "phase": 3,
                "split": split,
                "label": "purchase",
                "model": "two_tower_content",
                "metric": "recall",
                "k": retrieve_k,
                "value": ceiling[slice_name],
                "n_users": int(len(users)),
                "slice": slice_name,
            }
        )
        two_stage = _score_retrieved(model, users, retrieved, eval_state, t_eval, meta_by_item)
        rows.extend(
            _metric_rows(
                run_id, split, slice_name, "two_tower_then_gbdt", two_stage, {u: truth[u] for u in users}, ks, deciles
            )
        )

    # Real-data freeze check: eval item_n matches strict pre-cutoff counts.
    n_checked = 0
    past_counts = reviews.loc[reviews["ts"] < cutoff_unix, "item_id"].astype(str).value_counts()
    sample_items = [str(x) for x in train["item_id"].astype(str).drop_duplicates().head(25)]
    for item in sample_items:
        expect = int(past_counts.get(item, 0))
        got = int(eval_state.item_n.get(item, 0))
        if got != expect:
            raise AssertionError(f"eval PIT item_n leaked: item={item} got={got} expected={expect}")
        n_checked += 1

    zero_gain = imp.loc[imp["gain"] <= 0, "feature"].tolist()
    payload = {
        "run_id": run_id,
        "phase": 3,
        "split": split,
        "note": (
            "End-to-end Recall@K / NDCG@K are full-catalog metrics: retrieve 500 with the "
            "frozen two-tower, rerank those 500 with PIT GBDT. most-popular is reported on "
            "the same users. This is not sampled-negative ranking."
        ),
        "checkpoint": str(ckpt),
        "retrieve_k": retrieve_k,
        "cutoff_unix": cutoff_unix,
        "n_pit_item_checks": n_checked,
        "n_train_pos_in_retrieve_k": int(n_pos_in_k),
        "n_train_pos_injected": int(n_injected),
        "train_hit_rate_at_k": float(n_pos_in_k / max(len(queries), 1)),
        "recall_at_retrieve_k": ceiling,
        "top_feature": None if imp.empty else str(imp.iloc[0]["feature"]),
        "zero_gain_features": zero_gain,
        **built,
    }
    frame = pd.DataFrame(rows)
    out_csv = RESULTS / f"phase3_ranking_{run_id}.csv"
    frame.to_csv(out_csv, index=False)
    append_metrics(frame.to_dict(orient="records"))
    write_json(RESULTS / f"phase3_ranking_{run_id}.json", payload)

    headline = frame[
        (frame["metric"].isin(["recall", "ndcg"]))
        & (frame["k"].isin([10, 50, 100, retrieve_k]))
        & (frame["slice"].isin(["overall", "cold_user", "cold_item"]))
    ]
    print(headline.to_string(index=False), flush=True)
    print(imp.to_string(index=False), flush=True)
    print(f"wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
