from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.io import append_metrics, utc_now, write_json
from data.labels import LABEL_PURCHASE, attach_labels
from data.paths import MODELS, PROCESSED, RESULTS, SPLITS, ensure_dirs
from eval.baselines import most_popular_ranking, rank_for_users
from eval.coldstart import cold_slices
from eval.metrics import popularity_deciles, summarize_ranking, user_ground_truth, user_seen_items
from retrieval.content import build_content_tables, fit_content_vocab
from retrieval.index import FaissHnswIndex, overlap_at_k, retrieve_for_users
from retrieval.train import load_checkpoint, pick_device, save_checkpoint, train_one

HEADLINE_LABEL = "purchase"


def build_vocabs(train_df: pd.DataFrame, catalog_items: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    users = sorted(train_df["user_id"].astype(str).unique())
    user_to_idx = {u: i + 1 for i, u in enumerate(users)}
    item_to_idx = {it: i + 1 for i, it in enumerate(catalog_items)}
    return user_to_idx, item_to_idx


def catalog_items(train_df: pd.DataFrame, test_frames: list[pd.DataFrame], games: pd.DataFrame) -> list[str]:
    items: set[str] = set(games["item_id"].astype(str))
    items.update(train_df["item_id"].astype(str))
    for frame in test_frames:
        items.update(frame["item_id"].astype(str))
    return sorted(items)


@torch.no_grad()
def encode_users_mixed(
    model: TwoTower,
    users: list[str],
    user_to_idx: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    mean_u = F.normalize(model.user_emb.weight[1:].mean(dim=0, keepdim=True), dim=-1)
    out = mean_u.expand(len(users), -1).clone()
    known = [(i, user_to_idx[u]) for i, u in enumerate(users) if u in user_to_idx]
    if known:
        rows = torch.tensor([i for i, _ in known], device=device, dtype=torch.long)
        idx = torch.tensor([j for _, j in known], device=device, dtype=torch.long)
        out[rows] = model.encode_users(idx)
    return out


@torch.no_grad()
def model_rankings(
    model: TwoTower,
    blob: dict[str, Any],
    users: list[str],
    seen: dict,
    k: int,
    device: torch.device,
    content_tables: dict[str, np.ndarray] | None,
    allowed_item_idx: set[int] | None = None,
    batch_size: int = 512,
) -> dict:
    user_to_idx = blob["user_to_idx"]
    item_to_idx = blob["item_to_idx"]
    idx_to_item = [""] * model.n_items
    for item, idx in item_to_idx.items():
        idx_to_item[idx] = item
    train_ids = set(blob.get("train_item_ids") or [])
    train_mask = torch.zeros(model.n_items, dtype=torch.bool, device=device)
    train_mask[0] = True
    for item in train_ids:
        if item in item_to_idx:
            train_mask[item_to_idx[item]] = True
    genre = tag = numeric = None
    if model.use_content and content_tables is not None:
        genre = torch.tensor(content_tables["genre_ids"], device=device, dtype=torch.long)
        tag = torch.tensor(content_tables["tag_ids"], device=device, dtype=torch.long)
        numeric = torch.tensor(content_tables["numeric"], device=device, dtype=torch.float32)
    item_vectors = model.all_item_vectors(
        device,
        genre_ids=genre,
        tag_ids=tag,
        numeric=numeric,
        train_item_mask=train_mask if model.use_content else None,
    )
    if allowed_item_idx is not None:
        ban = torch.ones(model.n_items, dtype=torch.bool, device=device)
        for idx in allowed_item_idx:
            ban[idx] = False
        ban[0] = True
        item_vectors = item_vectors.clone()
        item_vectors[ban] = 0

    out: dict = {}
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        uvec = encode_users_mixed(model, batch_users, user_to_idx, device)
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


def metric_rows(
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
                "phase": 1,
                "split": split,
                "label": HEADLINE_LABEL,
                "model": model_name,
                **rec,
                "slice": rec["slice"] if rec["slice"] != "overall" else slice_name,
            }
        )
    return rows


def evaluate_split(
    split: str,
    config: dict[str, Any],
    device: torch.device,
    run_id: str,
    skip_train: bool,
) -> pd.DataFrame:
    train = pd.read_parquet(SPLITS / f"train_{split}_labeled.parquet")
    test_warm = pd.read_parquet(SPLITS / f"test_{split}_labeled.parquet")
    test_cold = pd.read_parquet(SPLITS / f"test_{split}_cold.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    hours_threshold = float(config["labels"]["hours_threshold"])
    percentile = float(config["labels"]["game_percentile"])
    _, test_cold_l, _ = attach_labels(train, test_cold, hours_threshold, percentile)

    catalog = catalog_items(train, [test_warm, test_cold_l], games)
    user_to_idx, item_to_idx = build_vocabs(train, catalog)
    train_item_ids = sorted(train["item_id"].astype(str).unique())
    vocab = fit_content_vocab(games, set(train_item_ids))
    tables = build_content_tables(item_to_idx, games, vocab)
    write_json(MODELS / f"content_vocab_{split}.json", vocab.to_json())

    ckpt = MODELS / f"two_tower_content_{split}_{HEADLINE_LABEL}.pt"
    if skip_train and ckpt.exists():
        model, blob = load_checkpoint(ckpt, device)
    else:
        model, stats = train_one(
            train,
            LABEL_PURCHASE,
            user_to_idx,
            item_to_idx,
            config["retrieval"],
            device,
            content_tables=tables,
            content_vocab=vocab,
        )
        extra = {k: v for k, v in stats.items() if k != "history"}
        extra["catalog_size"] = len(catalog)
        save_checkpoint(
            model,
            user_to_idx,
            item_to_idx,
            ckpt,
            extra,
            content_vocab=vocab,
            train_item_ids=train_item_ids,
        )
        write_json(RESULTS / f"train_log_phase1_{split}_{HEADLINE_LABEL}.json", stats)
        blob = {
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "train_item_ids": train_item_ids,
        }
        model.eval()

    keep_users = set(train["user_id"].astype(str))
    keep_items = set(train["item_id"].astype(str))
    all_test = pd.concat([test_warm, test_cold_l], ignore_index=True)
    slices = cold_slices(all_test, keep_users, keep_items)
    slices["overall"] = test_warm
    # seen items: everything the user reviewed in this split's train file.
    # Cold users with 1-4 interactions are absent from k-core train, so also
    # union the cold-file's complementary history isn't available; use train +
    # any warm history only. Remaining leakage is recommending a cold user's
    # other test items, which leave-one-out would handle; we mask train only.
    seen = user_seen_items(train)
    ks = list(config["eval"]["ks"])
    max_k = max(ks)
    max_users = int(config["eval"]["max_eval_users"])
    max_cold = int(config["eval"].get("max_cold_users", max_users))
    deciles = popularity_deciles(train, LABEL_PURCHASE)
    train_idx = {item_to_idx[i] for i in train_item_ids if i in item_to_idx}

    rows: list[dict] = []
    pop = most_popular_ranking(train, LABEL_PURCHASE)

    for slice_name, frame in slices.items():
        truth = user_ground_truth(frame, LABEL_PURCHASE)
        cap = max_users if slice_name == "overall" else max_cold
        users = [u for u in truth][:cap]
        if not users:
            continue
        pop_ranked = rank_for_users(pop, users, seen, max_k)
        rows.extend(
            metric_rows(run_id, split, slice_name, "most_popular", pop_ranked, {u: truth[u] for u in users}, ks, deciles)
        )
        ranked = model_rankings(
            model, blob, users, seen, max_k, device, tables, allowed_item_idx=None
        )
        rows.extend(
            metric_rows(run_id, split, slice_name, "two_tower_content", ranked, {u: truth[u] for u in users}, ks, deciles)
        )
        if slice_name == "overall":
            ranked_traincat = model_rankings(
                model, blob, users, seen, max_k, device, tables, allowed_item_idx=train_idx
            )
            rows.extend(
                metric_rows(
                    run_id,
                    split,
                    "overall_train_catalog",
                    "two_tower_content",
                    ranked_traincat,
                    {u: truth[u] for u in users},
                    ks,
                    None,
                )
            )

    # ANN index on the item vectors used at serving (full catalog, content encoding).
    genre = torch.tensor(tables["genre_ids"], device=device, dtype=torch.long)
    tag = torch.tensor(tables["tag_ids"], device=device, dtype=torch.long)
    numeric = torch.tensor(tables["numeric"], device=device, dtype=torch.float32)
    train_mask = torch.zeros(model.n_items, dtype=torch.bool, device=device)
    train_mask[0] = True
    for item in train_item_ids:
        if item in blob["item_to_idx"]:
            train_mask[blob["item_to_idx"][item]] = True
    item_vectors = model.all_item_vectors(
        device, genre_ids=genre, tag_ids=tag, numeric=numeric, train_item_mask=train_mask
    )
    vec_np = item_vectors.detach().cpu().numpy()
    ann = FaissHnswIndex(
        m=int(config["ann"]["m"]),
        ef_construction=int(config["ann"]["ef_construction"]),
        ef_search=int(config["ann"]["ef_search"]),
    )
    built = ann.build(vec_np)
    faiss_path = MODELS / f"ann_hnsw_{split}.faiss"
    try:
        import faiss

        faiss.write_index(ann.index, str(faiss_path))
    except Exception as exc:  # noqa: BLE001
        faiss_path = None
        built.extra["write_error"] = str(exc)

    probe_n = int(config["ann"].get("probe_users", 1000))
    probe_k = int(config["ann"].get("k", 500))
    probe_users = list(user_to_idx.keys())[:probe_n]
    uvec = encode_users_mixed(model, probe_users, user_to_idx, device)
    exact_scores = uvec @ item_vectors.t()
    exact_scores[:, 0] = -1e9
    exact_idx = torch.topk(exact_scores, k=probe_k, dim=1).indices.cpu().numpy()
    q = uvec.detach().cpu().numpy()
    t0 = time.perf_counter()
    _, ann_idx = ann.search(q, probe_k)
    query_ms = (time.perf_counter() - t0) * 1000.0
    # single-query latency (serving-shaped)
    single = []
    for row in q[: min(200, len(q))]:
        s0 = time.perf_counter()
        ann.search(row.reshape(1, -1), probe_k)
        single.append((time.perf_counter() - s0) * 1000.0)
    single_sorted = sorted(single)
    def _pct(vals, p):
        if not vals:
            return float("nan")
        return float(vals[min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1))))])

    ann_overlap = overlap_at_k(exact_idx, ann_idx, probe_k)
    ann_record = {
        "run_id": run_id,
        "split": split,
        "backend": built.backend,
        "n_vectors": built.n_vectors,
        "dim": built.dim,
        "build_ms": built.build_ms,
        "batch_query_ms_per_user": query_ms / max(len(probe_users), 1),
        "single_query_p50_ms": _pct(single_sorted, 50),
        "single_query_p95_ms": _pct(single_sorted, 95),
        "overlap_at_500": ann_overlap,
        "probe_users": len(probe_users),
        "k": probe_k,
        "index_path": None if faiss_path is None else str(faiss_path),
        **built.extra,
    }
    write_json(RESULTS / f"ann_{split}_{run_id}.json", ann_record)
    rows.append(
        {
            "run_id": run_id,
            "phase": 1,
            "split": split,
            "label": HEADLINE_LABEL,
            "model": "two_tower_content_ann",
            "metric": "ann_overlap",
            "k": probe_k,
            "value": ann_overlap,
            "n_users": len(probe_users),
            "slice": "ann_vs_exact",
        }
    )
    rows.append(
        {
            "run_id": run_id,
            "phase": 1,
            "split": split,
            "label": HEADLINE_LABEL,
            "model": "two_tower_content_ann",
            "metric": "ann_single_query_p50_ms",
            "k": probe_k,
            "value": ann_record["single_query_p50_ms"],
            "n_users": len(single),
            "slice": "ann_latency",
        }
    )
    return pd.DataFrame(rows)


def freeze_architecture(run_id: str, config: dict[str, Any]) -> None:
    payload = {
        "phase": 1,
        "frozen": True,
        "run_id": run_id,
        "frozen_at": utc_now(),
        "label": HEADLINE_LABEL,
        "user_tower": "user_id embedding, L2-normalized; unknown users use mean user vector",
        "item_tower": (
            "item_id embedding (unk/pad for items with no train interactions) + mean genre "
            "embedding + mean tag embedding + Linear(log1p price, price-missing, scaled year, "
            "year-missing), fused by Linear-ReLU-Linear, L2-normalized"
        ),
        "id_dropout": config["retrieval"].get("id_dropout", 0.2),
        "in_batch_negatives": True,
        "logq_correction": True,
        "embedding_dim": config["retrieval"]["embedding_dim"],
        "ann": "faiss.IndexHNSWFlat METRIC_INNER_PRODUCT",
        "eval": "exact inner product over the full metadata catalog; no sampled negatives",
        "do_not_change": (
            "Changing this architecture after Phase 1 invalidates ranking and "
            "experimentation comparisons."
        ),
    }
    write_json(RESULTS / "phase1_architecture.json", payload)
    write_json(MODELS / "phase1_architecture.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: content two-tower, ANN, cold-start eval.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1.yaml")
    parser.add_argument("--split", choices=["temporal", "random", "both"], default="both")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_dirs()
    device = pick_device()
    run_id = utc_now().replace(":", "").replace("-", "")
    splits = ["temporal", "random"] if args.split == "both" else [args.split]
    frames = []
    for split in splits:
        print(f"=== Phase 1 split={split} device={device} ===", flush=True)
        frames.append(evaluate_split(split, config, device, run_id, skip_train=args.skip_train))
    frame = pd.concat(frames, ignore_index=True)
    out_csv = RESULTS / f"phase1_retrieval_{run_id}.csv"
    frame.to_csv(out_csv, index=False)
    append_metrics(frame.to_dict(orient="records"), filename="metrics.csv")
    freeze_architecture(run_id, config)
    write_json(
        RESULTS / f"phase1_retrieval_{run_id}.json",
        {"run_id": run_id, "splits": splits, "n_metric_rows": int(len(frame)), "csv": str(out_csv)},
    )
    headline = frame[
        (frame["split"] == "temporal")
        & (frame["metric"].isin(["recall", "ndcg"]))
        & (frame["k"] == 10)
        & (frame["slice"].isin(["overall", "cold_user", "cold_item", "overall_train_catalog"]))
    ]
    print(headline.to_string(index=False), flush=True)
    print(f"wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
