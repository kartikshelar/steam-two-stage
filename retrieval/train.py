from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.io import utc_now, write_json
from data.labels import LABEL_PERCENTILE, LABEL_PURCHASE, LABEL_THRESHOLD
from data.paths import MODELS, RESULTS, SPLITS, ensure_dirs
from retrieval.content import ContentVocab
from retrieval.two_tower import TwoTower

LABEL_MAP = {
    "purchase": LABEL_PURCHASE,
    "threshold": LABEL_THRESHOLD,
    "percentile": LABEL_PERCENTILE,
}


class InteractionDataset(Dataset):
    def __init__(self, user_idx: np.ndarray, item_idx: np.ndarray):
        self.user_idx = user_idx.astype(np.int64)
        self.item_idx = item_idx.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.user_idx))

    def __getitem__(self, i: int) -> tuple[int, int]:
        return int(self.user_idx[i]), int(self.item_idx[i])


def build_vocab(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    users = sorted(train_df["user_id"].astype(str).unique())
    items = sorted(train_df["item_id"].astype(str).unique())
    user_to_idx = {u: i + 1 for i, u in enumerate(users)}  # 0 = pad
    item_to_idx = {it: i + 1 for i, it in enumerate(items)}
    return user_to_idx, item_to_idx


def logq_from_counts(item_idx: np.ndarray, n_items: int) -> torch.Tensor:
    counts = np.bincount(item_idx, minlength=n_items).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    log_q = np.log(np.clip(freq, 1e-12, 1.0))
    log_q[0] = 0.0
    return torch.tensor(log_q, dtype=torch.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one(
    train_df: pd.DataFrame,
    label_col: str,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    cfg: dict[str, Any],
    device: torch.device,
    content_tables: dict[str, np.ndarray] | None = None,
    content_vocab: ContentVocab | None = None,
) -> tuple[TwoTower, dict[str, Any]]:
    pos = train_df.loc[train_df[label_col]].copy()
    pos = pos[pos["user_id"].astype(str).isin(user_to_idx) & pos["item_id"].astype(str).isin(item_to_idx)]
    u = pos["user_id"].astype(str).map(user_to_idx).to_numpy()
    i = pos["item_id"].astype(str).map(item_to_idx).to_numpy()
    n_users = max(user_to_idx.values()) + 1
    n_items = max(item_to_idx.values()) + 1
    use_content = content_tables is not None
    ds = InteractionDataset(u, i)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    model = TwoTower(
        n_users,
        n_items,
        dim=int(cfg["embedding_dim"]),
        use_content=use_content,
        n_genres=int(content_vocab.n_genres) if content_vocab else 0,
        n_tags=int(content_vocab.n_tags) if content_vocab else 0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    log_q = None
    if cfg.get("logq_correction", True):
        log_q = logq_from_counts(i, n_items).to(device)
    temperature = float(cfg["temperature"])
    id_dropout = float(cfg.get("id_dropout", 0.0)) if use_content else 0.0
    genre_t = tag_t = num_t = None
    if use_content:
        genre_t = torch.tensor(content_tables["genre_ids"], device=device, dtype=torch.long)
        tag_t = torch.tensor(content_tables["tag_ids"], device=device, dtype=torch.long)
        num_t = torch.tensor(content_tables["numeric"], device=device, dtype=torch.float32)
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        losses = []
        for users, items in tqdm(loader, desc=f"{label_col} epoch {epoch+1}", leave=False):
            users = users.to(device)
            items = items.to(device)
            loss = model.inbatch_loss(
                users,
                items,
                log_q=log_q,
                temperature=temperature,
                genre_ids=None if genre_t is None else genre_t[items],
                tag_ids=None if tag_t is None else tag_t[items],
                numeric=None if num_t is None else num_t[items],
                id_dropout=id_dropout,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "loss": mean_loss, "n_batches": len(losses)})
        print(f"{label_col} epoch {epoch+1}/{cfg['epochs']} loss={mean_loss:.4f}")
    stats = {
        "n_positives": int(len(pos)),
        "n_users_positive": int(pos["user_id"].nunique()),
        "n_items_positive": int(pos["item_id"].nunique()),
        "n_batches_per_epoch": int(len(loader)),
        "use_content": use_content,
        "id_dropout": id_dropout,
        "history": history,
    }
    return model, stats


def save_checkpoint(
    model: TwoTower,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    path: Path,
    extra: dict[str, Any],
    content_vocab: ContentVocab | None = None,
    train_item_ids: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": model.n_users,
            "n_items": model.n_items,
            "dim": model.dim,
            "use_content": model.use_content,
            "n_genres": model.n_genres,
            "n_tags": model.n_tags,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "content_vocab": None if content_vocab is None else content_vocab.to_json(),
            "train_item_ids": train_item_ids or [],
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[TwoTower, dict[str, Any]]:
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    model = TwoTower(
        blob["n_users"],
        blob["n_items"],
        dim=blob["dim"],
        use_content=bool(blob.get("use_content", False)),
        n_genres=int(blob.get("n_genres", 0) or 0),
        n_tags=int(blob.get("n_tags", 0) or 0),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train two-tower models for each label (Phase 0).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--label", choices=list(LABEL_MAP) + ["all"], default="all")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_dirs()
    train_df = pd.read_parquet(SPLITS / f"train_{args.split}_labeled.parquet")
    user_to_idx, item_to_idx = build_vocab(train_df)
    write_json(
        MODELS / f"vocab_{args.split}.json",
        {"n_users": len(user_to_idx), "n_items": len(item_to_idx)},
    )
    device = pick_device()
    labels = list(LABEL_MAP) if args.label == "all" else [args.label]
    run = {"created_at": utc_now(), "split": args.split, "device": str(device), "labels": {}}
    for name in labels:
        model, stats = train_one(train_df, LABEL_MAP[name], user_to_idx, item_to_idx, config["retrieval"], device)
        ckpt = MODELS / f"two_tower_{args.split}_{name}.pt"
        save_checkpoint(model, user_to_idx, item_to_idx, ckpt, stats)
        run["labels"][name] = {**{k: v for k, v in stats.items() if k != "history"}, "checkpoint": str(ckpt)}
        write_json(RESULTS / f"train_log_{args.split}_{name}.json", stats)
    write_json(RESULTS / f"train_run_{args.split}.json", run)
    print(json.dumps(run, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
