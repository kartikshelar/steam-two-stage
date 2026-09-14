"""Inference for the frozen Phase 1 two-tower. Do not change the architecture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from retrieval.content import ContentVocab, build_content_tables
from retrieval.index import retrieve_scored
from retrieval.train import load_checkpoint
from retrieval.two_tower import TwoTower


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


class FrozenRetriever:
    """Exact inner-product retrieval over the full catalog. Architecture is frozen."""

    def __init__(self, ckpt: Path, games: pd.DataFrame, device: torch.device):
        self.device = device
        self.model, self.blob = load_checkpoint(ckpt, device)
        if not self.model.use_content:
            raise ValueError(f"{ckpt} is not the Phase 1 content two-tower")
        vocab_blob = self.blob.get("content_vocab")
        if not vocab_blob:
            raise ValueError(f"{ckpt} has no content vocab")
        vocab = ContentVocab.from_json(vocab_blob)
        self.user_to_idx: dict[str, int] = self.blob["user_to_idx"]
        self.item_to_idx: dict[str, int] = self.blob["item_to_idx"]
        self.train_item_ids = [str(x) for x in (self.blob.get("train_item_ids") or [])]
        self.tables = build_content_tables(self.item_to_idx, games, vocab)
        n_items = self.model.n_items
        self.idx_to_item = [""] * n_items
        for item, idx in self.item_to_idx.items():
            self.idx_to_item[idx] = item
        genre = torch.tensor(self.tables["genre_ids"], device=device, dtype=torch.long)
        tag = torch.tensor(self.tables["tag_ids"], device=device, dtype=torch.long)
        numeric = torch.tensor(self.tables["numeric"], device=device, dtype=torch.float32)
        train_mask = torch.zeros(n_items, dtype=torch.bool, device=device)
        train_mask[0] = True
        for item in self.train_item_ids:
            if item in self.item_to_idx:
                train_mask[self.item_to_idx[item]] = True
        self.item_vectors = self.model.all_item_vectors(
            device, genre_ids=genre, tag_ids=tag, numeric=numeric, train_item_mask=train_mask
        )

    def item_score(self, user_vec: torch.Tensor, item: str) -> float:
        idx = self.item_to_idx.get(item)
        if idx is None:
            return 0.0
        return float((user_vec @ self.item_vectors[idx]).item())

    def retrieve_rows(
        self,
        users: list[str],
        seen_per_row: list[set[str]],
        k: int,
        batch_size: int = 512,
    ) -> list[list[tuple[str, float]]]:
        """One retrieved list per row. Duplicate users are allowed (different seen masks)."""
        if len(users) != len(seen_per_row):
            raise ValueError("users and seen_per_row must align")
        out: list[list[tuple[str, float]]] = [[] for _ in users]
        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            uvec = encode_users_mixed(self.model, batch, self.user_to_idx, self.device)
            seen_idx: list[set[int]] = []
            for banned_items in seen_per_row[start : start + batch_size]:
                banned: set[int] = set()
                for item in banned_items:
                    if item in self.item_to_idx:
                        banned.add(self.item_to_idx[item])
                seen_idx.append(banned)
            recs = retrieve_scored(uvec, self.item_vectors, self.idx_to_item, seen_idx, k)
            out[start : start + len(recs)] = recs
        return out

    def retrieve(
        self,
        users: list[str],
        seen: dict[str, set[str]],
        k: int,
        batch_size: int = 512,
    ) -> dict[str, list[tuple[str, float]]]:
        rows = self.retrieve_rows(users, [seen.get(u, set()) for u in users], k, batch_size)
        return {u: ranked for u, ranked in zip(users, rows)}

    def user_vectors(self, users: list[str]) -> torch.Tensor:
        return encode_users_mixed(self.model, users, self.user_to_idx, self.device)
