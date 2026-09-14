from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    """Two-tower retriever.

    Phase 0: ID embeddings only (`use_content=False`).
    Phase 1: item tower is ID + genre/tag averages + price/year. Unseen items use
    the pad ID so the content path is the cold-start mechanism. ID dropout during
    training stops the model ignoring content.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        pad_idx: int = 0,
        use_content: bool = False,
        n_genres: int = 0,
        n_tags: int = 0,
        n_numeric: int = 4,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.pad_idx = pad_idx
        self.use_content = use_content
        self.n_genres = n_genres
        self.n_tags = n_tags
        self.user_emb = nn.Embedding(n_users, dim, padding_idx=pad_idx)
        self.item_emb = nn.Embedding(n_items, dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.user_emb.weight[pad_idx].zero_()
            self.item_emb.weight[pad_idx].zero_()

        if use_content:
            self.genre_emb = nn.Embedding(n_genres, dim, padding_idx=pad_idx)
            self.tag_emb = nn.Embedding(n_tags, dim, padding_idx=pad_idx)
            self.numeric_proj = nn.Linear(n_numeric, dim)
            self.item_fuse = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
            nn.init.xavier_uniform_(self.genre_emb.weight)
            nn.init.xavier_uniform_(self.tag_emb.weight)
            with torch.no_grad():
                self.genre_emb.weight[pad_idx].zero_()
                self.tag_emb.weight[pad_idx].zero_()

    def encode_users(self, user_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_emb(user_idx), dim=-1)

    def _masked_mean(self, emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != self.pad_idx).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def encode_items(
        self,
        item_idx: torch.Tensor,
        genre_ids: torch.Tensor | None = None,
        tag_ids: torch.Tensor | None = None,
        numeric: torch.Tensor | None = None,
        id_dropout: float = 0.0,
        force_unk_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = item_idx
        if self.training and id_dropout > 0:
            drop = torch.rand(item_idx.shape, device=item_idx.device) < id_dropout
            idx = torch.where(drop, torch.full_like(item_idx, self.pad_idx), item_idx)
        if force_unk_id is not None:
            idx = torch.where(force_unk_id, torch.full_like(item_idx, self.pad_idx), idx)
        h = self.item_emb(idx)
        if self.use_content:
            if genre_ids is None or tag_ids is None or numeric is None:
                raise ValueError("content tensors are required when use_content=True")
            g = self._masked_mean(self.genre_emb(genre_ids), genre_ids)
            t = self._masked_mean(self.tag_emb(tag_ids), tag_ids)
            n = self.numeric_proj(numeric)
            h = self.item_fuse(h + g + t + n)
        return F.normalize(h, dim=-1)

    def inbatch_loss(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        log_q: torch.Tensor | None,
        temperature: float,
        genre_ids: torch.Tensor | None = None,
        tag_ids: torch.Tensor | None = None,
        numeric: torch.Tensor | None = None,
        id_dropout: float = 0.0,
    ) -> torch.Tensor:
        users = self.encode_users(user_idx)
        items = self.encode_items(
            item_idx,
            genre_ids=genre_ids,
            tag_ids=tag_ids,
            numeric=numeric,
            id_dropout=id_dropout,
        )
        logits = users @ items.t() / temperature
        if log_q is not None:
            logits = logits - log_q[item_idx].unsqueeze(0)
        targets = torch.arange(user_idx.size(0), device=user_idx.device)
        return F.cross_entropy(logits, targets)

    def all_item_vectors(
        self,
        device: torch.device,
        genre_ids: torch.Tensor | None = None,
        tag_ids: torch.Tensor | None = None,
        numeric: torch.Tensor | None = None,
        train_item_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = torch.arange(self.n_items, device=device)
        force = None
        if train_item_mask is not None:
            force = ~train_item_mask
            force[self.pad_idx] = False
        return self.encode_items(
            idx,
            genre_ids=None if genre_ids is None else genre_ids.to(device),
            tag_ids=None if tag_ids is None else tag_ids.to(device),
            numeric=None if numeric is None else numeric.to(device),
            force_unk_id=force,
        )
