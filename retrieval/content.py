from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

PAD = 0


def _parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = [part.strip() for part in text.split(",")]
    out = []
    for item in raw:
        name = html.unescape(str(item)).strip()
        if name:
            out.append(name)
    return out


@dataclass
class ContentVocab:
    genre_to_idx: dict[str, int]
    tag_to_idx: dict[str, int]
    max_genres: int
    max_tags: int
    price_scale: float
    year_offset: float
    year_scale: float
    n_genres: int
    n_tags: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ContentVocab":
        return cls(
            genre_to_idx={str(k): int(v) for k, v in payload["genre_to_idx"].items()},
            tag_to_idx={str(k): int(v) for k, v in payload["tag_to_idx"].items()},
            max_genres=int(payload["max_genres"]),
            max_tags=int(payload["max_tags"]),
            price_scale=float(payload["price_scale"]),
            year_offset=float(payload["year_offset"]),
            year_scale=float(payload["year_scale"]),
            n_genres=int(payload["n_genres"]),
            n_tags=int(payload["n_tags"]),
        )


def fit_content_vocab(
    games: pd.DataFrame,
    train_item_ids: set[str],
    max_tags: int = 80,
    max_genres_per_item: int = 8,
    max_tags_per_item: int = 16,
) -> ContentVocab:
    """Vocab and numeric scales from TRAIN items only. Static metadata, not interaction counts."""
    train_games = games[games["item_id"].astype(str).isin(train_item_ids)]
    if train_games.empty:
        train_games = games

    genre_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for _, row in train_games.iterrows():
        for g in _parse_list(row.get("genres")):
            genre_counts[g] = genre_counts.get(g, 0) + 1
        for t in _parse_list(row.get("tags")):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    genre_to_idx = {name: i + 1 for i, name in enumerate(sorted(genre_counts))}
    top_tags = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_tags]
    tag_to_idx = {name: i + 1 for i, (name, _) in enumerate(top_tags)}

    prices = pd.to_numeric(train_games["price"], errors="coerce")
    logp = np.log1p(prices.clip(lower=0).dropna().to_numpy())
    price_scale = float(np.max(logp)) if len(logp) else 1.0
    if price_scale <= 0:
        price_scale = 1.0

    years = pd.to_datetime(train_games["release_date"], errors="coerce", utc=True).dt.year.dropna()
    if len(years):
        year_offset = float(years.min())
        span = float(years.max() - years.min())
        year_scale = span if span > 0 else 1.0
    else:
        year_offset, year_scale = 2000.0, 20.0

    return ContentVocab(
        genre_to_idx=genre_to_idx,
        tag_to_idx=tag_to_idx,
        max_genres=max_genres_per_item,
        max_tags=max_tags_per_item,
        price_scale=price_scale,
        year_offset=year_offset,
        year_scale=year_scale,
        n_genres=len(genre_to_idx) + 1,
        n_tags=len(tag_to_idx) + 1,
    )


def _numeric_row(row: pd.Series | None, vocab: ContentVocab) -> np.ndarray:
    if row is None:
        return np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
    if pd.isna(price) or float(price) < 0:
        log_price, price_missing = 0.0, 1.0
    else:
        log_price = float(np.log1p(float(price))) / vocab.price_scale
        price_missing = 0.0
    ts = pd.to_datetime(row.get("release_date"), errors="coerce", utc=True)
    if pd.isna(ts):
        year_scaled, year_missing = 0.0, 1.0
    else:
        year_scaled = (float(ts.year) - vocab.year_offset) / vocab.year_scale
        year_missing = 0.0
    return np.array([log_price, price_missing, year_scaled, year_missing], dtype=np.float32)


def build_content_tables(
    item_to_idx: dict[str, int],
    games: pd.DataFrame,
    vocab: ContentVocab,
) -> dict[str, np.ndarray]:
    """Tables indexed by item idx, row 0 = pad/unk zeros."""
    n_items = max(item_to_idx.values()) + 1
    genre_ids = np.zeros((n_items, vocab.max_genres), dtype=np.int64)
    tag_ids = np.zeros((n_items, vocab.max_tags), dtype=np.int64)
    numeric = np.zeros((n_items, 4), dtype=np.float32)
    numeric[:, 1] = 1.0
    numeric[:, 3] = 1.0

    games_by_id = {str(i): row for i, row in games.set_index("item_id").iterrows()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    for idx, item in idx_to_item.items():
        row = games_by_id.get(str(item))
        numeric[idx] = _numeric_row(row, vocab)
        if row is None:
            continue
        genres = [vocab.genre_to_idx[g] for g in _parse_list(row.get("genres")) if g in vocab.genre_to_idx]
        tags = [vocab.tag_to_idx[t] for t in _parse_list(row.get("tags")) if t in vocab.tag_to_idx]
        genre_ids[idx, : min(len(genres), vocab.max_genres)] = genres[: vocab.max_genres]
        tag_ids[idx, : min(len(tags), vocab.max_tags)] = tags[: vocab.max_tags]
    return {"genre_ids": genre_ids, "tag_ids": tag_ids, "numeric": numeric}
