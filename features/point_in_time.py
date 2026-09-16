from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from retrieval.content import _parse_list

DAY = 86400
WINDOWS = (7, 30, 90)
NO_HISTORY_DAYS = 3650.0


def _genre_bag() -> dict[str, int]:
    return defaultdict(int)

FEATURE_COLS = [
    "user_n",
    "user_hours_sum",
    "user_days_since_last",
    "user_n_7d",
    "user_n_30d",
    "user_n_90d",
    "item_n",
    "item_n_7d",
    "item_n_30d",
    "item_n_90d",
    "ui_n",
    "genre_affinity",
    "item_price",
    "item_log_price",
    "item_days_since_release",
    "item_is_free",
]


def _window_count(times: list[int], t: int, days: int) -> int:
    if not times:
        return 0
    return len(times) - bisect.bisect_left(times, t - days * DAY)


def _item_meta(games: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if games is None or games.empty:
        return out
    for row in games.itertuples(index=False):
        item_id = str(getattr(row, "item_id"))
        price = getattr(row, "price", None)
        try:
            price_f = float(price) if price is not None and not pd.isna(price) else float("nan")
        except (TypeError, ValueError):
            price_f = float("nan")
        rel = getattr(row, "release_date", None)
        rel_ts = pd.to_datetime(rel, utc=True, errors="coerce")
        rel_unix = int(rel_ts.timestamp()) if not pd.isna(rel_ts) else None
        out[item_id] = {
            "genres": _parse_list(getattr(row, "genres", None)),
            "price": price_f,
            "release_ts": rel_unix,
        }
    return out


def _static_item_feats(meta: dict | None, t: int) -> tuple[float, float, float, float]:
    if not meta:
        return 0.0, 0.0, NO_HISTORY_DAYS, 0.0
    price = meta["price"]
    if price != price:  # NaN
        price_v, logp, is_free = 0.0, 0.0, 0.0
    else:
        price_v = float(price)
        logp = float(np.log1p(max(price_v, 0.0)))
        is_free = 1.0 if price_v <= 0 else 0.0
    rel = meta["release_ts"]
    if rel is None:
        age = NO_HISTORY_DAYS
    else:
        age = max(0.0, (t - rel) / DAY)
    return price_v, logp, age, is_free


class PITState:
    """Running stats using only events already observed (strictly before the next emit)."""

    def __init__(self):
        self.user_n: dict[str, int] = defaultdict(int)
        self.item_n: dict[str, int] = defaultdict(int)
        self.ui_n: dict[tuple[str, str], int] = defaultdict(int)
        self.user_hours: dict[str, float] = defaultdict(float)
        self.user_last: dict[str, int] = {}
        self.user_ts: dict[str, list[int]] = defaultdict(list)
        self.item_ts: dict[str, list[int]] = defaultdict(list)
        self.user_genres: dict[str, dict[str, int]] = defaultdict(_genre_bag)

    def features(self, user: str, item: str, t: int, meta: dict | None) -> dict[str, float]:
        ut = self.user_ts.get(user, [])
        it = self.item_ts.get(item, [])
        last = self.user_last.get(user)
        recency = NO_HISTORY_DAYS if last is None else max(0.0, (t - last) / DAY)
        genres = meta["genres"] if meta else []
        ug = self.user_genres.get(user)
        if genres and ug:
            affinity = float(sum(ug.get(g, 0) for g in genres)) / (self.user_n.get(user, 0) + 1e-6)
        else:
            affinity = 0.0
        price_v, logp, age, is_free = _static_item_feats(meta, t)
        return {
            "user_n": float(self.user_n.get(user, 0)),
            "user_hours_sum": float(self.user_hours.get(user, 0.0)),
            "user_days_since_last": recency,
            "user_n_7d": float(_window_count(ut, t, 7)),
            "user_n_30d": float(_window_count(ut, t, 30)),
            "user_n_90d": float(_window_count(ut, t, 90)),
            "item_n": float(self.item_n.get(item, 0)),
            "item_n_7d": float(_window_count(it, t, 7)),
            "item_n_30d": float(_window_count(it, t, 30)),
            "item_n_90d": float(_window_count(it, t, 90)),
            "ui_n": float(self.ui_n.get((user, item), 0)),
            "genre_affinity": affinity,
            "item_price": price_v,
            "item_log_price": logp,
            "item_days_since_release": age,
            "item_is_free": is_free,
        }

    def observe(self, user: str, item: str, t: int, hours: float, genres: list[str]) -> None:
        self.user_n[user] += 1
        self.item_n[item] += 1
        self.ui_n[(user, item)] += 1
        if hours == hours:  # not NaN
            self.user_hours[user] += float(hours)
        self.user_last[user] = t
        self.user_ts[user].append(t)
        self.item_ts[item].append(t)
        if genres:
            bag = self.user_genres[user]
            for g in genres:
                bag[g] += 1


def naive_leaked_features(
    user: str,
    item: str,
    t: int,
    *,
    user_n_all: dict[str, int],
    item_n_all: dict[str, int],
    ui_n_all: dict[tuple[str, str], int],
    user_hours_all: dict[str, float],
    user_ts_all: dict[str, list[int]],
    item_ts_all: dict[str, list[int]],
    user_genres_all: dict[str, dict[str, int]],
    user_last_all: dict[str, int],
    meta: dict | None,
) -> dict[str, float]:
    """Wrong on purpose: counts use the whole timeline, including now and the future.

    `item_n` is the classic leak — popularity computed after the test period.
    Trailing windows are centered on t and include future events.
    """
    ut = user_ts_all.get(user, [])
    it = item_ts_all.get(item, [])
    last = user_last_all.get(user)
    recency = NO_HISTORY_DAYS if last is None else max(0.0, (last - t) / DAY)
    # last can be in the future; days-since-last then goes to 0 via max(0, negative)
    if last is not None and last >= t:
        recency = 0.0
    genres = meta["genres"] if meta else []
    ug = user_genres_all.get(user)
    un = float(user_n_all.get(user, 0))
    if genres and ug and un:
        affinity = float(sum(ug.get(g, 0) for g in genres)) / (un + 1e-6)
    else:
        affinity = 0.0
    price_v, logp, age, is_free = _static_item_feats(meta, t)

    def two_sided(times: list[int], days: int) -> float:
        if not times:
            return 0.0
        lo = t - days * DAY
        hi = t + days * DAY
        return float(bisect.bisect_right(times, hi) - bisect.bisect_left(times, lo))

    return {
        "user_n": un,
        "user_hours_sum": float(user_hours_all.get(user, 0.0)),
        "user_days_since_last": recency,
        "user_n_7d": two_sided(ut, 7),
        "user_n_30d": two_sided(ut, 30),
        "user_n_90d": two_sided(ut, 90),
        "item_n": float(item_n_all.get(item, 0)),
        "item_n_7d": two_sided(it, 7),
        "item_n_30d": two_sided(it, 30),
        "item_n_90d": two_sided(it, 90),
        "ui_n": float(ui_n_all.get((user, item), 0)),
        "genre_affinity": affinity,
        "item_price": price_v,
        "item_log_price": logp,
        "item_days_since_release": age,
        "item_is_free": is_free,
    }


def build_global_leak_stats(events: pd.DataFrame, meta_by_item: dict[str, dict]) -> dict:
    user_n: dict[str, int] = defaultdict(int)
    item_n: dict[str, int] = defaultdict(int)
    ui_n: dict[tuple[str, str], int] = defaultdict(int)
    user_hours: dict[str, float] = defaultdict(float)
    user_ts: dict[str, list[int]] = defaultdict(list)
    item_ts: dict[str, list[int]] = defaultdict(list)
    user_genres: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    user_last: dict[str, int] = {}
    for rec in events.itertuples(index=False):
        user = str(rec.user_id)
        item = str(rec.item_id)
        t = int(rec.ts)
        hours = float(rec.hours) if rec.hours == rec.hours else float("nan")
        user_n[user] += 1
        item_n[item] += 1
        ui_n[(user, item)] += 1
        if hours == hours:
            user_hours[user] += hours
        user_ts[user].append(t)
        item_ts[item].append(t)
        user_last[user] = t
        genres = (meta_by_item.get(item) or {}).get("genres") or []
        if genres:
            bag = user_genres[user]
            for g in genres:
                bag[g] += 1
    for lst in user_ts.values():
        lst.sort()
    for lst in item_ts.values():
        lst.sort()
    return {
        "user_n_all": dict(user_n),
        "item_n_all": dict(item_n),
        "ui_n_all": dict(ui_n),
        "user_hours_all": dict(user_hours),
        "user_ts_all": dict(user_ts),
        "item_ts_all": dict(item_ts),
        "user_genres_all": {k: dict(v) for k, v in user_genres.items()},
        "user_last_all": user_last,
    }


def assert_no_future_in_pit(feature_row: dict[str, float], history: pd.DataFrame, user: str, item: str, t: int) -> None:
    """Fail if PIT item_n/user_n disagree with a strict count of rows with ts < t."""
    past = history.loc[history["ts"] < t]
    expect_item = int((past["item_id"].astype(str) == str(item)).sum())
    expect_user = int((past["user_id"].astype(str) == str(user)).sum())
    if int(feature_row["item_n"]) != expect_item:
        raise AssertionError(f"PIT item_n leaked or drifted: got {feature_row['item_n']} expected {expect_item} at t={t}")
    if int(feature_row["user_n"]) != expect_user:
        raise AssertionError(f"PIT user_n leaked or drifted: got {feature_row['user_n']} expected {expect_user} at t={t}")


def features_at_events(
    events: pd.DataFrame,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One PIT feature row per event, using only earlier events."""
    meta_by_item = _item_meta(games if games is not None else pd.DataFrame())
    ordered = events.sort_values(["ts", "user_id", "item_id"]).reset_index(drop=True)
    state = PITState()
    rows = []
    for rec in ordered.itertuples(index=False):
        user = str(rec.user_id)
        item = str(rec.item_id)
        t = int(rec.ts)
        hours = float(rec.hours) if rec.hours == rec.hours else float("nan")
        meta = meta_by_item.get(item)
        feats = state.features(user, item, t, meta)
        feats["user_id"] = user
        feats["item_id"] = item
        feats["ts"] = t
        rows.append(feats)
        state.observe(user, item, t, hours, (meta or {}).get("genres") or [])
    return pd.DataFrame(rows)


def iter_sorted_events(events: pd.DataFrame) -> Iterable:
    return events.sort_values(["ts", "user_id", "item_id"]).itertuples(index=False)


def freeze_before(
    events: pd.DataFrame,
    games: pd.DataFrame | None,
    cutoff_unix: int,
) -> PITState:
    """Observe every event with ts < cutoff. Used by ranking eval and serving."""
    meta_by_item = _item_meta(games if games is not None else pd.DataFrame())
    state = PITState()
    past = events.loc[events["ts"] < cutoff_unix]
    ordered = past.sort_values(["ts", "user_id", "item_id"])
    try:
        from tqdm import tqdm

        iterator = tqdm(ordered.itertuples(index=False), total=len(ordered), desc="PIT freeze")
    except ImportError:
        iterator = ordered.itertuples(index=False)
    for rec in iterator:
        t = int(rec.ts)
        item = str(rec.item_id)
        hours = float(rec.hours) if rec.hours == rec.hours else float("nan")
        meta_i = meta_by_item.get(item)
        state.observe(str(rec.user_id), item, t, hours, (meta_i or {}).get("genres") or [])
    return state
