from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.io import utc_now, write_json
from data.paths import RESULTS, SPLITS

LABEL_PURCHASE = "label_purchase"
LABEL_THRESHOLD = "label_threshold"
LABEL_PERCENTILE = "label_percentile"
LABEL_COLS = (LABEL_PURCHASE, LABEL_THRESHOLD, LABEL_PERCENTILE)


def binary_purchase(df: pd.DataFrame) -> pd.Series:
    """Ignore play time. Every review is a positive."""
    return pd.Series(True, index=df.index, dtype=bool)


def thresholded_hours(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Liked if hours > threshold. Missing hours are not positives."""
    hours = pd.to_numeric(df["hours"], errors="coerce")
    return hours > threshold


def game_percentile_cutoffs(train: pd.DataFrame, percentile: float) -> pd.Series:
    """Per-item hour cutoffs from TRAIN only. `percentile` is 0-100."""
    hours = pd.to_numeric(train["hours"], errors="coerce")
    tmp = train.loc[hours.notna(), ["item_id"]].copy()
    tmp["hours"] = hours[hours.notna()]
    if tmp.empty:
        return pd.Series(dtype=float)
    q = percentile / 100.0
    return tmp.groupby("item_id")["hours"].quantile(q)


def within_game_percentile(
    df: pd.DataFrame,
    cutoffs: pd.Series,
    default_negative: bool = True,
) -> pd.Series:
    """Liked if hours >= the item's train-only percentile cutoff.

    Items with no train cutoff (unseen in train, or no hours) are negatives when
    `default_negative` is True — they are not silently scored from test hours.
    """
    hours = pd.to_numeric(df["hours"], errors="coerce")
    mapped = df["item_id"].map(cutoffs)
    out = hours >= mapped
    if default_negative:
        out = out.fillna(False)
    return out.astype(bool)


def attach_labels(
    train: pd.DataFrame,
    test: pd.DataFrame,
    hours_threshold: float,
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cutoffs = game_percentile_cutoffs(train, percentile)
    train = train.copy()
    test = test.copy()
    for frame in (train, test):
        frame[LABEL_PURCHASE] = binary_purchase(frame)
        frame[LABEL_THRESHOLD] = thresholded_hours(frame, hours_threshold)
        frame[LABEL_PERCENTILE] = within_game_percentile(frame, cutoffs)
    return train, test, cutoffs


def _rate(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    return {
        "n": int(len(df)),
        "n_positive": int(s.sum()),
        "positive_rate": float(s.mean()) if len(df) else None,
        "n_users_positive": int(df.loc[s, "user_id"].nunique()) if len(df) else 0,
        "n_items_positive": int(df.loc[s, "item_id"].nunique()) if len(df) else 0,
    }


def label_splits(config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config["labels"]["hours_threshold"])
    percentile = float(config["labels"]["game_percentile"])
    rows = []
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "hours_threshold": threshold,
        "game_percentile": percentile,
        "cutoff_source": "train split only",
    }

    for split_name in ("temporal", "random"):
        train_path = SPLITS / f"train_{split_name}.parquet"
        test_path = SPLITS / f"test_{split_name}.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing {train_path}. Run python -m data.splits first.")
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        train_l, test_l, cutoffs = attach_labels(train, test, threshold, percentile)
        train_l.to_parquet(SPLITS / f"train_{split_name}_labeled.parquet", index=False)
        test_l.to_parquet(SPLITS / f"test_{split_name}_labeled.parquet", index=False)
        cutoffs.rename("hours_cutoff").reset_index().to_parquet(
            SPLITS / f"game_hour_cutoffs_{split_name}.parquet", index=False
        )
        for split_part, frame in (("train", train_l), ("test", test_l)):
            for col, label_name in (
                (LABEL_PURCHASE, "purchase"),
                (LABEL_THRESHOLD, "threshold"),
                (LABEL_PERCENTILE, "percentile"),
            ):
                rec = {
                    "split": split_name,
                    "part": split_part,
                    "label": label_name,
                    **_rate(frame, col),
                }
                rows.append(rec)
        summary[split_name] = {
            "n_items_with_cutoff": int(len(cutoffs)),
            "cutoff_p50": float(cutoffs.median()) if len(cutoffs) else None,
            "cutoff_p90": float(cutoffs.quantile(0.9)) if len(cutoffs) else None,
        }

    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "label_stats.csv", index=False)
    write_json(RESULTS / "label_stats.json", {"summary": summary, "rows": rows})
    return {"summary": summary, "n_stat_rows": len(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach the three play-time label formulations.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    out = label_splits(load_config(args.config))
    print(json.dumps(out["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
