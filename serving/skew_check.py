"""Assert serving feature rows match PITState.features on sampled pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.io import utc_now, write_json
from data.paths import PROCESSED, RESULTS
from ranking.features import RANKING_COLS
from serving.service import load_service


def _allclose_row(a: dict, b: pd.Series, cols: list[str], rtol: float = 1e-6, atol: float = 1e-6) -> list[str]:
    bad = []
    for col in cols:
        x, y = float(a[col]), float(b[col])
        if not np.isfinite(x) and not np.isfinite(y):
            continue
        if not np.isclose(x, y, rtol=rtol, atol=atol):
            bad.append(f"{col}: train={x} serve={y}")
    return bad


def run_skew(n_pairs: int = 200, seed: int = 42) -> dict:
    """Compare RecsService.feature_frame to PITState.features (training path)."""
    svc = load_service()
    games = pd.read_parquet(PROCESSED / "games.parquet")
    catalog = games["item_id"].astype(str).tolist()
    users = list(svc.retriever.user_to_idx.keys())
    rng = np.random.RandomState(seed)
    n_checked = 0
    mismatches: list[str] = []
    for _ in range(n_pairs):
        user = str(users[int(rng.randint(0, len(users)))])
        item = str(catalog[int(rng.randint(0, len(catalog)))])
        train_feats = svc.pit.features(user, item, svc.cutoff_unix, svc.meta_by_item.get(item))
        train_feats["retrieval_score"] = 0.0
        serve_frame = svc.feature_frame(user, [(item, 0.0)])
        bad = _allclose_row(train_feats, serve_frame.iloc[0], RANKING_COLS)
        if bad:
            mismatches.append(f"{user}/{item}: " + "; ".join(bad))
        n_checked += 1
    return {
        "n_checked": n_checked,
        "n_mismatch": len(mismatches),
        "passed": len(mismatches) == 0,
        "feature_cols": list(RANKING_COLS),
        "examples": mismatches[:10],
        "note": (
            "Serving wraps PITState.features at the temporal cutoff — the same "
            "function the ranker was trained on. retrieval_score is compared at 0 "
            "here; recommend() fills it from the frozen two-tower."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training vs serving feature skew check.")
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    payload = run_skew(n_pairs=args.n_pairs, seed=args.seed)
    payload["run_id"] = utc_now().replace(":", "").replace("-", "")
    payload["phase"] = 5
    write_json(RESULTS / "skew_check.json", payload)
    print(json.dumps({k: payload[k] for k in ("n_checked", "n_mismatch", "passed")}, indent=2))
    if not payload["passed"]:
        print(payload["examples"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
