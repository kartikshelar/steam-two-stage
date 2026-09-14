from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"


FEATURES = ROOT / "features"
FEATURE_DATA = DATA / "features"


def ensure_dirs() -> None:
    for path in (RAW, PROCESSED, SPLITS, RESULTS, MODELS, DOCS, FEATURE_DATA):
        path.mkdir(parents=True, exist_ok=True)
