from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "phase0.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
