from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.paths import RESULTS


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def append_metrics(rows: list[dict], filename: str = "metrics.csv") -> Path:
    """Append metric rows to results/. Every reported number must come from here."""
    out = RESULTS / filename
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if out.exists():
        existing = pd.read_csv(out)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(out, index=False)
    return out
