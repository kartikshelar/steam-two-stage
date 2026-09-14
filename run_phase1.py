from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_retrieval import main as phase1_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 retrieval (content two-tower + ANN).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1.yaml")
    parser.add_argument("--split", choices=["temporal", "random", "both"], default="both")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)
    argv2 = ["--config", str(args.config), "--split", args.split]
    if args.skip_train:
        argv2.append("--skip-train")
    return phase1_main(argv2)


if __name__ == "__main__":
    raise SystemExit(main())
