from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_ranking import main as phase3_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3: GBDT on retrieved candidates.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase3.yaml")
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    args = parser.parse_args(argv)
    argv2 = ["--config", str(args.config)]
    if args.max_eval_users is not None:
        argv2.extend(["--max-eval-users", str(args.max_eval_users)])
    if args.max_train_queries is not None:
        argv2.extend(["--max-train-queries", str(args.max_train_queries)])
    return phase3_main(argv2)


if __name__ == "__main__":
    raise SystemExit(main())
