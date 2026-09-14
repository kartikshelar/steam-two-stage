from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.leak_gap import main as leak_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2: point-in-time features vs leak.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2.yaml")
    args = parser.parse_args(argv)
    return leak_main(["--config", str(args.config)])


if __name__ == "__main__":
    raise SystemExit(main())
