from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import load_config
from data.labels import main as labels_main
from data.prepare import main as prepare_main
from data.splits import main as splits_main
from experiments.label_experiment import main as experiment_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0: data, splits, labels, label experiment.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase0.yaml")
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    args = parser.parse_args(argv)

    if not args.skip_prepare:
        prepare_argv = ["--config", str(args.config)]
        if args.max_reviews is not None:
            prepare_argv += ["--max-reviews", str(args.max_reviews)]
        prepare_main(prepare_argv)

    split_argv = ["--config", str(args.config)]
    if args.force_splits:
        split_argv.append("--force")
    splits_main(split_argv)
    labels_main(["--config", str(args.config)])
    exp_argv = ["--config", str(args.config), "--split", args.split]
    if args.skip_train:
        exp_argv.append("--skip-train")
    return experiment_main(exp_argv)


if __name__ == "__main__":
    raise SystemExit(main())
