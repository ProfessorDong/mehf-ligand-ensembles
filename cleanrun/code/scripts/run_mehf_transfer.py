"""Phase C training entry point for MEHF parent->product ensemble transfer.

    export CUDA_VISIBLE_DEVICES=0
    PYTHONPATH=src python scripts/run_mehf_transfer.py \
        --config configs/mehf_transfer_smoke_v1.yaml

This phase may read only the PRIMARY split's `train`; the trainer refuses any
other split.  GPU 0 only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdwm.config import load_config  # noqa: E402
from mdwm.training.mehf_transfer import run_train_mehf_transfer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "0", "1"):
        # Both cards are authorised for this campaign; one variant per card,
        # never two trainings on the same card (a training peaked at 19.65 GB).
        raise SystemExit(
            f"Use GPU 0 or GPU 1, one job per card; CUDA_VISIBLE_DEVICES={visible!r}"
        )

    cfg = load_config(args.config)
    if args.output_root is not None:
        cfg.setdefault("outputs", {})["root"] = str(args.output_root)
    if args.max_pairs is not None:
        cfg.setdefault("data", {})["max_pairs"] = int(args.max_pairs)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = int(args.max_steps)
    summary = run_train_mehf_transfer(cfg)
    summary.pop("history", None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
