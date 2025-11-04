"""CLI utility to prepare CoNLL03 dataset splits for the DistilBERT-CRF project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from datasets import prepare_conll03_corpus  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for dataset preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Optional override for raw data directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Optional override for processed data directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Override validation split ratio defined in configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed defined in configuration.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the dataset preparation CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    raw_dir = Path(args.raw_dir) if args.raw_dir else config["paths"]["raw_data_dir"]
    processed_dir = (
        Path(args.processed_dir) if args.processed_dir else config["paths"]["processed_data_dir"]
    )
    val_ratio = args.val_ratio if args.val_ratio is not None else config["dataset"]["validation_ratio"]
    seed = args.seed if args.seed is not None else config["dataset"]["seed"]

    output_paths = prepare_conll03_corpus(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        val_ratio=val_ratio,
        seed=seed,
    )

    print("Processed splits written to:")
    for split_name, path in output_paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()

