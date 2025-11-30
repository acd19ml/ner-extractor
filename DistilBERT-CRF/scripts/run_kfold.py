"""Launch GroupKFold experiments for DistilBERT-CRF and log fold metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train_distilbert_crf.py"
DEFAULT_RESULTS_FILE = ROOT_DIR / "training_logs" / "kfold_results.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Base YAML configuration to reuse for each fold.",
    )
    parser.add_argument(
        "--splits-file",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "conll03" / "kfold_splits.json"),
        help="JSON file describing GroupKFold splits (list of fold objects with train/dev sentence indices).",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="distilbert_crf_fold",
        help="Prefix used for run names (suffix '_fold{i}' gets appended).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit the number of folds to execute (useful for smoke tests).",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(DEFAULT_RESULTS_FILE),
        help="CSV file where fold summaries will be appended.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to train_distilbert_crf.py.",
    )
    return parser


def run_fold(cmd: List[str], env: dict) -> None:
    process = subprocess.Popen(cmd, env=env)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Fold command failed with exit code {process.returncode}: {' '.join(cmd)}")


def load_splits(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    folds = split_data.get("folds") if isinstance(split_data, dict) else split_data
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Invalid splits file format: {path}")
    return folds


def load_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_result(summary: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "fold_index",
        "best_f1",
        "best_model_dir",
        "evaluation_split",
        "evaluation_f1",
        "test_f1",
    ]
    row = {
        "run_name": summary.get("run_name"),
        "fold_index": summary.get("fold_index"),
        "best_f1": summary.get("best_f1"),
        "best_model_dir": summary.get("best_model_dir"),
    }
    if summary.get("evaluation_split") and summary.get("evaluation_metrics"):
        row["evaluation_split"] = summary["evaluation_split"]
        row["evaluation_f1"] = summary["evaluation_metrics"].get("f1")
    if summary.get("test_metrics"):
        row["test_f1"] = summary["test_metrics"].get("f1")

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    splits_path = Path(args.splits_file).expanduser().resolve()
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    folds = load_splits(splits_path)

    max_folds = args.max_folds if args.max_folds is None else max(0, args.max_folds)
    fold_iter = list(enumerate(folds, start=1))
    if max_folds:
        fold_iter = fold_iter[:max_folds]

    results_path = Path(args.results_file).expanduser().resolve() if args.results_file else None

    for fold_idx, fold in fold_iter:
        run_name = f"{args.run_prefix}_fold{fold_idx}"
        fold_env = os.environ.copy()
        fold_env["NER_FOLD_INDEX"] = str(fold_idx)
        fold_env["NER_FOLD_TRAIN_INDICES"] = json.dumps(fold.get("train_indices"))
        fold_env["NER_FOLD_VALIDATION_INDICES"] = json.dumps(fold.get("validation_indices"))

        metrics_path = ROOT_DIR / "training_logs" / f"{run_name}_metrics.json"
        if metrics_path.exists():
            metrics_path.unlink()

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(args.config),
            "--run-name",
            run_name,
            "--metrics-output",
            str(metrics_path),
        ]
        if args.extra_args:
            cmd.extend(args.extra_args)

        print(f"Launching fold {fold_idx} with run name {run_name}")
        run_fold(cmd, fold_env)

        summary = load_summary(metrics_path)
        if not summary:
            eval_path = metrics_path.with_name(f"{run_name}_eval.json")
            summary = load_summary(eval_path)
        if not summary:
            print(f"[WARN] Metrics summary not found for {run_name} at {metrics_path}")
        elif results_path:
            append_result(summary, results_path)
            print(f"Appended fold {fold_idx} results to {results_path}")


if __name__ == "__main__":
    main()
