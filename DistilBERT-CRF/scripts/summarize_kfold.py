"""Summarize k-fold results (mean/std) from a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=str,
        default="training_logs/kfold_results.csv",
        help="Path to k-fold results CSV (run_name, fold_index, best_f1, ...).",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Only include rows whose run_name starts with this prefix.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="best_f1",
        help="Metric column to aggregate (default: best_f1).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the summary as JSON.",
    )
    return parser.parse_args()


def load_rows(path: Path, run_prefix: Optional[str]) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            name = row.get("run_name", "")
            if run_prefix and not name.startswith(run_prefix):
                continue
            rows.append(row)
    return rows


def aggregate(rows: List[dict], metric: str) -> dict:
    values: List[float] = []
    for row in rows:
        value = row.get(metric)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue

    if not values:
        raise ValueError("No metric values found to aggregate.")

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path, args.run_prefix)
    summary = aggregate(rows, args.metric)
    summary["metric"] = args.metric
    summary["run_prefix"] = args.run_prefix
    summary["csv"] = str(csv_path)

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
