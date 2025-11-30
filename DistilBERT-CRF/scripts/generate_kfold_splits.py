"""Generate doc-aware GroupKFold splits for DistilBERT-CRF datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from datasets import ConllSentence, read_conll_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Config file containing raw/processed data paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "conll03" / "kfold_splits.json"),
        help="Destination JSON file for fold definitions.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for GroupKFold.",
    )
    return parser.parse_args()


def locate_raw_files(raw_dir: Path) -> Tuple[Path, Path]:
    candidates = {
        "train": ["train.txt", "eng.train"],
        "validation": ["validation.txt", "valid.txt", "dev.txt", "eng.testa"],
    }

    def resolve(kind: str) -> Path:
        for name in candidates[kind]:
            candidate = raw_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not locate {kind} split under {raw_dir}")

    train_file = resolve("train")
    dev_file = resolve("validation")
    return train_file, dev_file


def read_raw_with_doc_ids(paths: Sequence[Path]) -> List[Tuple[ConllSentence, int]]:
    sentences: List[Tuple[ConllSentence, int]] = []
    doc_id = -1
    current: List[str] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("-DOCSTART-"):
                    if current:
                        sentences.append((ConllSentence.from_lines(current), doc_id))
                        current.clear()
                    doc_id += 1
                    continue
                if not stripped:
                    if current:
                        sentences.append((ConllSentence.from_lines(current), doc_id))
                        current.clear()
                    continue
                current.append(stripped)
        if current:
            sentences.append((ConllSentence.from_lines(current), doc_id))
            current.clear()

    return sentences


def build_sentence_key(sentence: ConllSentence) -> str:
    return "\n".join(sentence.to_lines())


def map_processed_doc_ids(
    processed_sentences: Sequence[ConllSentence],
    raw_mapping: Dict[str, List[int]],
) -> List[int]:
    doc_ids: List[int] = []
    for sentence in processed_sentences:
        key = build_sentence_key(sentence)
        bucket = raw_mapping.get(key)
        if not bucket:
            raise ValueError("Sentence not found in raw corpus while assigning doc ids.")
        doc_ids.append(bucket.pop())
    return doc_ids


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_data_dir"]).resolve()
    processed_dir = Path(config["paths"]["processed_data_dir"]).resolve()

    train_file, dev_file = locate_raw_files(raw_dir)
    raw_sentences = read_raw_with_doc_ids([train_file, dev_file])

    # Build mapping from sentence serialization to doc ids (stack to avoid duplicates)
    raw_mapping: Dict[str, List[int]] = defaultdict(list)
    for sentence, doc_id in raw_sentences:
        raw_mapping[build_sentence_key(sentence)].append(doc_id)

    train_processed = read_conll_file(processed_dir / "train.txt")
    val_processed = read_conll_file(processed_dir / "validation.txt")
    combined = list(train_processed) + list(val_processed)

    train_doc_ids = map_processed_doc_ids(train_processed, raw_mapping)
    val_doc_ids = map_processed_doc_ids(val_processed, raw_mapping)
    combined_doc_ids = train_doc_ids + val_doc_ids

    n_samples = len(combined)
    groups = combined_doc_ids
    indices = list(range(n_samples))

    splitter = GroupKFold(n_splits=args.n_splits)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups), start=1):
        folds.append(
            {
                "fold_index": fold_idx,
                "train_indices": train_idx.tolist(),
                "validation_indices": val_idx.tolist(),
                "train_size": len(train_idx),
                "validation_size": len(val_idx),
            }
        )

    output = {
        "n_splits": args.n_splits,
        "total_sentences": n_samples,
        "folds": folds,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Wrote GroupKFold splits to {output_path} ({args.n_splits} folds)")


if __name__ == "__main__":
    main()
