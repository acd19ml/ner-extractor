"""Utilities for loading and preparing the CoNLL03 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from sklearn.model_selection import train_test_split


@dataclass
class ConllSentence:
    """Represents a single CoNLL sentence with token-level annotations."""

    tokens: List[str]
    pos_tags: List[str]
    chunk_tags: List[str]
    ner_tags: List[str]

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> "ConllSentence":
        tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"Malformed CoNLL line: {line}")
            token, pos, chunk, ner = parts
            tokens.append(token)
            pos_tags.append(pos)
            chunk_tags.append(chunk)
            ner_tags.append(ner)
        return cls(tokens=tokens, pos_tags=pos_tags, chunk_tags=chunk_tags, ner_tags=ner_tags)

    def to_lines(self) -> List[str]:
        return [f"{w} {p} {c} {n}" for w, p, c, n in zip(self.tokens, self.pos_tags, self.chunk_tags, self.ner_tags)]

    def primary_label(self) -> str:
        for tag in self.ner_tags:
            if tag != "O":
                return tag.split("-", maxsplit=1)[-1]
        return "O"


def read_conll_file(path: Union[str, Path]) -> List[ConllSentence]:
    """Parse a CoNLL formatted file into a list of :class:`ConllSentence` objects."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"CoNLL file not found: {resolved}")

    sentences: List[ConllSentence] = []
    current: List[str] = []

    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if current:
                    sentences.append(ConllSentence.from_lines(current))
                    current.clear()
                continue
            if stripped.startswith("-DOCSTART-"):
                continue
            current.append(stripped)

    if current:
        sentences.append(ConllSentence.from_lines(current))

    return sentences


def write_conll_file(sentences: Iterable[ConllSentence], path: Union[str, Path]) -> Path:
    """Write a sequence of :class:`ConllSentence` instances back to CoNLL format."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            for line in sentence.to_lines():
                handle.write(f"{line}\n")
            handle.write("\n")

    return target


def stratified_split(
    sentences: Sequence[ConllSentence],
    val_ratio: float,
    seed: int,
) -> Tuple[List[ConllSentence], List[ConllSentence]]:
    """Split sentences into train/validation sets while preserving entity distribution."""

    if not 0 < val_ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1.")

    if len(sentences) < 2:
        raise ValueError("At least two sentences required for stratified split.")

    labels = [sentence.primary_label() for sentence in sentences]
    indices = list(range(len(sentences)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_sentences = [sentences[i] for i in sorted(train_idx)]
    val_sentences = [sentences[i] for i in sorted(val_idx)]
    return train_sentences, val_sentences


def prepare_conll03_corpus(
    raw_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Path]:
    """Read raw CoNLL03 files and create processed train/val/test splits."""

    raw_path = Path(raw_dir).expanduser().resolve()
    processed_path = Path(processed_dir).expanduser().resolve()
    processed_path.mkdir(parents=True, exist_ok=True)

    candidates = {
        "train": ["train.txt", "eng.train"],
        "validation": ["validation.txt", "valid.txt", "dev.txt", "eng.testa"],
        "test": ["test.txt", "eng.testb"],
    }

    def locate_file(kind: str) -> Path:
        for name in candidates[kind]:
            candidate = raw_path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing {kind} file in {raw_path}. Expected one of {candidates[kind]}")

    train_file = locate_file("train")
    dev_file = None
    try:
        dev_file = locate_file("validation")
    except FileNotFoundError:
        dev_file = None
    test_file = locate_file("test")

    train_sentences = read_conll_file(train_file)
    if dev_file:
        dev_sentences = read_conll_file(dev_file)
        train_sentences.extend(dev_sentences)

    train_split, val_split = stratified_split(train_sentences, val_ratio=val_ratio, seed=seed)
    test_sentences = read_conll_file(test_file)

    output_paths = {
        "train": processed_path / "train.txt",
        "validation": processed_path / "validation.txt",
        "test": processed_path / "test.txt",
    }

    write_conll_file(train_split, output_paths["train"])
    write_conll_file(val_split, output_paths["validation"])
    write_conll_file(test_sentences, output_paths["test"])

    return output_paths

