"""Data loading and tokenization pipeline for DistilBERT-CRF NER training."""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast

from augmentation import EntityAugmentationConfig, entity_aware_augmentation
from datasets import ConllSentence, read_conll_file

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NerLabels:
    """Container storing label mappings used across the project."""

    labels: List[str]
    label_to_id: Mapping[str, int]
    id_to_label: Mapping[int, str]


def collect_unique_labels(sentences: Iterable[ConllSentence]) -> NerLabels:
    """Derive sorted label vocabularies from a collection of sentences."""

    label_counter = Counter()
    for sentence in sentences:
        label_counter.update(sentence.ner_tags)

    labels = sorted(label_counter)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return NerLabels(labels=labels, label_to_id=label_to_id, id_to_label=id_to_label)


def load_processed_conll(processed_dir: Union[str, Path]) -> Dict[str, List[ConllSentence]]:
    """Load processed CoNLL splits (train/validation/test) from disk."""

    base_path = Path(processed_dir).expanduser().resolve()
    splits = {
        "train": base_path / "train.txt",
        "validation": base_path / "validation.txt",
        "test": base_path / "test.txt",
    }

    loaded: Dict[str, List[ConllSentence]] = {}
    for split_name, split_path in splits.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Missing processed split: {split_path}")
        loaded[split_name] = read_conll_file(split_path)
    return loaded


class TokenizedNERDataset(Dataset):
    """PyTorch dataset that tokenizes sentences and aligns BIO labels."""

    def __init__(
        self,
        sentences: Sequence[ConllSentence],
        tokenizer: PreTrainedTokenizerFast,
        label_to_id: Mapping[str, int],
        max_length: int,
        label_all_tokens: bool = False,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.sample_weights = sample_weights

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.sentences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sentence = self.sentences[index]
        encoding = self.tokenizer(
            sentence.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        if word_ids is None:
            raise ValueError("Tokenizer must provide word IDs for alignment.")

        labels = []
        previous_word_id: Optional[int] = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                label = sentence.ner_tags[word_id]
                label_id = self.label_to_id[label]
                if word_id != previous_word_id:
                    labels.append(label_id)
                else:
                    labels.append(label_id if self.label_all_tokens else -100)
            previous_word_id = word_id

        item = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        item["sentence_index"] = torch.tensor(index, dtype=torch.long)
        weight = 1.0
        if self.sample_weights is not None and index < len(self.sample_weights):
            weight = float(self.sample_weights[index])
        item["loss_weight"] = torch.tensor(weight, dtype=torch.float)
        return item


class NerDataCollator:
    """Custom collator that preserves loss weights alongside tokenization outputs."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.base = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        weights = torch.tensor(
            [float(feature.pop("loss_weight", 1.0)) for feature in features],
            dtype=torch.float,
        )
        batch = self.base(features)
        batch["loss_weight"] = weights
        return batch


def create_dataloaders(
    processed_dir: Union[str, Path],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    shuffle: bool = True,
    label_all_tokens: bool = False,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    augmentation_cfg: Optional[Mapping[str, Any]] = None,
    seed: int = 42,
    fold_indices: Optional[Mapping[str, Sequence[int]]] = None,
) -> Tuple[Dict[str, DataLoader], NerLabels]:
    """Construct PyTorch dataloaders for train/validation/test splits."""

    splits = load_processed_conll(processed_dir)

    def maybe_trim(sentences: List[ConllSentence], limit: Optional[int]) -> List[ConllSentence]:
        if limit is not None:
            return sentences[: max(0, limit)]
        return sentences

    if fold_indices:
        _apply_fold_indices(splits, fold_indices)

    if max_train_samples is not None:
        splits["train"] = maybe_trim(splits["train"], max_train_samples)
    if max_eval_samples is not None:
        splits["validation"] = maybe_trim(splits["validation"], max_eval_samples)
        splits["test"] = maybe_trim(splits["test"], max_eval_samples)

    train_weights: Optional[List[float]] = None
    if augmentation_cfg and augmentation_cfg.get("enabled", False):
        entity_cfg = _build_entity_aug_config(augmentation_cfg, seed)
        augmented = entity_aware_augmentation(splits["train"], entity_cfg)
        if augmented:
            base_len = len(splits["train"])
            splits["train"] = list(splits["train"]) + augmented
            train_weights = [1.0] * base_len + [entity_cfg.loss_weight] * len(augmented)
            rng = random.Random(entity_cfg.seed)
            combined = list(zip(splits["train"], train_weights))
            rng.shuffle(combined)
            splits["train"], train_weights = zip(*combined)
            splits["train"] = list(splits["train"])
            train_weights = list(train_weights)
            logger.info(
                "Augmented training split: base=%s augmented=%s total=%s",
                base_len,
                len(augmented),
                len(splits["train"]),
            )
        else:
            logger.info("Augmentation enabled but no candidates produced new sentences.")

    all_sentences: List[ConllSentence] = []
    for split_sentences in splits.values():
        all_sentences.extend(split_sentences)
    label_info = collect_unique_labels(all_sentences)
    eval_bs = eval_batch_size or batch_size

    datasets = {
        split_name: TokenizedNERDataset(
            sentences=split_sentences,
            tokenizer=tokenizer,
            label_to_id=label_info.label_to_id,
            max_length=max_length,
            label_all_tokens=label_all_tokens,
            sample_weights=train_weights if split_name == "train" else None,
        )
        for split_name, split_sentences in splits.items()
    }

    collator = NerDataCollator(tokenizer=tokenizer)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=num_workers,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=eval_bs,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_bs,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        ),
    }

    return dataloaders, label_info


def _build_entity_aug_config(cfg: Mapping[str, Any], fallback_seed: int) -> EntityAugmentationConfig:
    """Translate YAML dict into an :class:`EntityAugmentationConfig`."""

    max_gen = cfg.get("max_generated_samples")
    if max_gen is not None:
        try:
            max_gen = int(max_gen)
        except (TypeError, ValueError):
            max_gen = None

    return EntityAugmentationConfig(
        replace_prob=float(cfg.get("entity_replace_prob", 0.3)),
        max_replacements=int(cfg.get("max_entity_replacements", 2)),
        copies_per_sentence=int(cfg.get("copies_per_sample", 1)),
        max_generated=max_gen,
        loss_weight=float(cfg.get("loss_weight", 0.5)),
        seed=int(cfg.get("seed", fallback_seed)),
        shuffle=bool(cfg.get("shuffle", True)),
    )


def _apply_fold_indices(
    splits: Dict[str, List[ConllSentence]],
    fold_indices: Mapping[str, Sequence[int]],
) -> None:
    """Override train/validation splits using explicit index selections."""

    train_ids = fold_indices.get("train")
    val_ids = fold_indices.get("validation")
    if train_ids is None or val_ids is None:
        logger.warning("Fold indices provided without both train/validation keys. Ignoring overrides.")
        return

    pool = list(splits["train"]) + list(splits["validation"])
    total_available = len(pool)

    def select(indices: Sequence[int]) -> List[ConllSentence]:
        selected: List[ConllSentence] = []
        for idx in indices:
            if idx < 0 or idx >= total_available:
                raise IndexError(f"Fold index {idx} out of range (total={total_available}).")
            selected.append(pool[idx])
        return selected

    splits["train"] = select(train_ids)
    splits["validation"] = select(val_ids)
    logger.info(
        "Applied custom fold indices | train=%s validation=%s pool_size=%s",
        len(train_ids),
        len(val_ids),
        total_available,
    )
