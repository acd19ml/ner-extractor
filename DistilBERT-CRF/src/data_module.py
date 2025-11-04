"""Data loading and tokenization pipeline for DistilBERT-CRF NER training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast

from datasets import ConllSentence, read_conll_file


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
    ) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

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
        return item


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
) -> Tuple[Dict[str, DataLoader], NerLabels]:
    """Construct PyTorch dataloaders for train/validation/test splits."""

    splits = load_processed_conll(processed_dir)

    def maybe_trim(sentences: List[ConllSentence], limit: Optional[int]) -> List[ConllSentence]:
        if limit is not None:
            return sentences[: max(0, limit)]
        return sentences

    if max_train_samples is not None:
        splits["train"] = maybe_trim(splits["train"], max_train_samples)
    if max_eval_samples is not None:
        splits["validation"] = maybe_trim(splits["validation"], max_eval_samples)
        splits["test"] = maybe_trim(splits["test"], max_eval_samples)

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
        )
        for split_name, split_sentences in splits.items()
    }

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")

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
