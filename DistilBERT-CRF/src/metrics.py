"""Evaluation utilities for NER sequence labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


@dataclass(frozen=True)
class MetricsResult:
    """Container for aggregate metric scores."""

    precision: float
    recall: float
    f1: float
    accuracy: float


def align_predictions(
    predictions: Sequence[Sequence[int]],
    label_ids: Sequence[Sequence[int]],
    id_to_label: Mapping[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Align CRF predictions with gold BIO tags, skipping indices marked as -100."""

    preds: List[List[str]] = []
    refs: List[List[str]] = []

    for sequence_preds, sequence_labels in zip(predictions, label_ids):
        filtered_preds: List[str] = []
        filtered_labels: List[str] = []
        for pred_id, label_id in zip(sequence_preds, sequence_labels):
            if label_id == -100:
                continue
            filtered_preds.append(id_to_label[pred_id])
            filtered_labels.append(id_to_label[label_id])
        preds.append(filtered_preds)
        refs.append(filtered_labels)

    return preds, refs


def compute_ner_metrics(
    predictions: Sequence[Sequence[int]],
    label_ids: Sequence[Sequence[int]],
    id_to_label: Mapping[int, str],
) -> MetricsResult:
    """Compute seqeval precision/recall/F1/accuracy scores."""

    pred_labels, ref_labels = align_predictions(predictions, label_ids, id_to_label)

    precision = precision_score(ref_labels, pred_labels)
    recall = recall_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels)
    accuracy = accuracy_score(ref_labels, pred_labels)
    return MetricsResult(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def compute_entity_counts(labels: Iterable[Iterable[str]]) -> Dict[str, int]:
    """Count entity occurrences per type from BIO sequences."""

    counts: Dict[str, int] = {}
    for sequence in labels:
        for entity_type in (entity[0] for entity in get_entities(sequence)):
            counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts

