"""Entity-aware data augmentation utilities for NER."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from seqeval.metrics.sequence_labeling import get_entities

from datasets import ConllSentence

logger = logging.getLogger(__name__)


@dataclass
class EntityAugmentationConfig:
    """Configurable knobs for entity-aware augmentation."""

    replace_prob: float = 0.3
    max_replacements: int = 2
    copies_per_sentence: int = 1
    max_generated: Optional[int] = None
    loss_weight: float = 0.5
    seed: int = 42
    shuffle: bool = True


def entity_aware_augmentation(
    sentences: Sequence[ConllSentence],
    config: EntityAugmentationConfig,
) -> List[ConllSentence]:
    """Create augmented copies of input sentences via type-consistent replacements."""

    if not sentences:
        return []

    rng = random.Random(config.seed)
    entity_pool = _build_entity_pool(sentences)
    augmented: List[ConllSentence] = []

    for sentence in sentences:
        for _ in range(max(1, config.copies_per_sentence)):
            augmented_sentence = _augment_single_sentence(sentence, entity_pool, config, rng)
            if augmented_sentence is not None:
                augmented.append(augmented_sentence)
                if config.max_generated and len(augmented) >= config.max_generated:
                    logger.info("Reached augmentation limit of %s sentences.", config.max_generated)
                    return augmented

    if config.shuffle:
        rng.shuffle(augmented)

    logger.info("Entity-aware augmentation produced %s sentences.", len(augmented))
    return augmented


def _build_entity_pool(sentences: Sequence[ConllSentence]) -> Dict[str, List[List[str]]]:
    """Collect entity spans (per label) across the corpus."""

    pool: DefaultDict[str, List[List[str]]] = DefaultDict(list)
    for sentence in sentences:
        spans = _extract_entity_spans(sentence.ner_tags)
        for start, end, label in spans:
            span_tokens = sentence.tokens[start : end + 1]
            if span_tokens:
                pool[label].append(span_tokens)
    return pool


def _augment_single_sentence(
    sentence: ConllSentence,
    entity_pool: Mapping[str, List[List[str]]],
    config: EntityAugmentationConfig,
    rng: random.Random,
) -> Optional[ConllSentence]:
    """Return an augmented variant of ``sentence`` or ``None`` if no replacement occurred."""

    spans = _extract_entity_spans(sentence.ner_tags)
    if not spans:
        return None

    selected: List[Tuple[int, int, str, List[str]]] = []
    replacements = 0

    for start, end, label in spans:
        if replacements >= max(1, config.max_replacements):
            break
        if rng.random() > config.replace_prob:
            continue
        candidates = entity_pool.get(label)
        if not candidates:
            continue
        candidate_tokens = rng.choice(candidates)
        original_tokens = sentence.tokens[start : end + 1]
        if candidate_tokens == original_tokens:
            continue
        selected.append((start, end, label, candidate_tokens))
        replacements += 1

    if not selected:
        return None

    selected.sort(key=lambda span: span[0])
    return _build_augmented_sentence(sentence, selected)


def _build_augmented_sentence(
    sentence: ConllSentence,
    selected_spans: Sequence[Tuple[int, int, str, Sequence[str]]],
) -> ConllSentence:
    """Construct a new sentence after applying replacements."""

    tokens: List[str] = []
    pos_tags: List[str] = []
    chunk_tags: List[str] = []
    ner_tags: List[str] = []

    span_iter = iter(selected_spans)
    current = next(span_iter, None)
    index = 0

    while index < len(sentence.tokens):
        if current and index == current[0]:
            start, end, label, replacement_tokens = current
            tokens.extend(replacement_tokens)
            pos_tags.extend(["NN"] * len(replacement_tokens))
            chunk_tags.extend(["O"] * len(replacement_tokens))
            ner_tags.extend(_build_bio_tags(label, len(replacement_tokens)))
            index = end + 1
            current = next(span_iter, None)
        else:
            tokens.append(sentence.tokens[index])
            pos_tags.append(sentence.pos_tags[index])
            chunk_tags.append(sentence.chunk_tags[index])
            ner_tags.append(sentence.ner_tags[index])
            index += 1

    return ConllSentence(tokens=tokens, pos_tags=pos_tags, chunk_tags=chunk_tags, ner_tags=ner_tags)


def _build_bio_tags(label: str, length: int) -> List[str]:
    """Return a BIO tag sequence for an entity of ``length`` tokens."""

    if length <= 0:
        return []
    tags = [f"B-{label}"]
    for _ in range(length - 1):
        tags.append(f"I-{label}")
    return tags


def _extract_entity_spans(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Get spans as (start, end, label) tuples from BIO-style tags."""

    spans = []
    for label, start, end in get_entities(tags):
        spans.append((start, end, label))
    return spans
