"""Tokenizer utilities for DistilBERT-based NER models."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast


@lru_cache(maxsize=4)
def load_tokenizer(model_name_or_path: str, use_fast: bool = True) -> PreTrainedTokenizerFast:
    """Load a Hugging Face tokenizer with caching to avoid repeated downloads.

    Args:
        model_name_or_path: Hugging Face model identifier or local directory.
        use_fast: Whether to load the fast tokenizer variant.

    Returns:
        A :class:`PreTrainedTokenizerFast` instance configured for split tokens.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        add_prefix_space=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"Expected a fast tokenizer for DistilBERT-CRF pipeline, got {type(tokenizer).__name__}"
        )
    return tokenizer


def prepare_tokenizer(
    pretrained_model_name: str,
    max_length: Optional[int] = None,
) -> PreTrainedTokenizerFast:
    """Load and configure the tokenizer for sequence tagging."""

    tokenizer = load_tokenizer(pretrained_model_name)
    if max_length is not None:
        tokenizer.model_max_length = max_length
    # Ensure tokenizer will not silently truncate beyond the configured length
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    return tokenizer

