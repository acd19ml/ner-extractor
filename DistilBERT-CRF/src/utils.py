"""Utility helpers for reproducibility, logging, and filesystem management."""

from __future__ import annotations

import datetime as _dt
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent during lightweight tasks
    torch = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on GPU availability
            torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: Iterable[Union[str, Path]]) -> Dict[str, Path]:
    """Create directories if they do not exist and return their absolute paths."""

    resolved: Dict[str, Path] = {}
    for original in paths:
        expanded = Path(original).expanduser().resolve()
        expanded.mkdir(parents=True, exist_ok=True)
        resolved[str(original)] = expanded
    return resolved


def timestamped_run_dir(root: Union[str, Path], prefix: str = "run") -> Path:
    """Create a timestamped subdirectory under ``root`` to store experiment artifacts."""

    base_dir = Path(root).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{prefix}_{timestamp}"
    counter = 1
    while candidate.exists():
        candidate = base_dir / f"{prefix}_{timestamp}_{counter}"
        counter += 1

    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def create_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a namespaced logger with optional file logging."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None and not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(Path(log_file).resolve())
        for handler in logger.handlers
    ):
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

