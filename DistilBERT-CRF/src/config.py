"""Configuration loading utilities for the DistilBERT-CRF project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and resolve relative paths.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        A dictionary representation of the configuration where entries under
        the ``paths`` section are converted to absolute :class:`pathlib.Path`
        instances relative to the configuration file's directory.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty or does not contain a
            top-level mapping.
    """

    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    if not isinstance(config_data, dict):
        raise ValueError(f"Configuration file must define a mapping: {resolved_path}")

    paths_section = config_data.get("paths")
    if isinstance(paths_section, Mapping):
        base_dir = resolved_path.parent
        resolved_paths = {}
        for key, value in paths_section.items():
            if value is None:
                resolved_paths[key] = None
            else:
                absolute_path = (base_dir / str(value)).expanduser().resolve()
                resolved_paths[key] = absolute_path
        config_data["paths"] = resolved_paths

    return config_data


def save_config(config: Mapping[str, Any], output_path: Union[str, Path]) -> Path:
    """Persist a configuration mapping to disk as YAML.

    Args:
        config: Configuration mapping to be serialized. Path-like objects under
            the ``paths`` key are automatically converted to relative strings.
        output_path: Destination path for the YAML file.

    Returns:
        Absolute :class:`pathlib.Path` to the written configuration file.
    """

    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    serializable: Dict[str, Any] = dict(config)
    paths_section = serializable.get("paths")
    if isinstance(paths_section, Mapping):
        serialized_paths = {}
        for key, value in paths_section.items():
            if isinstance(value, Path):
                try:
                    relative_value = value.relative_to(target_path.parent)
                except ValueError:
                    relative_value = value
                serialized_paths[key] = str(relative_value)
            else:
                serialized_paths[key] = value
        serializable["paths"] = serialized_paths

    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=False)

    return target_path

