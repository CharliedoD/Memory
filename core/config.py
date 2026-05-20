from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def get_config(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def set_if_not_none(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    if value is None:
        return
    target = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        target = target.setdefault(key, {})
    target[parts[-1]] = value

