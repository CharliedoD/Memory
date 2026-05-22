from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(replace_env_var, value)
    return value


def replace_env_var(match: re.Match[str]) -> str:
    key = match.group(1)
    default = match.group(2)
    if key in os.environ:
        return os.environ[key]
    if default is not None:
        return default
    return ""


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
