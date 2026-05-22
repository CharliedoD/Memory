from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.schema import Example
from datasets.locomo import load_locomo_records
from datasets.longmemeval import load_longmemeval_records


def load_examples(path: str | Path, dataset: str = "auto") -> list[Example]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = _records(payload)

    resolved = dataset
    if resolved == "auto":
        resolved = "locomo" if records and "conversation" in records[0] and "qa" in records[0] else "longmemeval"

    if resolved == "locomo":
        return load_locomo_records(records)
    if resolved == "longmemeval":
        return load_longmemeval_records(records)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("examples", []))
    if not isinstance(payload, list):
        raise ValueError("Dataset file must contain a list of examples.")
    return [record for record in payload if isinstance(record, dict)]

