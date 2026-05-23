from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from core.io import read_json, write_json
from core.schema import Chunk


def save_store(path: str | Path, chunks: list[Chunk], embeddings: np.ndarray, stats: dict[str, Any]) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "chunks.json", [chunk.to_dict() for chunk in chunks])
    write_json(root / "build_stats.json", stats)
    np.save(root / "embeddings.npy", embeddings)


def load_store(path: str | Path) -> tuple[list[Chunk], np.ndarray, dict[str, Any]]:
    root = Path(path)
    allowed = {field.name for field in fields(Chunk)}
    chunks = [
        Chunk(**normalize_chunk_row(row, allowed))
        for row in read_json(root / "chunks.json")
    ]
    embeddings = np.load(root / "embeddings.npy")
    stats = read_json(root / "build_stats.json")
    return chunks, embeddings, stats


def normalize_chunk_row(row: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    normalized = {key: value for key, value in row.items() if key in allowed}
    if "role" not in normalized and row.get("source_role"):
        normalized["role"] = row.get("source_role")
    return normalized


def store_exists(path: str | Path) -> bool:
    root = Path(path)
    return (root / "chunks.json").exists() and (root / "embeddings.npy").exists()
