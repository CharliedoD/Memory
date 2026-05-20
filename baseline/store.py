from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.memory.core.io import read_json, write_json
from src.memory.core.schema import Chunk


def save_store(path: str | Path, chunks: list[Chunk], embeddings: np.ndarray, stats: dict[str, Any]) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "chunks.json", [chunk.to_dict() for chunk in chunks])
    write_json(root / "build_stats.json", stats)
    np.save(root / "embeddings.npy", embeddings)


def load_store(path: str | Path) -> tuple[list[Chunk], np.ndarray, dict[str, Any]]:
    root = Path(path)
    chunks = [Chunk(**row) for row in read_json(root / "chunks.json")]
    embeddings = np.load(root / "embeddings.npy")
    stats = read_json(root / "build_stats.json")
    return chunks, embeddings, stats


def store_exists(path: str | Path) -> bool:
    root = Path(path)
    return (root / "chunks.json").exists() and (root / "embeddings.npy").exists()

