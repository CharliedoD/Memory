from __future__ import annotations

import numpy as np

from core.schema import Chunk, RetrievedChunk


def retrieve_top_k(chunks: list[Chunk], embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    if not chunks or embeddings.size == 0:
        return []
    scores = embeddings @ query_embedding.reshape(-1)
    order = np.argsort(scores)[::-1][:top_k]
    retrieved: list[RetrievedChunk] = []
    for rank, index in enumerate(order, start=1):
        chunk = chunks[int(index)]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                session_id=chunk.session_id,
                score=float(scores[int(index)]),
                rank=rank,
            )
        )
    return retrieved

