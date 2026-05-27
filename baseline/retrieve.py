from __future__ import annotations

import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from core.schema import Chunk, RetrievedChunk


def retrieve_top_k(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
    *,
    query_text: str = "",
    keyword_enabled: bool = True,
    overfetch_multiplier: int = 4,
    overfetch_min: int = 60,
) -> list[RetrievedChunk]:
    if not chunks or embeddings.size == 0:
        return []
    scores = embeddings @ query_embedding.reshape(-1)
    candidate_k = top_k
    if keyword_enabled and query_text.strip():
        candidate_k = max(top_k * overfetch_multiplier, overfetch_min)
    candidate_k = min(candidate_k, len(chunks))
    order = np.argsort(scores)[::-1][:candidate_k]
    bm25_scores = bm25_scores_by_index(chunks, query_text, top_k=candidate_k) if keyword_enabled else {}
    has_bm25 = bool(bm25_scores)
    max_possible = 2.0 if has_bm25 else 1.0
    scored_order: list[tuple[float, float, int]] = []
    for index in order:
        chunk_index = int(index)
        semantic_score = float(scores[chunk_index])
        bm25_score = bm25_scores.get(chunk_index, 0.0)
        combined_score = min((semantic_score + bm25_score) / max_possible, 1.0)
        scored_order.append((combined_score, semantic_score, chunk_index))
    scored_order.sort(key=lambda item: (item[0], item[1]), reverse=True)
    retrieved: list[RetrievedChunk] = []
    for rank, (combined_score, _semantic_score, index) in enumerate(scored_order[:top_k], start=1):
        chunk = chunks[index]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                event_date=chunk.event_date,
                session_id=chunk.session_id,
                score=combined_score,
                rank=rank,
                role=chunk.role,
            )
        )
    return retrieved


def bm25_scores_by_index(chunks: list[Chunk], query: str, top_k: int) -> dict[int, float]:
    query_terms = tokenize_for_bm25(query)
    if not query_terms:
        return {}

    docs = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    if not docs or not any(docs):
        return {}

    bm25 = BM25Okapi(docs, k1=1.5, b=0.75)
    scores = bm25.get_scores(query_terms)
    raw_scores = [(float(score), index) for index, score in enumerate(scores) if score > 0]

    if not raw_scores:
        return {}

    midpoint, steepness = bm25_normalization_params(len(query_terms))
    raw_scores.sort(reverse=True)
    return {
        index: normalize_bm25(raw_score, midpoint=midpoint, steepness=steepness)
        for raw_score, index in raw_scores[:top_k]
    }


def tokenize_for_bm25(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in STOPWORDS]


def bm25_normalization_params(num_terms: int) -> tuple[float, float]:
    if num_terms <= 3:
        return 5.0, 0.7
    if num_terms <= 6:
        return 7.0, 0.6
    if num_terms <= 9:
        return 9.0, 0.5
    if num_terms <= 15:
        return 10.0, 0.5
    return 12.0, 0.5


def normalize_bm25(raw_score: float, *, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
}
