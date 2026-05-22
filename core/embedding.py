from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from openai import OpenAI, RateLimitError


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    tokens: int


class EmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        batch_size: int = 64,
        normalize: bool = True,
        query_instruction: str = "",
        max_input_bytes: int = 0,
        max_retries: int = 8,
        retry_base_seconds: float = 2.0,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.max_input_bytes = max_input_bytes
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        return self._embed(texts)

    def embed_query(self, text: str) -> EmbeddingResult:
        query = text
        if "qwen3" in self.model.lower() and self.query_instruction:
            query = f"Instruct: {self.query_instruction}\nQuery:{text}"
        return self._embed([query])

    def _embed(self, texts: Iterable[str]) -> EmbeddingResult:
        items = [truncate_utf8(" ".join(str(text).split()), self.max_input_bytes) for text in texts]
        if not items:
            return EmbeddingResult(np.empty((0, 0), dtype=np.float32), 0)

        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            response = self._create_embedding(batch)
            vectors.extend(row.embedding for row in sorted(response.data, key=lambda item: item.index))
            usage = getattr(response, "usage", None)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

        arr = np.asarray(vectors, dtype=np.float32)
        if self.normalize and arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        return EmbeddingResult(arr, total_tokens)

    def _create_embedding(self, batch: list[str]):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.embeddings.create(model=self.model, input=batch)
            except RateLimitError:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_base_seconds * (2 ** min(attempt, 5))
                delay += random.uniform(0.0, self.retry_base_seconds)
                time.sleep(delay)

        raise RuntimeError("Embedding retry loop exited unexpectedly.")


def truncate_utf8(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    marker = "\n...[truncated]...\n".encode("utf-8")
    budget = max(0, max_bytes - len(marker))
    head = budget // 2
    tail = budget - head
    head_text = encoded[:head].decode("utf-8", errors="ignore")
    tail_text = encoded[-tail:].decode("utf-8", errors="ignore") if tail else ""
    return head_text + marker.decode("utf-8") + tail_text
