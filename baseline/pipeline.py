from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.memory.baseline.chunking import build_chunks
from src.memory.baseline.retrieve import retrieve_top_k
from src.memory.baseline.store import load_store, save_store
from src.memory.core.embedding import EmbeddingClient
from src.memory.core.llm import ChatClient
from src.memory.core.schema import Example
from src.memory.prompts.answer import answer_messages, parse_answer


class NaiveRagBaseline:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        answer_client: ChatClient,
        config: dict[str, Any],
    ) -> None:
        self.embedding_client = embedding_client
        self.answer_client = answer_client
        self.config = config

    def build_memory(self, example: Example, store_dir: Path) -> dict[str, Any]:
        started = time.time()
        chunk_unit = str(self.config["retrieval"]["chunk_unit"])
        chunks = build_chunks(example, chunk_unit=chunk_unit)
        embedded = self.embedding_client.embed_documents(embedding_text(chunk) for chunk in chunks)
        stats = {
            "memory_id": example.memory_id,
            "num_chunks": len(chunks),
            "chunk_unit": chunk_unit,
            "build_tokens": embedded.tokens,
            "build_time_seconds": round(time.time() - started, 3),
        }
        save_store(store_dir, chunks, embedded.vectors, stats)
        return stats

    def answer(self, example: Example, store_dir: Path) -> dict[str, Any]:
        started = time.time()
        chunks, embeddings, build_stats = load_store(store_dir)
        query_embedding = self.embedding_client.embed_query(retrieval_query_text(example))
        top_k = int(self.config["retrieval"]["top_k"])
        retrieved = retrieve_top_k(chunks, embeddings, query_embedding.vectors[0], top_k)

        answer_cfg = self.config["answer"]
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=answer_messages(example, retrieved),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=int(answer_cfg["max_tokens"]),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        hypothesis = parse_answer(result.content)
        query_tokens = query_embedding.tokens + result.tokens
        return {
            "sample_id": example.sample_id,
            "memory_id": example.memory_id,
            "dataset": example.dataset,
            "question": example.question,
            "question_date": example.question_date,
            "question_type": example.question_type,
            "answer": example.answer,
            "hypothesis": hypothesis,
            "raw_response": result.content,
            "method": "naive_rag_baseline",
            "model": answer_cfg["name"],
            "embedding_model": self.config["embedding"]["name"],
            "chunk_unit": build_stats.get("chunk_unit"),
            "top_k": top_k,
            "num_chunks": build_stats.get("num_chunks", len(chunks)),
            "build_tokens": int(build_stats.get("build_tokens", 0)),
            "query_tokens": query_tokens,
            "build_time_seconds": float(build_stats.get("build_time_seconds", 0.0)),
            "query_time_seconds": round(time.time() - started, 3),
            "error": None,
        }


def embedding_text(chunk: Any) -> str:
    if chunk.date:
        return f"Date: {chunk.date}\nContent: {chunk.text}"
    return chunk.text


def retrieval_query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question
