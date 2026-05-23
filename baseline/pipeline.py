from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from baseline.chunking import build_chunks
from baseline.retrieve import retrieve_top_k
from baseline.store import load_store, save_store
from core.embedding import EmbeddingClient
from core.llm import ChatClient
from core.schema import Chunk, Example, RetrievedChunk
from prompts.answer import answer_messages, parse_answer
from prompts.extract import extraction_messages, parse_extracted_memories


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

    def build_memory(
        self,
        example: Example,
        store_dir: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        chunk_unit = str(self.config["retrieval"]["chunk_unit"])
        memory_mode = str(self.config.get("memory", {}).get("mode", "raw"))
        chunks = build_chunks(example, chunk_unit=chunk_unit)
        source_chunks = len(chunks)
        extraction_tokens = 0
        if memory_mode == "extract":
            chunks, extraction_tokens = self.extract_memory_chunks(chunks, progress_callback=progress_callback)
        elif memory_mode != "raw":
            raise ValueError(f"Unsupported memory mode: {memory_mode}")

        embedded = self.embedding_client.embed_documents(embedding_text(chunk) for chunk in chunks)
        stats = {
            "memory_id": example.memory_id,
            "memory_mode": memory_mode,
            "num_source_chunks": source_chunks,
            "num_chunks": len(chunks),
            "chunk_unit": chunk_unit,
            "extraction_tokens": extraction_tokens,
            "build_tokens": embedded.tokens,
            "build_time_seconds": round(time.time() - started, 3),
        }
        save_store(store_dir, chunks, embedded.vectors, stats)
        return stats

    def extract_memory_chunks(
        self,
        chunks: list[Chunk],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[Chunk], int]:
        answer_cfg = self.config["answer"]
        memory_cfg = self.config.get("memory", {})
        extracted_chunks: list[Chunk] = []
        total_tokens = 0
        total_sources = len(chunks)
        for source_index, source in enumerate(chunks, start=1):
            result = self.answer_client.complete(
                model=str(answer_cfg["name"]),
                messages=extraction_messages(source),
                temperature=float(memory_cfg.get("extract_temperature", answer_cfg.get("temperature", 0.0))),
                max_tokens=int(memory_cfg.get("extract_max_tokens", 2048)),
                thinking=str(answer_cfg.get("thinking", "default")),
                response_format={"type": "json_object"},
            )
            total_tokens += result.tokens
            memories = parse_extracted_memories(result.content)
            for index, memory in enumerate(memories):
                extracted_chunks.append(
                    Chunk(
                        chunk_id=f"extract-{source.chunk_id}-{index:02d}",
                        text=memory.text,
                        date=source.date,
                        event_date=clean_event_date(memory.event_date, source.event_date or source.date),
                        session_id=source.session_id,
                        role=memory.role,
                    )
                )
            if progress_callback:
                progress_callback(
                    {
                        "stage": "extract",
                        "source_index": source_index,
                        "source_total": total_sources,
                        "source_chunk_id": source.chunk_id,
                        "session_id": source.session_id,
                        "date": source.date,
                        "facts": len(memories),
                        "total_facts": len(extracted_chunks),
                        "tokens": total_tokens,
                    }
                )
        return extracted_chunks, total_tokens

    def answer(self, example: Example, store_dir: Path) -> dict[str, Any]:
        started = time.time()
        chunks, embeddings, build_stats = load_store(store_dir)
        query_embedding = self.embedding_client.embed_query(retrieval_query_text(example))
        top_k = int(self.config["retrieval"]["top_k"])
        retrieved = retrieve_top_k(chunks, embeddings, query_embedding.vectors[0], top_k)
        retrieved = sort_retrieved_timeline(retrieved)
        retrieved_chunks = [chunk.to_dict() for chunk in retrieved]
        retrieved_session_ids = unique_nonempty(chunk.session_id for chunk in retrieved)
        gold_session_ids = [str(value) for value in example.metadata.get("answer_session_ids", [])]
        recall_stats = evidence_recall(gold_session_ids, retrieved_session_ids)

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
        extraction_tokens = int(build_stats.get("extraction_tokens", 0))
        embedding_build_tokens = int(build_stats.get("build_tokens", 0))
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
            "memory_mode": build_stats.get("memory_mode", "raw"),
            "chunk_unit": build_stats.get("chunk_unit"),
            "top_k": top_k,
            "context_order": "event_date_session_date_rank",
            "gold_session_ids": gold_session_ids,
            "retrieved_session_ids": retrieved_session_ids,
            "retrieved_chunks": retrieved_chunks,
            **recall_stats,
            "num_chunks": build_stats.get("num_chunks", len(chunks)),
            "build_tokens": extraction_tokens,
            "extraction_tokens": extraction_tokens,
            "embedding_build_tokens": embedding_build_tokens,
            "query_tokens": query_tokens,
            "build_time_seconds": float(build_stats.get("build_time_seconds", 0.0)),
            "query_time_seconds": round(time.time() - started, 3),
            "error": None,
        }


def embedding_text(chunk: Any) -> str:
    event_date = getattr(chunk, "event_date", "") or getattr(chunk, "date", "")
    session_date = getattr(chunk, "date", "")
    if event_date and session_date and event_date != session_date:
        return f"Event Date: {event_date}\nSession Date: {session_date}\nContent: {chunk.text}"
    if event_date:
        return f"Event Date: {event_date}\nContent: {chunk.text}"
    return chunk.text


def retrieval_query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question


def sort_retrieved_timeline(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(retrieved, key=retrieved_timeline_key)


def retrieved_timeline_key(chunk: RetrievedChunk) -> tuple[datetime, datetime, int]:
    return (
        parse_date_for_sort(chunk.event_date or chunk.date),
        parse_date_for_sort(chunk.date),
        chunk.rank,
    )


def clean_event_date(event_date: str, fallback_date: str) -> str:
    value = str(event_date or "").strip()
    fallback = str(fallback_date or "").strip()
    if value and parse_date_for_sort(value) != datetime.max:
        return value
    return fallback


def parse_date_for_sort(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.max
    text = re.sub(r"^[A-Za-z]+,\s+", "", text)
    candidates = [
        text,
        text.replace("/", "-"),
        text.replace(".", "-"),
    ]
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m",
        "%Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    for candidate in candidates:
        try:
            iso_candidate = candidate.replace("Z", "+00:00") if candidate.endswith("Z") else candidate
            return datetime.fromisoformat(iso_candidate).replace(tzinfo=None)
        except ValueError:
            pass
        for date_format in formats:
            try:
                return datetime.strptime(candidate, date_format)
            except ValueError:
                continue
    return datetime.max


def unique_nonempty(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def evidence_recall(gold_session_ids: list[str], retrieved_session_ids: list[str]) -> dict[str, Any]:
    gold = set(gold_session_ids)
    retrieved = set(retrieved_session_ids)
    hits = sorted(gold & retrieved)
    total = len(gold)
    return {
        "evidence_session_hits": hits,
        "evidence_session_hit_count": len(hits),
        "evidence_session_total": total,
        "evidence_session_recall": (len(hits) / total) if total else None,
        "evidence_session_all_found": bool(gold) and gold <= retrieved,
    }
