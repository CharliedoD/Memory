from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    date: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class Example:
    sample_id: str
    memory_id: str
    dataset: str
    question: str
    answer: Any
    turns: list[Turn]
    question_date: str | None = None
    question_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    date: str = ""
    event_date: str = ""
    session_id: str = ""
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: item for key, item in value.items() if item != ""}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    date: str
    session_id: str
    score: float
    rank: int
    event_date: str = ""
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
