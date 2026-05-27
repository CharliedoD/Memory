from __future__ import annotations

import re

from core.schema import Chunk, Example, Turn


def build_chunks(example: Example, *, chunk_unit: str) -> list[Chunk]:
    positions = _turn_positions(example.turns)
    if chunk_unit == "sentence":
        return _sentence_chunks(example.turns, positions)
    if chunk_unit == "turn":
        return _turn_chunks(example.turns, positions)
    if chunk_unit == "pair":
        return _pair_chunks(example.turns, positions)
    if chunk_unit == "session":
        return _session_chunks(example.turns)
    raise ValueError(f"Unsupported chunk_unit: {chunk_unit}")


def _sentence_chunks(turns: list[Turn], positions: list[tuple[int, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for turn_index, turn in enumerate(turns):
        _source_id, session_id = positions[turn_index]
        for sentence in split_sentences(turn.content):
            chunks.append(
                Chunk(
                    chunk_id=f"sentence-{len(chunks):05d}",
                    text=_role_line(Turn(role=turn.role, content=sentence)),
                    date=turn.date,
                    session_id=session_id,
                )
            )
    return chunks


def _turn_chunks(turns: list[Turn], positions: list[tuple[int, str]]) -> list[Chunk]:
    chunks = []
    for index, turn in enumerate(turns):
        _source_id, session_id = positions[index]
        text = _role_line(turn)
        chunks.append(
            Chunk(
                chunk_id=f"turn-{index:05d}",
                text=text,
                date=turn.date,
                session_id=session_id,
            )
        )
    return chunks


def _pair_chunks(turns: list[Turn], positions: list[tuple[int, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending_user: tuple[int, Turn] | None = None
    for turn_index, turn in enumerate(turns):
        if pending_user and positions[pending_user[0]][1] != positions[turn_index][1]:
            pending_index, pending_turn = pending_user
            chunks.append(_single_turn_chunk(len(chunks), pending_turn, positions[pending_index]))
            pending_user = None
        if turn.role.lower() == "user":
            if pending_user:
                pending_index, pending_turn = pending_user
                chunks.append(_single_turn_chunk(len(chunks), pending_turn, positions[pending_index]))
            pending_user = (turn_index, turn)
            continue
        if pending_user:
            pending_index, pending_turn = pending_user
            _source_id, session_id = positions[turn_index]
            text = "\n".join(
                [
                    _role_line(pending_turn),
                    _role_line(turn),
                ]
            )
            chunks.append(
                Chunk(
                    chunk_id=f"pair-{len(chunks):05d}",
                    text=text,
                    date=turn.date or pending_turn.date,
                    session_id=session_id,
                )
            )
            pending_user = None
        else:
            chunks.append(_single_turn_chunk(len(chunks), turn, positions[turn_index]))
    if pending_user:
        pending_index, pending_turn = pending_user
        chunks.append(_single_turn_chunk(len(chunks), pending_turn, positions[pending_index]))
    return chunks


def _session_chunks(turns: list[Turn]) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_session_id = None
    current_date = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_session_id, current_date
        if not current_lines:
            return
        chunks.append(
            Chunk(
                chunk_id=f"session-{len(chunks):05d}",
                text="\n".join(current_lines),
                date=current_date,
                session_id=str(current_session_id or ""),
            )
        )
        current_lines = []

    for turn_index, turn in enumerate(turns):
        session_id = turn.session_id or f"turn-{turn_index:05d}"
        if current_session_id is not None and session_id != current_session_id:
            flush()
            current_date = ""
        current_session_id = session_id
        current_date = turn.date or current_date
        current_lines.append(_role_line(turn))

    flush()
    return chunks


def _single_turn_chunk(index: int, turn: Turn, position: tuple[int, str]) -> Chunk:
    _source_id, session_id = position
    return Chunk(
        chunk_id=f"turn-{index:05d}",
        text=_role_line(turn),
        date=turn.date,
        session_id=session_id,
    )


def _turn_positions(turns: list[Turn]) -> list[tuple[int, str]]:
    source_counts: dict[str, int] = {}
    positions: list[tuple[int, str]] = []
    for index, turn in enumerate(turns):
        session_id = turn.session_id or f"turn-{index:05d}"
        source_id = source_counts.get(session_id, 0)
        source_counts[session_id] = source_id + 1
        positions.append((source_id, session_id))
    return positions


def _role_line(turn: Turn) -> str:
    role = turn.role or "unknown"
    return f"{role}: {turn.content}"


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
