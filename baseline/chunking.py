from __future__ import annotations

from core.schema import Chunk, Example, Turn


def build_chunks(example: Example, *, chunk_unit: str) -> list[Chunk]:
    if chunk_unit == "turn":
        return _turn_chunks(example.turns)
    if chunk_unit == "pair":
        return _pair_chunks(example.turns)
    if chunk_unit == "session":
        return _session_chunks(example.turns)
    raise ValueError(f"Unsupported chunk_unit: {chunk_unit}")


def _turn_chunks(turns: list[Turn]) -> list[Chunk]:
    chunks = []
    for index, turn in enumerate(turns):
        text = f"{turn.role}: {turn.content}"
        chunks.append(Chunk(chunk_id=f"turn-{index:05d}", text=text, date=turn.date, session_id=turn.session_id))
    return chunks


def _pair_chunks(turns: list[Turn]) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending_user: Turn | None = None
    for turn in turns:
        if turn.role.lower() == "user":
            if pending_user:
                chunks.append(_single_turn_chunk(len(chunks), pending_user))
            pending_user = turn
            continue
        if pending_user:
            text = f"{pending_user.role}: {pending_user.content}\n{turn.role}: {turn.content}"
            chunks.append(Chunk(chunk_id=f"pair-{len(chunks):05d}", text=text, date=turn.date or pending_user.date, session_id=turn.session_id or pending_user.session_id))
            pending_user = None
        else:
            chunks.append(_single_turn_chunk(len(chunks), turn))
    if pending_user:
        chunks.append(_single_turn_chunk(len(chunks), pending_user))
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

    for turn in turns:
        session_id = turn.session_id or f"turn-{len(chunks):05d}"
        if current_session_id is not None and session_id != current_session_id:
            flush()
        current_session_id = session_id
        current_date = turn.date or current_date
        current_lines.append(f"{turn.role}: {turn.content}")

    flush()
    return chunks


def _single_turn_chunk(index: int, turn: Turn) -> Chunk:
    return Chunk(
        chunk_id=f"turn-{index:05d}",
        text=f"{turn.role}: {turn.content}",
        date=turn.date,
        session_id=turn.session_id,
    )
