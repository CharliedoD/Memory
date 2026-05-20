from __future__ import annotations

from typing import Any, Iterable

from src.memory.core.schema import Example, Turn


def load_longmemeval_records(records: list[dict[str, Any]]) -> list[Example]:
    examples: list[Example] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("question_id") or record.get("sample_id") or record.get("id") or index)
        sessions = record.get("haystack_sessions") or record.get("sessions") or []
        dates = record.get("haystack_dates") or []
        session_ids = record.get("haystack_session_ids") or []
        turns: list[Turn] = []

        for session_index, session in enumerate(sessions):
            date = str(dates[session_index]) if session_index < len(dates) else ""
            session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
            for role, content in iter_turns(session):
                text = " ".join(str(content).split())
                if text:
                    turns.append(Turn(role=role, content=text, date=date, session_id=session_id))

        examples.append(
            Example(
                sample_id=sample_id,
                memory_id=sample_id,
                dataset="longmemeval",
                question=str(record.get("question") or ""),
                question_date=record.get("question_date") or record.get("date"),
                answer=record.get("answer"),
                question_type=str(record.get("question_type") or ""),
                turns=turns,
                metadata=record,
            )
        )
    return examples


def iter_turns(session: Any) -> Iterable[tuple[str, str]]:
    if isinstance(session, list):
        for turn in session:
            if isinstance(turn, dict):
                yield str(turn.get("role") or turn.get("speaker") or "unknown"), str(turn.get("content", turn.get("text", "")))
        return

    if not isinstance(session, dict):
        return

    for key in ("turns", "messages"):
        turns = session.get(key)
        if isinstance(turns, list):
            for turn in turns:
                if isinstance(turn, dict):
                    yield str(turn.get("role") or turn.get("speaker") or "unknown"), str(turn.get("content", turn.get("text", "")))
            return

    for key, role in (("user", "user"), ("assistant", "assistant")):
        if key in session:
            yield role, str(session[key])

