from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.llm import extract_json_object
from core.schema import Chunk


@dataclass(frozen=True)
class ExtractedMemory:
    text: str
    event_date: str = ""
    role: str = ""


EXTRACT_SYSTEM_PROMPT = """You are a Memory Extractor. Extract self-contained, evidence-bound memories from a conversation chunk.

Rules:
- Extract only information that is explicitly present in the new chunk.
- Preserve specific dates, time expressions, entities, objects, quantities, and event details.
- Add event_date for every memory.
- event_date should be the actual date of the event/fact when it can be inferred.
- If the source contains a relative time expression such as yesterday, last Friday, three days ago, next week, or last year, compute event_date using Observation Date as the reference date.
- If the source does not contain an explicit or relative event time, set event_date to the Observation Date.
- Use ISO format YYYY-MM-DD when a specific day is known. If only month/year or year is known, preserve that granularity as YYYY-MM or YYYY.
- Never output relative phrases such as "yesterday", "last week", or "three days ago" in event_date.
- Split distinct events or facts into separate memories.
- Each input message is prefixed with a role label such as "user:" or "assistant:". For every memory, copy the role of the message that most directly supports it.
- If a memory is supported by multiple messages, use the role of the message containing the main new fact.
- Do not extract greetings, filler, or generic acknowledgements.
- Return only valid JSON.

Output format:
{
  "memory": [
    {"id": "0", "event_date": "2024-03-08", "role": "user", "text": "Self-contained memory text"},
    {"id": "1", "event_date": "2024-03-10", "role": "assistant", "text": "Another self-contained memory text"}
  ]
}

If nothing useful should be remembered, return {"memory": []}.
"""


def extraction_messages(chunk: Chunk) -> list[dict[str, str]]:
    user_prompt = "\n\n".join(
        [
            f"Observation Date: {chunk.date or 'unknown'}",
            f"Session ID: {chunk.session_id or 'unknown'}",
            "Messages:",
            chunk.text,
            "Return JSON only.",
        ]
    )
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_extracted_memories(raw_response: str) -> list[ExtractedMemory]:
    value = extract_json_object(raw_response)
    if not value:
        return []
    memories = value.get("memory") or value.get("memories") or value.get("facts") or value.get("data")
    if not isinstance(memories, list):
        return []

    extracted: list[ExtractedMemory] = []
    for item in memories:
        text = ""
        event_date = ""
        role = ""
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("memory") or item.get("fact") or item.get("content") or "").strip()
            event_date = str(_first_present(item, "event_date", "date", "time") or "").strip()
            role = str(_first_present(item, "role", "source_role", "speaker") or "").strip()
        elif isinstance(item, str):
            text = item.strip()
        if text:
            extracted.append(ExtractedMemory(text=text, event_date=event_date, role=role))
    return extracted


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None
