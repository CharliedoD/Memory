from __future__ import annotations

from core.llm import extract_json_object
from core.schema import Chunk


EXTRACT_SYSTEM_PROMPT = """You are a Memory Extractor. Extract self-contained, evidence-bound memories from a conversation chunk.

Rules:
- Extract only information that is explicitly present in the new chunk.
- Do not use previous memories or recent messages; this task has no external context.
- Preserve specific dates, time expressions, entities, objects, quantities, and event details.
- If a relative time expression appears, ground it to the provided observation date when possible.
- Split distinct events or facts into separate memories.
- Do not extract greetings, filler, or generic acknowledgements.
- Return only valid JSON.

Output format:
{
  "memory": [
    {"id": "0", "text": "Self-contained memory text"},
    {"id": "1", "text": "Another self-contained memory text"}
  ]
}

If nothing useful should be remembered, return {"memory": []}.
"""


def extraction_messages(chunk: Chunk) -> list[dict[str, str]]:
    user_prompt = "\n\n".join(
        [
            f"Observation Date: {chunk.date or 'unknown'}",
            f"Session ID: {chunk.session_id or 'unknown'}",
            "New Chunk:",
            chunk.text,
            "Return JSON only.",
        ]
    )
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_extracted_memories(raw_response: str) -> list[str]:
    value = extract_json_object(raw_response)
    if not value:
        return []
    memories = value.get("memory")
    if not isinstance(memories, list):
        return []

    extracted: list[str] = []
    for item in memories:
        text = ""
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("memory") or "").strip()
        elif isinstance(item, str):
            text = item.strip()
        if text:
            extracted.append(text)
    return extracted
