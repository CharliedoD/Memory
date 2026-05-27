from __future__ import annotations

from dataclasses import dataclass

from core.llm import extract_json_object
from core.schema import Chunk


@dataclass(frozen=True)
class RewrittenMemory:
    text: str


EXTRACT_SYSTEM_PROMPT = """You are a Memory Rewriter. Rewrite a conversation chunk into one retrieval-friendly memory.

Rules:
- Do not answer any question.
- Generate exactly one concise rewritten memory.
- Preserve concrete entities, objects, dates, relative time expressions, quantities, locations, and named activities.
- Keep information that may help later retrieval, including user facts, assistant recommendations, plans, preferences, events, and constraints.
- Remove greetings, filler, generic acknowledgements, and repetitive wording.
- If the source uses relative time such as yesterday, last Friday, three days ago, next week, or last year, ground it in the rewritten text using Observation Date as the reference.
- Do not output metadata fields such as event_date, date, role, or source_id.
- Keep the style close to a search query / evidence sentence: specific, compact, and easy to match against a rewritten retrieval query.
- Return only valid JSON.

Output format:
{
  "memory": "single rewritten memory text"
}

If nothing useful should be remembered, return {"memory": ""}.
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


def parse_rewritten_memories(raw_response: str) -> list[RewrittenMemory]:
    value = extract_json_object(raw_response)
    if not value:
        return []
    memory = value.get("memory") or value.get("rewrite") or value.get("text") or value.get("content")
    if isinstance(memory, list):
        memory = memory[0] if memory else ""
    if isinstance(memory, dict):
        memory = memory.get("text") or memory.get("memory") or memory.get("content") or ""
    text = str(memory or "").strip()
    return [RewrittenMemory(text=text)] if text else []
