from __future__ import annotations

from core.llm import extract_json_object
from core.schema import Example


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user questions into a single evidence retrieval query for a memory system.

Rules:
- Do not answer the question.
- Generate exactly one concise search query that directly retrieves the facts needed to answer.
- Preserve concrete entities, objects, dates, quantities, locations, and named activities from the question.
- If the question date matters, include it only when it helps retrieve relative-date evidence.
- Remove generic wording like "which happened first", "how many days", or "what was the answer" when it does not help retrieval.
- Keep all key evidence targets in the same query instead of splitting them into multiple queries.
- Return only valid JSON.

Output format:
{
  "query": "single rewritten evidence query"
}
"""


def query_rewrite_messages(example: Example) -> list[dict[str, str]]:
    question = example.question
    if example.question_date:
        question = f"Question Date: {example.question_date}\nQuestion: {example.question}"
    user_prompt = "\n\n".join(
        [
            "Rewrite the question into one evidence retrieval query.",
            question,
            "Return JSON only.",
        ]
    )
    return [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_retrieval_query(raw_response: str) -> str:
    value = extract_json_object(raw_response)
    if not value:
        return ""
    raw_query = value.get("query") or value.get("retrieval_query") or value.get("search_query")
    if isinstance(raw_query, list):
        raw_query = " ".join(str(item or "").strip() for item in raw_query if str(item or "").strip())
    return str(raw_query or "").strip()
