from __future__ import annotations

from core.llm import extract_json_object
from core.schema import Example


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user questions into evidence retrieval queries for a memory system.

Rules:
- Do not answer the question.
- Generate short search queries that directly retrieve the facts needed to answer.
- If the question compares, orders, or calculates time across multiple events, create one query per event.
- Preserve concrete entities, objects, dates, quantities, locations, and named activities from the question.
- If the question date matters, include it only when it helps retrieve relative-date evidence.
- Avoid generic words like "which happened first" unless they are part of a concrete event.
- Return only valid JSON.

Output format:
{
  "queries": [
    "first evidence query",
    "second evidence query"
  ]
}
"""


def query_rewrite_messages(example: Example, max_queries: int) -> list[dict[str, str]]:
    question = example.question
    if example.question_date:
        question = f"Question Date: {example.question_date}\nQuestion: {example.question}"
    user_prompt = "\n\n".join(
        [
            "Rewrite the question into evidence retrieval queries.",
            f"Maximum queries: {max_queries}",
            question,
            "Return JSON only.",
        ]
    )
    return [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_retrieval_queries(raw_response: str, *, max_queries: int) -> list[str]:
    value = extract_json_object(raw_response)
    if not value:
        return []
    raw_queries = value.get("queries") or value.get("query") or value.get("search_queries")
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        return []

    queries: list[str] = []
    seen: set[str] = set()
    for item in raw_queries:
        query = str(item or "").strip()
        key = " ".join(query.lower().split())
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries
