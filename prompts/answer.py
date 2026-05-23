from __future__ import annotations

import hashlib
from typing import Any

from core.llm import extract_json_object
from core.schema import Example, RetrievedChunk


ANSWER_TEMPLATE = """Answer the user's question based on the provided context.

User Question: {query}

Relevant Context: {context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}

Now answer the question. Return ONLY the JSON, no other text.
"""


CATEGORY5_TEMPLATE = """Based on the context below, answer the following question.

Context:{context_str}

Question: {question}

Select the correct answer from the following two options. If the given answer is wrong or not answerable based on the context, you should choose "Not mentioned in the conversation".

Option A: {option_a}
Option B: {option_b}

Requirements:
1. Choose the option that best matches the context
2. If neither answer is supported by the context, or if the provided specific answer is incorrect, choose "Not mentioned in the conversation"
3. Return your response in JSON format

Output Format:
{{
  "reasoning": "Brief explanation of your choice",
  "answer": "Your selected answer"
}}

Return ONLY the JSON, no other text.
"""


def answer_messages(example: Example, retrieved: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = format_context(retrieved)
    if is_locomo_category5(example):
        option_a, option_b = category5_options(example)
        prompt = CATEGORY5_TEMPLATE.format(
            context_str=context,
            question=example.question,
            option_a=option_a,
            option_b=option_b,
        )
    else:
        query = example.question
        if example.question_date:
            query = f"Current Date: {example.question_date}\nQuestion: {example.question}"
        prompt = ANSWER_TEMPLATE.format(query=query, context_str=context)
    return [{"role": "user", "content": prompt}]


def format_context(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "None"
    blocks = []
    for display_index, chunk in enumerate(retrieved, start=1):
        event_date = chunk.event_date or chunk.date
        lines = [
            f"### Memory {display_index}",
            f"Event Date: {event_date}",
        ]
        if chunk.date and chunk.date != event_date:
            lines.append(f"Session Date: {chunk.date}")
        if chunk.role:
            lines.append(f"Role: {chunk.role}")
        lines.extend(["Content:", chunk.text])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_answer(raw_response: str) -> str:
    value = extract_json_object(raw_response)
    if value and value.get("answer") is not None:
        return str(value["answer"]).strip()
    return raw_response.strip()


def is_locomo_category5(example: Example) -> bool:
    return example.dataset == "locomo" and str(example.metadata.get("category")) == "5"


def category5_options(example: Example) -> tuple[str, str]:
    not_mentioned = "Not mentioned in the conversation"
    adversarial = str(example.metadata.get("adversarial_answer") or "")
    options = [not_mentioned, adversarial or not_mentioned]
    digest = hashlib.md5(example.sample_id.encode("utf-8")).hexdigest()
    if int(digest, 16) % 2:
        options.reverse()
    return options[0], options[1]
