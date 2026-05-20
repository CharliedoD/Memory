from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.memory.core.io import ensure_parent, load_jsonl
from src.memory.datasets.locomo import locomo_category_name


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokens(text: Any) -> list[str]:
    value = "" if text is None else str(text)
    return normalize(value).split()


def token_f1(prediction: Any, answer: Any) -> float:
    pred = tokens(prediction)
    gold = tokens(answer)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def corpus_bleu_4(rows: list[dict[str, Any]]) -> float:
    max_order = 4
    matches = [0] * max_order
    possible = [0] * max_order
    hyp_len = 0
    ref_len = 0
    for row in rows:
        hyp = tokens(row.get("hypothesis"))
        ref = tokens(row.get("answer"))
        hyp_len += len(hyp)
        ref_len += len(ref)
        for order in range(1, max_order + 1):
            hyp_counts = ngrams(hyp, order)
            ref_counts = ngrams(ref, order)
            matches[order - 1] += sum((hyp_counts & ref_counts).values())
            possible[order - 1] += max(len(hyp) - order + 1, 0)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    precisions = [math.log((m + 1) / (p + 1)) for m, p in zip(matches, possible) if p > 0]
    return bp * math.exp(sum(precisions) / len(precisions)) if precisions else 0.0


def ngrams(items: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(items[i : i + order]) for i in range(len(items) - order + 1))


def report(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in predictions if not row.get("error") and str(row.get("hypothesis") or "").strip()]
    build_rows = dedupe_by_memory(valid)
    overall = {
        "num_predictions": len(predictions),
        "num_valid": len(valid),
        **score_rows(valid),
        "build_tokens": sum(int(row.get("build_tokens") or 0) for row in build_rows),
        "query_tokens": sum(int(row.get("query_tokens") or 0) for row in valid),
        "build_time_seconds": sum(float(row.get("build_time_seconds") or 0.0) for row in build_rows),
        "query_time_seconds": sum(float(row.get("query_time_seconds") or 0.0) for row in valid),
    }
    return {
        **overall,
        "by_type": by_type_report(valid),
    }


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if "judge_label" in row]
    correct = sum(1 for row in judged if is_correct_label(row.get("judge_label")))
    return {
        "accuracy": correct / len(judged) if judged else None,
        "f1": mean(token_f1(row.get("hypothesis"), row.get("answer")) for row in rows) if rows else 0.0,
        "bleu": corpus_bleu_4(rows),
    }


def by_type_report(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(type_key(row), []).append(row)
    return {
        key: {
            "num_valid": len(group_rows),
            **score_rows(group_rows),
        }
        for key, group_rows in sorted(grouped.items())
    }


def type_key(row: dict[str, Any]) -> str:
    question_type = str(row.get("question_type") or "").strip()
    if question_type:
        match = re.fullmatch(r"locomo-category-(\d+)", question_type)
        if match:
            return locomo_category_name(match.group(1))
        return question_type
    category = row.get("category")
    if category is not None:
        return locomo_category_name(category)
    return "unknown"


def is_correct_label(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in {"yes", "correct", "true"}
    return False


def dedupe_by_memory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("memory_id") or row.get("sample_id") or index)
        selected.setdefault(key, row)
    return list(selected.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Report baseline metrics.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    result = report(load_jsonl(args.pred))
    output_format = "json" if args.json else args.format
    content = format_json(result) if output_format == "json" else format_markdown(result)
    print(content)
    if args.out:
        ensure_parent(args.out)
        args.out.write_text(content + "\n", encoding="utf-8")


def format_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def format_markdown(result: dict[str, Any]) -> str:
    rows = [
        ("num_predictions", result.get("num_predictions")),
        ("num_valid", result.get("num_valid")),
        ("accuracy", format_float(result.get("accuracy"))),
        ("f1", format_float(result.get("f1"))),
        ("bleu", format_float(result.get("bleu"))),
        ("build_tokens", result.get("build_tokens")),
        ("query_tokens", result.get("query_tokens")),
        ("build_time_seconds", format_float(result.get("build_time_seconds"))),
        ("query_time_seconds", format_float(result.get("query_time_seconds"))),
    ]
    lines = ["## Overall", "", "| metric | value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    by_type = result.get("by_type") or {}
    if by_type:
        lines.extend(["", "## By Type", "", "| type | num_valid | accuracy | f1 | bleu |", "|---|---:|---:|---:|---:|"])
        for key, values in by_type.items():
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    key,
                    values.get("num_valid"),
                    format_float(values.get("accuracy")),
                    format_float(values.get("f1")),
                    format_float(values.get("bleu")),
                )
            )
    return "\n".join(lines)


def format_float(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
