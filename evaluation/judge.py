from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from src.memory.core.config import get_config, load_config
from src.memory.core.io import TeeLogger, append_jsonl, load_env_file, load_jsonl
from src.memory.core.llm import ChatClient, extract_json_object
from src.memory.core.schema import Example
from src.memory.prompts.judge import judge_messages


def parse_judge_label(dataset: str, response: str) -> bool | None:
    text = response.strip()
    if dataset == "locomo":
        value = extract_json_object(text)
        label = str((value or {}).get("label", "")).upper()
        if label == "CORRECT":
            return True
        if label == "WRONG":
            return False
        return None

    return parse_yes_no_label(text)


def parse_yes_no_label(text: str) -> bool | None:
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    candidates = [cleaned]
    candidates.extend(line.strip().lower() for line in cleaned.splitlines() if line.strip())

    for candidate in reversed(candidates):
        candidate = re.sub(r"</?[^>]+>", "", candidate).strip()
        candidate = candidate.strip("`'\" \t\r\n.,;:!")
        match = re.fullmatch(r"(?:answer\s*[:：]\s*)?(yes|no)", candidate)
        if match:
            return match.group(1) == "yes"
    return None


def row_to_example(row: dict[str, Any]) -> Example:
    return Example(
        sample_id=str(row.get("sample_id") or row.get("question_id") or ""),
        memory_id=str(row.get("memory_id") or row.get("sample_id") or ""),
        dataset=str(row.get("dataset") or "longmemeval"),
        question=str(row.get("question") or ""),
        question_date=row.get("question_date"),
        answer=row.get("answer"),
        question_type=str(row.get("question_type") or ""),
        turns=[],
        metadata=row,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge baseline predictions.")
    parser.add_argument("--config", type=Path, default=Path("src/memory/configs/base.yaml"))
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    load_env_file(get_config(config, "paths.env_file", ".env"))

    judge_cfg = config["judge"]
    api_key = os.environ.get(str(judge_cfg.get("api_key_env", "DEEPSEEK_API_KEY")), "")
    if not api_key:
        raise ValueError("Missing judge model API key environment variable.")
    client = ChatClient(
        api_key=api_key,
        base_url=str(judge_cfg.get("base_url") or ""),
        timeout_seconds=float(judge_cfg.get("timeout_seconds", 120.0)),
        max_retries=int(judge_cfg.get("max_retries", 2)),
    )

    out = args.out or args.pred.with_suffix(args.pred.suffix + ".judge.jsonl")
    if args.overwrite and out.exists():
        out.unlink()

    rows = load_jsonl(args.pred)
    if args.limit:
        rows = rows[: args.limit]

    with TeeLogger(args.log_file) as logger:
        logger.log(f"judge start pred={args.pred} out={out} records={len(rows)}")
        for index, row in enumerate(rows, start=1):
            example = row_to_example(row)
            result = client.complete(
                model=str(judge_cfg["name"]),
                messages=judge_messages(example, str(row.get("hypothesis") or "")),
                temperature=float(judge_cfg.get("temperature", 0.0)),
                max_tokens=int(judge_cfg.get("max_tokens", 8192)),
                thinking=str(judge_cfg.get("thinking", "default")),
                response_format={"type": "json_object"} if example.dataset == "locomo" else None,
            )
            judged = {
                **row,
                "judge_model": judge_cfg["name"],
                "judge_response": result.content,
                "judge_label": parse_judge_label(example.dataset, result.content),
                "judge_tokens": result.tokens,
            }
            append_jsonl(out, judged)
            logger.log(f"[{index}/{len(rows)}] {example.sample_id} label={judged['judge_label']}")
        logger.log("judge done")


if __name__ == "__main__":
    main()
