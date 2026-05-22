from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from core.config import expand_env_vars, get_config, load_config
from core.io import TeeLogger, append_jsonl, completed_ids, load_env_file, load_jsonl
from core.llm import ChatClient, extract_json_object
from core.schema import Example
from prompts.judge import judge_messages


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
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--pred", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--failure-out", type=Path, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment name. When set, default paths are read/written under outputs/<run-name>/.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--question-type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def derived_failure_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.failures{out.suffix}")


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("--run-name cannot be empty after sanitization.")
    return cleaned


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None, Path | None]:
    if not args.run_name:
        if not args.pred:
            raise ValueError("--pred is required unless --run-name is provided.")
        out = args.out or args.pred.with_suffix(args.pred.suffix + ".judge.jsonl")
        failure_out = args.failure_out or derived_failure_path(out)
        return args.pred, out, failure_out, args.log_file, None

    run_dir = Path("outputs") / safe_run_name(args.run_name)
    pred = args.pred or run_dir / "predictions.jsonl"
    out = args.out or run_dir / "predictions.judge.jsonl"
    failure_out = args.failure_out or run_dir / "judge.failures.jsonl"
    log_file = args.log_file or run_dir / "judge.log"
    return pred, out, failure_out, log_file, run_dir


def judge_failure_record(row: dict[str, Any], exc: Exception | str, *, index: int) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id") or row.get("question_id"),
        "memory_id": row.get("memory_id") or row.get("sample_id"),
        "dataset": row.get("dataset"),
        "index": index,
        "stage": "judge",
        "question": row.get("question"),
        "question_date": row.get("question_date"),
        "question_type": row.get("question_type"),
        "answer": row.get("answer"),
        "hypothesis": row.get("hypothesis"),
        "error": str(exc),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    load_env_file(get_config(config, "paths.env_file", ".env"))
    config = expand_env_vars(config)

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

    pred, out, failure_out, log_file, run_dir = resolve_paths(args)
    if args.overwrite and out.exists():
        out.unlink()
    if args.overwrite and failure_out.exists():
        failure_out.unlink()
    if args.overwrite and log_file and log_file.exists():
        log_file.unlink()

    rows = load_jsonl(pred)
    if args.question_type:
        rows = [row for row in rows if str(row.get("question_type") or "") == args.question_type]
    if args.limit:
        rows = rows[: args.limit]
    done = set() if args.overwrite else completed_ids(out)

    with TeeLogger(log_file) as logger:
        logger.log(
            f"judge start pred={pred} out={out} failure_out={failure_out} "
            f"run_name={args.run_name or 'none'} run_dir={run_dir or 'none'} "
            f"question_type={args.question_type or 'all'} records={len(rows)}"
        )
        for index, row in enumerate(rows, start=1):
            example = row_to_example(row)
            if example.sample_id in done:
                logger.log(f"[{index}/{len(rows)}] skip {example.sample_id}")
                continue
            if not str(row.get("hypothesis") or "").strip():
                append_jsonl(failure_out, judge_failure_record(row, "Missing hypothesis.", index=index))
                logger.log(f"[{index}/{len(rows)}] failure {example.sample_id}: missing hypothesis")
                continue
            try:
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
            except Exception as exc:
                append_jsonl(failure_out, judge_failure_record(row, exc, index=index))
                logger.log(f"[{index}/{len(rows)}] failure {example.sample_id}: {exc}")
        logger.log("judge done")


if __name__ == "__main__":
    main()
