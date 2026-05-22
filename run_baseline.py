from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from baseline.pipeline import NaiveRagBaseline
from baseline.store import store_exists
from core.config import expand_env_vars, get_config, load_config, set_if_not_none
from core.embedding import EmbeddingClient
from core.io import TeeLogger, append_jsonl, completed_ids, load_env_file
from core.llm import ChatClient
from core.schema import Example
from datasets import load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent Memory naive RAG baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--dataset", choices=("auto", "longmemeval", "locomo"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s_cleaned.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--failure-out", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment name. When set, default outputs and stores are isolated under outputs/<run-name>/.",
    )
    parser.add_argument("--mode", choices=("full", "build", "query", "answer"), default="full")
    parser.add_argument("--memory-mode", choices=("raw", "extract"), default=None)
    parser.add_argument("--chunk-unit", choices=("turn", "pair", "session"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--question-type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-stores",
        action="store_true",
        help="Delete and rebuild memory stores. Output overwrite is controlled separately by --overwrite.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel memory groups. Defaults: LoCoMo=2, LongMemEval=4.",
    )
    return parser.parse_args()


def make_baseline(config: dict[str, Any], answer_api_key: str) -> NaiveRagBaseline:
    embedding_cfg = config["embedding"]
    embedding_api_key_env = embedding_cfg.get("api_key_env")
    embedding_api_key = (
        os.environ.get(str(embedding_api_key_env), "")
        if embedding_api_key_env
        else str(embedding_cfg.get("api_key", "EMPTY"))
    )
    return NaiveRagBaseline(
        embedding_client=EmbeddingClient(
            model=str(embedding_cfg["name"]),
            base_url=str(embedding_cfg["base_url"]),
            api_key=embedding_api_key,
            backend=str(embedding_cfg.get("backend", "openai")),
            device=str(embedding_cfg.get("device") or "") or None,
            dtype=str(embedding_cfg.get("dtype", "float32")),
            batch_size=int(embedding_cfg.get("batch_size", 64)),
            normalize=bool(embedding_cfg.get("normalize", True)),
            query_instruction=str(embedding_cfg.get("query_instruction", "")),
            max_input_bytes=int(embedding_cfg.get("max_input_bytes", 0)),
            max_retries=int(embedding_cfg.get("max_retries", 8)),
            retry_base_seconds=float(embedding_cfg.get("retry_base_seconds", 2.0)),
        ),
        answer_client=ChatClient(
            api_key=answer_api_key,
            base_url=str(get_config(config, "answer.base_url", "")),
            timeout_seconds=float(get_config(config, "answer.timeout_seconds", 120.0)),
            max_retries=int(get_config(config, "answer.max_retries", 2)),
        ),
        config=config,
    )


def group_by_memory(indexed_examples: list[tuple[int, Example]]) -> list[tuple[str, list[tuple[int, Example]]]]:
    groups: OrderedDict[str, list[tuple[int, Example]]] = OrderedDict()
    for index, example in indexed_examples:
        groups.setdefault(example.memory_id, []).append((index, example))
    return list(groups.items())


def default_workers(examples: list[Example]) -> int:
    if examples and examples[0].dataset == "locomo":
        return 2
    return 4


def derived_failure_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.failures{out.suffix}")


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("--run-name cannot be empty after sanitization.")
    return cleaned


def resolve_run_paths(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    if not args.run_name:
        out = args.out or Path(str(get_config(config, "paths.prediction_file")))
        failure_out = args.failure_out or derived_failure_path(out)
        store_root = args.store_root or Path(str(get_config(config, "paths.store_root")))
        return out, failure_out, store_root, args.log_file, None

    run_dir = Path("outputs") / safe_run_name(args.run_name)
    out = args.out or run_dir / "predictions.jsonl"
    failure_out = args.failure_out or run_dir / "predictions.failures.jsonl"
    store_root = args.store_root or run_dir / "stores"
    log_file = args.log_file or run_dir / f"{args.mode}.log"
    return out, failure_out, store_root, log_file, run_dir


def error_record(example: Example, exc: Exception, *, stage: str, index: int | None = None) -> dict[str, Any]:
    return {
        "sample_id": example.sample_id,
        "memory_id": example.memory_id,
        "dataset": example.dataset,
        "index": index,
        "stage": stage,
        "question": example.question,
        "question_date": example.question_date,
        "question_type": example.question_type,
        "answer": example.answer,
        "method": "naive_rag_baseline",
        "error": repr(exc),
    }


def get_memory_mode(config: dict[str, Any]) -> str:
    return str(config.get("memory", {}).get("mode", "raw"))


def store_dir_for(store_root: Path, memory_id: str, config: dict[str, Any]) -> Path:
    memory_mode = get_memory_mode(config)
    if memory_mode == "raw":
        return store_root / memory_id
    chunk_unit = str(get_config(config, "retrieval.chunk_unit", "pair"))
    return store_root / f"{memory_mode}_{chunk_unit}" / memory_id


def progress_label(stage: str, current: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return f"[{stage} 0/0 0.0% |{'-' * width}|]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = round(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{stage} {current}/{total} {ratio * 100:5.1f}% |{bar}|]"


def format_inner_progress(memory_id: str, sample_index: int, total: int, event: dict[str, Any]) -> str:
    source_index = int(event.get("source_index") or 0)
    source_total = int(event.get("source_total") or 0)
    session_id = str(event.get("session_id") or "unknown")
    date = str(event.get("date") or "unknown")
    facts = int(event.get("facts") or 0)
    total_facts = int(event.get("total_facts") or 0)
    tokens = int(event.get("tokens") or 0)
    return (
        f"{progress_label('extract', source_index, source_total, width=18)} "
        f"sample={sample_index}/{total} memory={memory_id} "
        f"session={session_id} date={date} facts+={facts} total_facts={total_facts} tokens={tokens}"
    )


def main() -> None:
    args = parse_args()
    if args.mode == "answer":
        args.mode = "query"
    config = load_config(args.config)
    env_file = get_config(config, "paths.env_file", ".env")
    load_env_file(env_file)
    config = expand_env_vars(config)
    set_if_not_none(config, "memory.mode", args.memory_mode)
    set_if_not_none(config, "retrieval.chunk_unit", args.chunk_unit)
    set_if_not_none(config, "retrieval.top_k", args.top_k)

    out, failure_out, store_root, log_file, run_dir = resolve_run_paths(args, config)
    answer_api_key = os.environ.get(str(get_config(config, "answer.api_key_env", "OPENAI_API_KEY")), "")
    if not answer_api_key:
        raise ValueError("Missing answer model API key environment variable.")

    all_examples = load_examples(args.data, dataset=args.dataset)
    indexed_examples = list(enumerate(all_examples, start=1))
    if args.question_type:
        indexed_examples = [
            (index, example)
            for index, example in indexed_examples
            if example.question_type == args.question_type
        ]
    if args.start:
        indexed_examples = indexed_examples[args.start :]
    if args.limit:
        indexed_examples = indexed_examples[: args.limit]
    examples = [example for _, example in indexed_examples]

    if args.overwrite and out.exists() and args.mode != "build":
        out.unlink()
    if args.overwrite and failure_out.exists():
        failure_out.unlink()
    if args.overwrite and log_file and log_file.exists():
        log_file.unlink()
    done = set() if args.overwrite or args.mode == "build" else completed_ids(out)
    groups = group_by_memory(indexed_examples)
    workers = args.workers or default_workers(examples)
    workers = max(1, workers)

    with TeeLogger(log_file) as logger:
        log_lock = Lock()
        write_lock = Lock()
        progress_lock = Lock()
        total = len(examples)
        completed_groups = 0

        def log(message: str) -> None:
            with log_lock:
                logger.log(message)

        def write_record(record: dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(out, record)

        def write_failure(record: dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(failure_out, record)

        def next_completed_group() -> int:
            nonlocal completed_groups
            with progress_lock:
                completed_groups += 1
                return completed_groups

        def run_group(memory_id: str, group: list[tuple[int, Example]]) -> None:
            pending = [
                (index, example)
                for index, example in group
                if args.mode == "build" or example.sample_id not in done
            ]
            if args.mode != "build":
                for index, example in group:
                    if example.sample_id in done:
                        log(f"[{index}/{total}] skip {example.sample_id}")
                if not pending:
                    return

            baseline = make_baseline(config, answer_api_key)
            store_dir = store_dir_for(store_root, memory_id, config)

            try:
                if args.mode in {"full", "build"}:
                    if args.rebuild_stores and store_dir.exists():
                        shutil.rmtree(store_dir)
                    if args.rebuild_stores or not store_exists(store_dir):
                        first_index = group[0][0]
                        stats = baseline.build_memory(
                            group[0][1],
                            store_dir,
                            progress_callback=lambda event, first_index=first_index: log(
                                format_inner_progress(memory_id, first_index, total, event)
                            ),
                        )
                        done_count = next_completed_group()
                        log(
                            f"{progress_label('build', done_count, len(groups))} "
                            f"sample={first_index}/{total} "
                            f"memory {memory_id} built chunks={stats['num_chunks']}"
                        )
                    else:
                        done_count = next_completed_group()
                        first_index = group[0][0]
                        log(
                            f"{progress_label('build', done_count, len(groups))} "
                            f"sample={first_index}/{total} "
                            f"memory {memory_id} exists, skip"
                        )
            except Exception as exc:
                for index, example in pending:
                    write_failure(error_record(example, exc, stage="build", index=index))
                done_count = next_completed_group()
                first_index = group[0][0]
                log(
                    f"{progress_label('build', done_count, len(groups))} "
                    f"sample={first_index}/{total} "
                    f"memory {memory_id} build error: {exc}"
                )
                if args.stop_on_error:
                    raise
                return

            if args.mode == "build":
                return

            for index, example in pending:
                try:
                    record = baseline.answer(example, store_dir)
                    record["index"] = index
                    write_record(record)
                    log(f"[{index}/{total}] done {example.sample_id} answer={record['hypothesis'][:80]!r}")
                except Exception as exc:
                    write_failure(error_record(example, exc, stage="answer", index=index))
                    log(f"[{index}/{total}] error {example.sample_id}: {exc}")
                    if args.stop_on_error:
                        raise

        log(
            f"run start dataset={args.dataset} data={args.data} mode={args.mode} "
            f"question_type={args.question_type or 'all'} records={len(examples)} "
            f"memories={len(groups)} workers={workers} "
            f"run_name={args.run_name or 'none'} run_dir={run_dir or 'none'} "
            f"memory_mode={get_memory_mode(config)} out={out} failure_out={failure_out} store_root={store_root} "
            f"rebuild_stores={args.rebuild_stores}"
        )

        if workers == 1 or len(groups) <= 1:
            for memory_id, group in groups:
                run_group(memory_id, group)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_group, memory_id, group): memory_id
                    for memory_id, group in groups
                }
                for future in as_completed(futures):
                    memory_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log(f"[memory {memory_id}] failed: {exc}")
                        if args.stop_on_error:
                            raise

        if args.mode == "build":
            log(f"build done: {len(groups)} memory stores")
        log("run done")


if __name__ == "__main__":
    main()
