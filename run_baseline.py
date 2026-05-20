from __future__ import annotations

import argparse
import os
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from src.memory.baseline.pipeline import NaiveRagBaseline
from src.memory.baseline.store import store_exists
from src.memory.core.config import get_config, load_config, set_if_not_none
from src.memory.core.embedding import EmbeddingClient
from src.memory.core.io import TeeLogger, append_jsonl, completed_ids, load_env_file
from src.memory.core.llm import ChatClient
from src.memory.core.schema import Example
from src.memory.datasets import load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent Memory naive RAG baseline.")
    parser.add_argument("--config", type=Path, default=Path("src/memory/configs/base.yaml"))
    parser.add_argument("--dataset", choices=("auto", "longmemeval", "locomo"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("src/memory/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("full", "build", "query"), default="full")
    parser.add_argument("--chunk-unit", choices=("turn", "pair"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
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
    return NaiveRagBaseline(
        embedding_client=EmbeddingClient(
            model=str(embedding_cfg["name"]),
            base_url=str(embedding_cfg["base_url"]),
            api_key=str(embedding_cfg.get("api_key", "EMPTY")),
            batch_size=int(embedding_cfg.get("batch_size", 64)),
            normalize=bool(embedding_cfg.get("normalize", True)),
            query_instruction=str(embedding_cfg.get("query_instruction", "")),
            max_input_bytes=int(embedding_cfg.get("max_input_bytes", 0)),
        ),
        answer_client=ChatClient(
            api_key=answer_api_key,
            base_url=str(get_config(config, "answer.base_url", "")),
        ),
        config=config,
    )


def group_by_memory(examples: list[Example]) -> list[tuple[str, list[tuple[int, Example]]]]:
    groups: OrderedDict[str, list[tuple[int, Example]]] = OrderedDict()
    for index, example in enumerate(examples, start=1):
        groups.setdefault(example.memory_id, []).append((index, example))
    return list(groups.items())


def default_workers(examples: list[Example]) -> int:
    if examples and examples[0].dataset == "locomo":
        return 2
    return 4


def error_record(example: Example, exc: Exception) -> dict[str, Any]:
    return {
        "sample_id": example.sample_id,
        "memory_id": example.memory_id,
        "dataset": example.dataset,
        "question": example.question,
        "answer": example.answer,
        "hypothesis": "",
        "method": "naive_rag_baseline",
        "error": repr(exc),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_if_not_none(config, "retrieval.chunk_unit", args.chunk_unit)
    set_if_not_none(config, "retrieval.top_k", args.top_k)

    env_file = get_config(config, "paths.env_file", ".env")
    load_env_file(env_file)

    out = args.out or Path(str(get_config(config, "paths.prediction_file")))
    store_root = args.store_root or Path(str(get_config(config, "paths.store_root")))
    answer_api_key = os.environ.get(str(get_config(config, "answer.api_key_env", "LOCAL_LLM_API_KEY")), "")
    if not answer_api_key:
        raise ValueError("Missing answer model API key environment variable.")

    examples = load_examples(args.data, dataset=args.dataset)
    if args.start:
        examples = examples[args.start :]
    if args.limit:
        examples = examples[: args.limit]

    if args.overwrite and out.exists() and args.mode != "build":
        out.unlink()
    if args.overwrite and args.log_file and args.log_file.exists():
        args.log_file.unlink()
    done = set() if args.overwrite or args.mode == "build" else completed_ids(out)
    groups = group_by_memory(examples)
    workers = args.workers or default_workers(examples)
    workers = max(1, workers)

    with TeeLogger(args.log_file) as logger:
        log_lock = Lock()
        write_lock = Lock()
        total = len(examples)

        def log(message: str) -> None:
            with log_lock:
                logger.log(message)

        def write_record(record: dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(out, record)

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
            store_dir = store_root / memory_id

            try:
                if args.mode in {"full", "build"}:
                    if args.overwrite and store_dir.exists():
                        shutil.rmtree(store_dir)
                    if args.overwrite or not store_exists(store_dir):
                        stats = baseline.build_memory(group[0][1], store_dir)
                        log(f"[memory {memory_id}] built chunks={stats['num_chunks']}")
            except Exception as exc:
                if args.mode != "build":
                    for _, example in pending:
                        write_record(error_record(example, exc))
                log(f"[memory {memory_id}] build error: {exc}")
                if args.stop_on_error:
                    raise
                return

            if args.mode == "build":
                return

            for index, example in pending:
                try:
                    record = baseline.answer(example, store_dir)
                    write_record(record)
                    log(f"[{index}/{total}] done {example.sample_id} answer={record['hypothesis'][:80]!r}")
                except Exception as exc:
                    write_record(error_record(example, exc))
                    log(f"[{index}/{total}] error {example.sample_id}: {exc}")
                    if args.stop_on_error:
                        raise

        log(
            f"run start dataset={args.dataset} data={args.data} mode={args.mode} "
            f"records={len(examples)} memories={len(groups)} workers={workers} "
            f"out={out} store_root={store_root}"
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
