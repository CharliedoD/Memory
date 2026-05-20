from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def completed_ids(path: str | Path) -> set[str]:
    return {
        str(row.get("sample_id"))
        for row in load_jsonl(path)
        if row.get("hypothesis") and not row.get("error")
    }


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class TeeLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.file: TextIO | None = None
        if path:
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self.file:
            self.file.write(message + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self) -> "TeeLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
