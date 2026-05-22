from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from core.io import ensure_parent, load_jsonl
from datasets.locomo import locomo_category_name


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
    extraction_tokens = sum(int(row.get("extraction_tokens") or 0) for row in build_rows)
    embedding_build_tokens = sum(embedding_tokens_for_row(row) for row in build_rows)
    build_tokens = extraction_tokens
    query_tokens = sum(int(row.get("query_tokens") or 0) for row in valid)
    overall = {
        "num_predictions": len(predictions),
        "num_valid": len(valid),
        **score_rows(valid),
        **retrieval_rows(valid),
        "build_tokens": build_tokens,
        "extraction_tokens": extraction_tokens,
        "embedding_build_tokens": embedding_build_tokens,
        "query_tokens": query_tokens,
        "build_tokens_per_valid": build_tokens / len(valid) if valid else None,
        "query_tokens_per_valid": query_tokens / len(valid) if valid else None,
        "total_tokens_per_valid": (build_tokens + query_tokens) / len(valid) if valid else None,
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
            **retrieval_rows(group_rows),
        }
        for key, group_rows in sorted(grouped.items())
    }


def retrieval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall_rows = [row for row in rows if row.get("evidence_session_recall") is not None]
    all_found_rows = [row for row in rows if row.get("evidence_session_total")]
    return {
        "evidence_session_recall": mean(float(row.get("evidence_session_recall")) for row in recall_rows)
        if recall_rows
        else None,
        "evidence_session_all_found_rate": (
            sum(1 for row in all_found_rows if row.get("evidence_session_all_found")) / len(all_found_rows)
        )
        if all_found_rows
        else None,
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


def embedding_tokens_for_row(row: dict[str, Any]) -> int:
    if "embedding_build_tokens" in row:
        return int(row.get("embedding_build_tokens") or 0)
    if "extraction_tokens" not in row:
        return int(row.get("build_tokens") or 0)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Report baseline metrics.")
    parser.add_argument("--pred", type=Path, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment name. When set, default prediction/report paths are under outputs/<run-name>/.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json", "html"), default="markdown")
    args = parser.parse_args()
    pred, out = resolve_metric_paths(args)
    result = report(load_jsonl(pred))
    output_format = "json" if args.json else args.format
    if output_format == "json":
        content = format_json(result)
    elif output_format == "html":
        content = format_html(result, title=pred.stem)
    else:
        content = format_markdown(result)
    print(content)
    if out:
        ensure_parent(out)
        out.write_text(content + "\n", encoding="utf-8")


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("--run-name cannot be empty after sanitization.")
    return cleaned


def resolve_metric_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    output_format = "json" if args.json else args.format
    if not args.run_name:
        if not args.pred:
            raise ValueError("--pred is required unless --run-name is provided.")
        return args.pred, args.out

    run_dir = Path("outputs") / safe_run_name(args.run_name)
    default_pred = run_dir / "predictions.judge.jsonl"
    if not default_pred.exists():
        default_pred = run_dir / "predictions.jsonl"
    pred = args.pred or default_pred
    suffix = {"markdown": "md", "json": "json", "html": "html"}[output_format]
    out = args.out or run_dir / f"metrics.{suffix}"
    return pred, out


def format_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def format_markdown(result: dict[str, Any]) -> str:
    rows = [
        ("num_predictions", result.get("num_predictions")),
        ("num_valid", result.get("num_valid")),
        ("accuracy", format_float(result.get("accuracy"))),
        ("evidence_session_recall", format_float(result.get("evidence_session_recall"))),
        ("evidence_session_all_found_rate", format_float(result.get("evidence_session_all_found_rate"))),
        ("f1", format_float(result.get("f1"))),
        ("bleu", format_float(result.get("bleu"))),
        ("build_tokens", result.get("build_tokens")),
        ("extraction_tokens", result.get("extraction_tokens")),
        ("embedding_build_tokens", result.get("embedding_build_tokens")),
        ("query_tokens", result.get("query_tokens")),
        ("build_tokens_per_valid", format_float(result.get("build_tokens_per_valid"))),
        ("query_tokens_per_valid", format_float(result.get("query_tokens_per_valid"))),
        ("total_tokens_per_valid", format_float(result.get("total_tokens_per_valid"))),
        ("build_time_seconds", format_float(result.get("build_time_seconds"))),
        ("query_time_seconds", format_float(result.get("query_time_seconds"))),
    ]
    lines = ["## Overall", "", "| metric | value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    by_type = result.get("by_type") or {}
    if by_type:
        lines.extend(
            [
                "",
                "## By Type",
                "",
                "| type | num_valid | accuracy | evidence recall | all found | f1 | bleu |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, values in by_type.items():
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    key,
                    values.get("num_valid"),
                    format_float(values.get("accuracy")),
                    format_float(values.get("evidence_session_recall")),
                    format_float(values.get("evidence_session_all_found_rate")),
                    format_float(values.get("f1")),
                    format_float(values.get("bleu")),
                )
            )
    return "\n".join(lines)


def format_html(result: dict[str, Any], *, title: str = "metrics") -> str:
    accuracy = result.get("accuracy")
    by_type = result.get("by_type") or {}
    max_count = max([values.get("num_valid") or 0 for values in by_type.values()] + [1])
    cards = [
        ("Accuracy", percent(accuracy), "Answer judge correctness"),
        ("Valid Samples", format_int(result.get("num_valid")), "Usable predictions"),
        ("Evidence Recall", percent(result.get("evidence_session_recall")), "Average gold-session recall"),
        ("All Evidence Found", percent(result.get("evidence_session_all_found_rate")), "Samples with full evidence coverage"),
        ("Build Tokens / Q", format_number(result.get("build_tokens_per_valid")), "LLM extract tokens only"),
        ("Query Tokens / Q", format_number(result.get("query_tokens_per_valid")), "Retrieval query + answer tokens"),
    ]
    card_html = "\n".join(
        f"""
        <article class="card">
          <div class="card-label">{escape(label)}</div>
          <div class="card-value">{escape(value)}</div>
          <div class="card-note">{escape(note)}</div>
        </article>"""
        for label, value, note in cards
    )
    overall_rows = [
        ("num_predictions", format_int(result.get("num_predictions"))),
        ("num_valid", format_int(result.get("num_valid"))),
        ("accuracy", percent(result.get("accuracy"))),
        ("evidence_session_recall", percent(result.get("evidence_session_recall"))),
        ("evidence_session_all_found_rate", percent(result.get("evidence_session_all_found_rate"))),
        ("f1", format_float(result.get("f1"))),
        ("bleu", format_float(result.get("bleu"))),
        ("build_tokens", format_int(result.get("build_tokens"))),
        ("extraction_tokens", format_int(result.get("extraction_tokens"))),
        ("embedding_build_tokens", format_int(result.get("embedding_build_tokens"))),
        ("query_tokens", format_int(result.get("query_tokens"))),
        ("build_tokens_per_valid", format_number(result.get("build_tokens_per_valid"))),
        ("query_tokens_per_valid", format_number(result.get("query_tokens_per_valid"))),
        ("total_tokens_per_valid", format_number(result.get("total_tokens_per_valid"))),
        ("build_time_seconds", format_number(result.get("build_time_seconds"))),
        ("query_time_seconds", format_number(result.get("query_time_seconds"))),
    ]
    overall_html = "\n".join(
        f"<tr><th>{escape(metric)}</th><td>{escape(value)}</td></tr>"
        for metric, value in overall_rows
    )
    type_html = "\n".join(
        format_type_row(key, values, max_count)
        for key, values in sorted(by_type.items())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} Metrics</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #e4e7ec;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --good: #059669;
      --warn: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
    }}
    .page {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 24px 56px;
    }}
    header {{
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .summary {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .card-value {{
      margin-top: 8px;
      font-size: 26px;
      font-weight: 700;
    }}
    .card-note {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    section {{
      margin-top: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .section-head {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    thead th {{
      color: var(--muted);
      background: #fbfcfe;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .03em;
    }}
    tbody tr:last-child th, tbody tr:last-child td {{
      border-bottom: 0;
    }}
    .metric-table {{
      min-width: 0;
    }}
    .metric-table th {{
      width: 45%;
      color: var(--muted);
      font-weight: 500;
    }}
    .metric-table td {{
      font-variant-numeric: tabular-nums;
    }}
    .bar-cell {{
      min-width: 150px;
    }}
    .bar {{
      height: 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      overflow: hidden;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .pill {{
      display: inline-block;
      min-width: 58px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #f2f4f7;
      color: #344054;
      font-variant-numeric: tabular-nums;
    }}
    .pill.good {{ background: #dcfce7; color: #166534; }}
    .pill.warn {{ background: #fef3c7; color: #92400e; }}
    @media (max-width: 760px) {{
      .page {{ padding: 28px 14px 40px; }}
      .cards {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <h1>{escape(title)} Metrics</h1>
      <p class="summary">Accuracy {escape(percent(accuracy))}; evidence recall {escape(percent(result.get("evidence_session_recall")))}; token cost counts LLM extraction and query tokens, excluding embedding tokens from total cost.</p>
    </header>
    <div class="cards">{card_html}
    </div>
    <section>
      <div class="section-head"><h2>By Question Type</h2></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Type</th>
              <th>Samples</th>
              <th>Share</th>
              <th>Accuracy</th>
              <th>Evidence Recall</th>
              <th>All Found</th>
              <th>F1</th>
              <th>BLEU</th>
            </tr>
          </thead>
          <tbody>
            {type_html}
          </tbody>
        </table>
      </div>
    </section>
    <section>
      <div class="section-head"><h2>Overall Details</h2></div>
      <div class="table-wrap">
        <table class="metric-table">
          <tbody>
            {overall_html}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>"""


def format_type_row(key: str, values: dict[str, Any], max_count: int) -> str:
    count = int(values.get("num_valid") or 0)
    width = max(4, round(count / max_count * 100)) if count else 0
    accuracy = values.get("accuracy")
    all_found = values.get("evidence_session_all_found_rate")
    accuracy_class = "good" if isinstance(accuracy, float) and accuracy >= 0.75 else "warn" if isinstance(accuracy, float) and accuracy < 0.5 else ""
    return f"""<tr>
      <th>{escape(key)}</th>
      <td>{format_int(count)}</td>
      <td class="bar-cell"><div class="bar" aria-label="{count} samples"><span style="width: {width}%"></span></div></td>
      <td><span class="pill {accuracy_class}">{escape(percent(accuracy))}</span></td>
      <td>{escape(percent(values.get("evidence_session_recall")))}</td>
      <td>{escape(percent(all_found))}</td>
      <td>{escape(format_float(values.get("f1")))}</td>
      <td>{escape(format_float(values.get("bleu")))}</td>
    </tr>"""


def escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def percent(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (float, int)):
        return f"{float(value) * 100:.1f}%"
    return str(value)


def format_int(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def format_number(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def format_float(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
