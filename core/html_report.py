from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any

from core.io import load_jsonl


def write_predictions_html(jsonl_path: str | Path, html_path: str | Path | None = None, *, title: str = "") -> Path:
    source = Path(jsonl_path)
    target = Path(html_path) if html_path else source.with_suffix(".html")
    rows = load_jsonl(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_predictions_html(rows, title=title or source.name), encoding="utf-8")
    return target


def render_predictions_html(rows: list[dict[str, Any]], *, title: str) -> str:
    body = "\n".join(render_record(index, row) for index, row in enumerate(rows, start=1))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647085;
      --line: #dfe4ee;
      --accent: #2f6fed;
      --soft: #edf3ff;
      --bad: #b42318;
      --good: #067647;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 64px;
    }}
    header {{
      margin-bottom: 20px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .summary {{
      color: var(--muted);
      font-size: 14px;
    }}
    .record {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 14px 0;
      box-shadow: 0 1px 2px rgba(20, 32, 50, 0.04);
    }}
    .record-head {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .idx {{
      font-weight: 700;
      color: var(--accent);
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      background: var(--soft);
      color: #214b9a;
      padding: 3px 9px;
      font-size: 12px;
      border: 1px solid #d6e4ff;
    }}
    .chip.good {{ background: #ecfdf3; color: var(--good); border-color: #abefc6; }}
    .chip.bad {{ background: #fef3f2; color: var(--bad); border-color: #fecdca; }}
    .chip.warn {{ background: #fffaeb; color: var(--warn); border-color: #fedf89; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .field {{
      border-top: 1px solid var(--line);
      padding-top: 11px;
      min-width: 0;
    }}
    .field.full {{ grid-column: 1 / -1; }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }}
    .value {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      background: #f6f8fb;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .field.full {{ grid-column: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escape(title)}</h1>
      <div class="summary">{len(rows)} questions. Compact prediction report.</div>
    </header>
    {body}
  </main>
</body>
</html>
"""


def render_record(index: int, row: dict[str, Any]) -> str:
    judge = str(row.get("judge_label") or "").strip().upper()
    class_name = "good" if judge in {"CORRECT", "YES"} else "bad" if judge else ""
    judge_text = judge or "UNJUDGED"

    return f"""
    <section class="record">
      <div class="record-head">
        <span class="idx">#{index}</span>
        <span class="chip">{escape(str(row.get("question_type", "") or "unknown type"))}</span>
        <span class="chip {class_name}">{escape(judge_text)}</span>
      </div>
      <div class="grid">
        {field("Question", row.get("question"), full=True)}
        {field("Model Answer", row.get("hypothesis"))}
        {field("Gold Answer", format_jsonish(row.get("answer")))}
        {field("Retrieved Memory", format_retrieved_memory(row.get("answer_prompt")), full=True, pre=True)}
      </div>
    </section>
"""


def field(label: str, value: Any, *, full: bool = False, pre: bool = False) -> str:
    class_name = "field full" if full else "field"
    content = escape("" if value is None else str(value))
    value_html = f"<pre>{content}</pre>" if pre else f'<div class="value">{content}</div>'
    return f"""
        <div class="{class_name}">
          <div class="label">{escape(label)}</div>
          {value_html}
        </div>
"""


def format_retrieved_memory(value: Any) -> str:
    prompt = prompt_content(value)
    if not prompt:
        return ""
    match = re.search(
        r"Relevant Context:\s*(.*?)(?:\n\nRequirements:|\nRequirements:)",
        prompt,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"Context:\s*(.*?)(?:\n\nQuestion:|\nQuestion:)", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def prompt_content(value: Any) -> str:
    if not isinstance(value, list):
        return format_jsonish(value)
    contents = []
    for message in value:
        if isinstance(message, dict):
            contents.append(str(message.get("content", "")))
        else:
            contents.append(str(message))
    return "\n\n".join(contents)


def format_jsonish(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return "" if value is None else str(value)
