"""Aggregate evaluation result JSONs into a Markdown benchmark table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_COLUMNS = [
    ("model", "Model"),
    ("params", "Params"),
    ("PPL", "PPL"),
    ("QA acc", "QA acc"),
    ("distinct-1", "distinct-1"),
    ("distinct-2", "distinct-2"),
    ("repetition", "Repetition"),
    ("tokens_per_sec", "Throughput (tok/s)"),
]


def _read_results(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_markdown_table(results_dir: Path) -> str:
    rows: dict[str, dict[str, Any]] = {}
    for rec in _read_results(results_dir):
        model = str(rec.get("model") or rec.get("ckpt") or "unknown")
        row = rows.setdefault(model, {"model": model})
        if rec.get("task") == "qa_accuracy":
            row["QA acc"] = rec.get("accuracy")
        if rec.get("task") == "inference_benchmark":
            row["tokens_per_sec"] = rec.get("tokens_per_second")
            row["params"] = rec.get("param_count")
        if "PPL" in rec:  # lm_perplexity
            row["PPL"] = rec["PPL"]
        agg = rec.get("aggregate")
        if agg:
            for k in ("distinct-1", "distinct-2", "repetition", "tokens_per_sec"):
                if k in agg:
                    row[k] = agg[k]
        for k in (
            "QA acc",
            "distinct-1",
            "distinct-2",
            "repetition",
            "tokens_per_sec",
            "params",
        ):
            if k in rec and k not in row:
                row[k] = rec[k]

    if not rows:
        return "No evaluation results found in eval/results/."

    header = [label for _, label in _COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for model in sorted(rows):
        r = rows[model]
        lines.append("| " + " | ".join(_cell(r.get(key)) for key, _ in _COLUMNS) + " |")
    return "\n".join(lines)
