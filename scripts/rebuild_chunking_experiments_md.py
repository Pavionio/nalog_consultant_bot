#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "metrics" / "chunking_experiments.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "metrics" / "chunking_experiments.md"
NONZERO_METRIC_KEYS = ("hit_rate", "doc_hit", "overlap_hit", "precision", "mrr", "ndcg")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_nonzero_metrics(row: dict[str, Any]) -> bool:
    for key in NONZERO_METRIC_KEYS:
        value = _to_float(row.get(key))
        if value is not None and value != 0.0:
            return True
    return False


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _build_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "| time | variant | dataset | method | match | k | Hit@k | Doc@k | Overlap@k | Precision@k | MRR@k | nDCG@k | dense ms | rerank ms | p95 ms | chunk workers | embed batch | upsert batch | points | avg chars | status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines = header[:]
    for row in rows:
        lines.append(
            "| {timestamp} | {variant} | {dataset} | {method} | {match_mode} | {k} | {hit_rate} | {doc_hit} | {overlap_hit} | {precision} | {mrr} | {ndcg} | {dense_latency_avg_ms} | {rerank_latency_avg_ms} | {total_latency_p95_ms} | {chunk_workers} | {embed_batch_size} | {upsert_batch_size} | {points_count} | {avg_chunk_chars} | {status} |".format(
                **{
                    key: _fmt(row.get(key))
                    for key in [
                        "timestamp",
                        "variant",
                        "dataset",
                        "method",
                        "match_mode",
                        "k",
                        "hit_rate",
                        "doc_hit",
                        "overlap_hit",
                        "precision",
                        "mrr",
                        "ndcg",
                        "dense_latency_avg_ms",
                        "rerank_latency_avg_ms",
                        "total_latency_p95_ms",
                        "chunk_workers",
                        "embed_batch_size",
                        "upsert_batch_size",
                        "points_count",
                        "avg_chunk_chars",
                        "status",
                    ]
                }
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild chunking_experiments.md with only status=ok rows and non-zero metrics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    rows = [row for row in rows if row.get("status") == "ok" and _has_nonzero_metrics(row)]
    rows.sort(key=lambda row: (str(row.get("variant") or ""), str(row.get("dataset") or ""), str(row.get("timestamp") or "")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_build_table(rows), encoding="utf-8")
    print(f"written {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
