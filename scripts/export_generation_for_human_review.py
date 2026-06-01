#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))


CSV_COLUMNS = [
    "id",
    "query",
    "answer",
    "gold_relevant_context_preview",
    "hard_negative_context_preview",
    "unknown_context_preview",
    "gold_relevant_ranks",
    "has_gold_relevant_context",
    "source_1",
    "source_2",
    "source_3",
    "human_precision",
    "human_completeness",
    "human_format",
    "human_comment",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preview(chunks: Iterable[Dict[str, Any]], label: str, limit: int = 1200) -> str:
    parts: List[str] = []
    for chunk in chunks:
        if chunk.get("relevance_label") != label:
            continue
        text = str(chunk.get("text") or "").strip().replace("\r\n", "\n")
        if len(text) > 500:
            text = text[:500].rstrip() + "..."
        parts.append(f"[{chunk.get('rank')}] {text}")
    out = "\n\n".join(parts)
    if len(out) > limit:
        return out[:limit].rstrip() + "..."
    return out


def source_value(chunk: Dict[str, Any]) -> str:
    parts = [
        f"rank={chunk.get('rank')}",
        f"source_code={chunk.get('source_code')}",
        f"external_id={chunk.get('external_id')}",
        f"chunk_i={chunk.get('chunk_i')}",
    ]
    url = chunk.get("canonical_url")
    if url:
        parts.append(f"url={url}")
    return "; ".join(parts)


def row_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    retrieved = record.get("retrieved") or []
    summary = record.get("retrieval_gold_summary") or {}
    human = record.get("human") or {}
    sources = [source_value(x) for x in retrieved[:3]]
    while len(sources) < 3:
        sources.append("")
    return {
        "id": record.get("id", ""),
        "query": record.get("query", ""),
        "answer": record.get("answer", ""),
        "gold_relevant_context_preview": preview(retrieved, "gold_relevant"),
        "hard_negative_context_preview": preview(retrieved, "hard_negative"),
        "unknown_context_preview": preview(retrieved, "unknown"),
        "gold_relevant_ranks": ",".join(str(x) for x in summary.get("gold_relevant_ranks") or []),
        "has_gold_relevant_context": summary.get("has_gold_relevant_context", ""),
        "source_1": sources[0],
        "source_2": sources[1],
        "source_3": sources[2],
        "human_precision": human.get("precision", -1),
        "human_completeness": human.get("completeness", -1),
        "human_format": human.get("format", -1),
        "human_comment": human.get("comment", ""),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export generation eval JSONL to CSV for human review.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    rows = load_jsonl(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in rows:
            writer.writerow(row_for_record(record))
    print(f"Exported {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()
