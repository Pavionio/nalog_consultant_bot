#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_generation_judges import compute_summary, print_summary


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_id(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id:
            raise SystemExit(f"{label}: record without id")
        if row_id in out:
            raise SystemExit(f"{label}: duplicate id={row_id}")
        out[row_id] = row
    return out


def merge_human_into_judged(
    judged_rows: List[Dict[str, Any]],
    human_rows: List[Dict[str, Any]],
    *,
    strict: bool,
) -> List[Dict[str, Any]]:
    human_by_id = by_id(human_rows, "human-jsonl")
    merged: List[Dict[str, Any]] = []
    missing_human: List[str] = []
    for judged in judged_rows:
        row_id = str(judged.get("id") or "")
        human_source = human_by_id.get(row_id)
        if human_source is None:
            missing_human.append(row_id)
            merged.append(dict(judged))
            continue
        row = dict(judged)
        row["human"] = dict(human_source.get("human") or {})
        merged.append(row)
    if strict and missing_human:
        raise SystemExit(f"Missing human records for ids: {', '.join(missing_human[:20])}")
    return merged


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare LLM judge scores with separately imported human generation scores."
    )
    ap.add_argument("--judged-jsonl", required=True, help="JSONL with judge field from evaluate_generation_judges.py")
    ap.add_argument("--human-jsonl", required=True, help="JSONL with human scores from import_human_generation_scores.py")
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--out", default=None, help="Optional merged JSONL with judge + human fields")
    ap.add_argument("--strict", action="store_true", help="Fail if any judged record has no matching human record")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    judged_path = Path(args.judged_jsonl)
    human_path = Path(args.human_jsonl)
    if not judged_path.exists():
        raise SystemExit(f"Judged JSONL not found: {judged_path}")
    if not human_path.exists():
        raise SystemExit(f"Human JSONL not found: {human_path}")

    judged_rows = load_jsonl(judged_path)
    human_rows = load_jsonl(human_path)
    merged = merge_human_into_judged(judged_rows, human_rows, strict=args.strict)

    if args.out:
        write_jsonl(Path(args.out), merged)

    summary = compute_summary(merged, str(judged_path))
    summary["human_input"] = str(human_path)
    if args.out:
        summary["merged_output"] = args.out
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\nSummary: {summary_path}")
    if args.out:
        print(f"Merged JSONL: {args.out}")


if __name__ == "__main__":
    main()
