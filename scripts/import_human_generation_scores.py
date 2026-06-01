#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_score(value: Any, *, row_id: str, field: str) -> int:
    text = str(value or "").strip()
    if text == "":
        return -1
    try:
        score = int(text)
    except ValueError as exc:
        raise SystemExit(f"Invalid {field} for id={row_id}: expected -1 or integer 1..5, got {value!r}") from exc
    if score == -1 or 1 <= score <= 5:
        return score
    raise SystemExit(f"Invalid {field} for id={row_id}: expected -1 or integer 1..5, got {score}")


def load_csv_scores(path: Path) -> Dict[str, Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "human_precision", "human_completeness", "human_format", "human_comment"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                continue
            scores[row_id] = {
                "precision": parse_score(row.get("human_precision"), row_id=row_id, field="human_precision"),
                "completeness": parse_score(row.get("human_completeness"), row_id=row_id, field="human_completeness"),
                "format": parse_score(row.get("human_format"), row_id=row_id, field="human_format"),
                "comment": str(row.get("human_comment") or ""),
            }
    return scores


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import human generation scores from review CSV into JSONL.")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL not found: {jsonl_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    records = load_jsonl(jsonl_path)
    scores = load_csv_scores(csv_path)
    updated = 0
    for record in records:
        row_id = str(record.get("id") or "")
        if row_id not in scores:
            continue
        human = dict(record.get("human") or {})
        human.update(scores[row_id])
        record["human"] = human
        updated += 1
    write_jsonl(out_path, records)
    print(f"Updated {updated} records; wrote {out_path}")


if __name__ == "__main__":
    main()
