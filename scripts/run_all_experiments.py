"""
Run every configured RAG evaluation method on every eval dataset.

This is a thin launcher over scripts/run_eval_matrix.py, so the actual
experiment logic stays in one place.

Examples:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --k 5 --skip-precompute
    python scripts/run_all_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_DATASETS = [
    "eval_dataset.jsonl",
    "eval_hard_dataset.jsonl",
    "eval_superhard_dataset.jsonl",
]

ALL_METHODS = [
    "baseline",
    "rewrite",
    "hyde",
    "reranker",
    "hyde+reranker",
]


def default_python() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def discover_datasets() -> List[str]:
    ordered = [p for p in DEFAULT_DATASETS if Path(p).exists()]
    seen = set(ordered)
    extra = sorted(
        str(p)
        for p in Path(".").glob("eval*_dataset.jsonl")
        if str(p) not in seen
    )
    return ordered + extra


def quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all RAG experiment variants for all eval datasets")
    ap.add_argument("--datasets", nargs="+", default=None, help="Override dataset list")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], help="K values to evaluate")
    ap.add_argument("--out-dir", default="data/metrics")
    ap.add_argument("--log", default="data/metrics/eval_log.jsonl")
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:8080"))
    ap.add_argument("--llm-model", default=os.getenv("LLM", "Qwen/Qwen3-8B-GGUF:q5_k_m"))
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--skip-precompute", action="store_true", help="Reuse cached rewrite/HyDE queries if present")
    ap.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    ap.add_argument("--python", default=default_python(), help="Python executable for scripts/run_eval_matrix.py")
    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    missing = [d for d in datasets if not Path(d).exists()]
    if missing:
        raise SystemExit(f"Dataset files not found: {', '.join(missing)}")
    if not datasets:
        raise SystemExit("No eval datasets found. Expected files like eval_dataset.jsonl.")

    cmd = [
        args.python,
        "scripts/run_eval_matrix.py",
        "--k",
        *[str(k) for k in args.k],
        "--datasets",
        *datasets,
        "--methods",
        *ALL_METHODS,
        "--out-dir",
        args.out_dir,
        "--log",
        args.log,
        "--llm-base-url",
        args.llm_base_url,
        "--llm-model",
        args.llm_model,
        "--qdrant-url",
        args.qdrant_url,
    ]
    if args.skip_precompute:
        cmd.append("--skip-precompute")

    print("Running all experiments:")
    print(f"  datasets: {', '.join(datasets)}")
    print(f"  methods: {', '.join(ALL_METHODS)}")
    print(f"  k: {', '.join(str(k) for k in args.k)}")
    print(f"  command: {quote_cmd(cmd)}")

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
