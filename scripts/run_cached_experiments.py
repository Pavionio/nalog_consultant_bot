"""
Run all eval experiments from precomputed rewrite/HyDE cache.

Use this after scripts/precompute_eval_queries.py finishes and after the LLM
server has been stopped. This script refuses to run if the cache file is absent.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

PRECOMPUTE_CACHE = "data/metrics/precomputed_queries.json"

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
    ap = argparse.ArgumentParser(description="Run all RAG experiments using cached rewrite/HyDE queries")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--out-dir", default="data/metrics")
    ap.add_argument("--log", default="data/metrics/eval_log.jsonl")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--python", default=default_python(), help="Python executable for scripts/run_eval_matrix.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    missing = [d for d in datasets if not Path(d).exists()]
    if missing:
        raise SystemExit(f"Dataset files not found: {', '.join(missing)}")
    if not datasets:
        raise SystemExit("No eval datasets found. Expected files like eval_dataset.jsonl.")

    cache_path = Path(args.out_dir) / Path(PRECOMPUTE_CACHE).name
    cache_exists = cache_path.exists()
    if not cache_exists and not args.dry_run:
        raise SystemExit(
            f"Precompute cache not found: {cache_path}. "
            "Run: python scripts/precompute_eval_queries.py"
        )

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
        "--qdrant-url",
        args.qdrant_url,
        "--skip-precompute",
    ]

    print("Running cached experiments:")
    print(f"  datasets: {', '.join(datasets)}")
    print(f"  methods: {', '.join(ALL_METHODS)}")
    print(f"  k: {', '.join(str(k) for k in args.k)}")
    print(f"  cache: {cache_path}")
    if not cache_exists:
        print("  cache status: missing")
    print(f"  command: {quote_cmd(cmd)}")

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
