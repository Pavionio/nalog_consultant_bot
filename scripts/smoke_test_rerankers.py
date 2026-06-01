#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.core import RAGConfig
from src.rag.rerankers import auto_detect_reranker_type, build_reranker


DEFAULT_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "jinaai/jina-reranker-v2-base-multilingual",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
]

HEAVY_MODELS = [
    "Qwen/Qwen3-Reranker-8B",
    "BAAI/bge-reranker-v2-gemma",
    "BAAI/bge-reranker-v2.5-gemma2-lightweight",
]

QUERY = "какой срок подачи уведомления по НДФЛ при выплате дивидендов"
PASSAGES = [
    "Уведомление об исчисленных суммах НДФЛ подается налоговым агентом в установленный срок...",
    "Налог на имущество организаций рассчитывается исходя из кадастровой стоимости...",
    "Порядок применения патентной системы налогообложения индивидуальными предпринимателями...",
]


def run_one(model_name: str, reranker_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dataclasses.replace(
        RAGConfig(),
        use_reranker=True,
        reranker_model=model_name,
        reranker_type=reranker_type,
        reranker_max_length=args.reranker_max_length,
        reranker_batch_size=args.reranker_batch_size,
        reranker_device=args.reranker_device or RAGConfig().reranker_device,
    )
    actual_type = auto_detect_reranker_type(model_name) if reranker_type == "auto" else reranker_type
    try:
        reranker = build_reranker(cfg)
        scores = reranker.score(QUERY, PASSAGES)
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {
            "status": "ok",
            "reranker_model": model_name,
            "reranker_type": getattr(reranker, "reranker_type", actual_type),
            "scores": scores,
            "ranking": [i + 1 for i in ranking],
            "expected_top": 1,
            "passed": bool(ranking and ranking[0] == 0),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reranker_model": model_name,
            "reranker_type": actual_type,
            "error": str(exc).splitlines()[0],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--reranker-type", default="auto")
    ap.add_argument("--all-defaults", action="store_true", help="Test all default reranker experiment models")
    ap.add_argument("--include-heavy", action="store_true", help="Also test heavy optional rerankers")
    ap.add_argument("--reranker-max-length", type=int, default=1024)
    ap.add_argument("--reranker-batch-size", type=int, default=8)
    ap.add_argument("--reranker-device", default=None)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    models = DEFAULT_MODELS if args.all_defaults else [args.reranker_model]
    if args.all_defaults and args.include_heavy:
        models = models + HEAVY_MODELS

    if args.all_defaults and not args._child:
        failed = False
        for model in models:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--reranker-model",
                model,
                "--reranker-type",
                args.reranker_type,
                "--reranker-max-length",
                str(args.reranker_max_length),
                "--reranker-batch-size",
                str(args.reranker_batch_size),
                "--_child",
            ]
            if args.reranker_device:
                cmd.extend(["--reranker-device", args.reranker_device])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = (proc.stdout or "") + (proc.stderr or "")
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            failed = failed or proc.returncode != 0
        if failed:
            sys.exit(1)
        return

    results: List[Dict[str, Any]] = []
    for model in models:
        print(f"=== {model} ===", flush=True)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            result = run_one(model, args.reranker_type, args)
        captured_stdout = stdout_buf.getvalue()
        captured_stderr = stderr_buf.getvalue()
        if captured_stdout:
            print(captured_stdout, end="")
        if captured_stderr:
            print(captured_stderr, end="", file=sys.stderr)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.stdout.flush()

    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
