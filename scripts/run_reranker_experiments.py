#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.eval.eval import evaluate_dataset
from src.rag.core import RAGConfig, Retriever, STEmbedder
from src.rag.rerankers import auto_detect_reranker_type


DEFAULT_DATASETS = [
    "data/eval/eval_easy_v2.jsonl",
    "data/eval/eval_hard_v2.jsonl",
    "data/eval/eval_superhard_v2.jsonl",
]

DEFAULT_RERANKERS = [
    "none",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "jinaai/jina-reranker-v2-base-multilingual",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
]

HEAVY_RERANKERS = [
    "Qwen/Qwen3-Reranker-8B",
    "BAAI/bge-reranker-v2-gemma",
    "BAAI/bge-reranker-v2.5-gemma2-lightweight",
]

DEFAULT_BATCH_SIZES = {
    "BAAI/bge-reranker-v2-m3": 16,
    "Qwen/Qwen3-Reranker-0.6B": 8,
    "Qwen/Qwen3-Reranker-4B": 4,
    "Qwen/Qwen3-Reranker-8B": 2,
    "jinaai/jina-reranker-v2-base-multilingual": 8,
    "mixedbread-ai/mxbai-rerank-base-v2": 8,
    "mixedbread-ai/mxbai-rerank-large-v2": 4,
    "Alibaba-NLP/gte-multilingual-reranker-base": 8,
    "cross-encoder/ms-marco-MiniLM-L6-v2": 32,
    "BAAI/bge-reranker-v2-gemma": 2,
    "BAAI/bge-reranker-v2.5-gemma2-lightweight": 2,
}


def experiment_reranker_type(model_name: str, args: argparse.Namespace) -> str:
    if auto_detect_reranker_type(model_name) == "mixedbread":
        return "mixedbread"
    if args.reranker_type:
        return args.reranker_type
    return "auto"


def short_model_name(model_name: str) -> str:
    if model_name == "none":
        return "none"
    name = model_name.split("/")[-1].lower()
    name = name.replace("reranker", "rr").replace("rerank", "rr")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_successes(log_path: Path) -> set[tuple[str, str, int, int]]:
    keys = set()
    for row in load_jsonl(log_path):
        if row.get("status") == "ok":
            keys.add((row.get("dataset"), row.get("reranker_model"), int(row.get("fetch_k") or 0), int(row.get("final_k") or 0)))
    return keys


def cleanup_accelerators() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def apply_allocator_env() -> None:
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    extra = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"
    if current:
        if "expandable_segments" not in current:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = current + "," + extra
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = extra
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_markdown_summary(log_path: Path, md_path: Path) -> None:
    latest: Dict[tuple[str, str, int, int], Dict[str, Any]] = {}
    for row in load_jsonl(log_path):
        if row.get("status") != "ok":
            continue
        key = (
            str(row.get("dataset", "")),
            str(row.get("reranker_model", "")),
            int(row.get("fetch_k") or 0),
            int(row.get("final_k") or 0),
        )
        latest[key] = row
    rows = list(latest.values())
    rows.sort(key=lambda r: (
        str(r.get("reranker_model", "")),
        str(r.get("dataset", "")),
        int(r.get("fetch_k") or 0),
        int(r.get("final_k") or 0),
    ))
    headers = [
        "dataset",
        "reranker_model",
        "reranker_type",
        "fetch_k",
        "final_k",
        "hit@k",
        "precision@k",
        "mrr@k",
        "ndcg@k",
        "hard_negative_rate@k",
        "dense_latency_avg_ms",
        "rerank_latency_avg_ms",
        "total_latency_p95_ms",
        "status",
    ]

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|")[:160]

    lines = [
        "# Reranker Experiments",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        row_k = int(row.get("final_k") or 0)
        values = [
            row.get("dataset"),
            row.get("reranker_model"),
            row.get("reranker_type"),
            row.get("fetch_k"),
            row.get("final_k"),
            row.get(f"recall@{row_k}") or row.get(f"hit@{row_k}"),
            row.get(f"precision@{row_k}"),
            row.get(f"mrr@{row_k}"),
            row.get(f"ndcg@{row_k}"),
            row.get(f"hard_negative_rate@{row_k}"),
            row.get("dense_latency_avg_ms"),
            row.get("rerank_latency_avg_ms"),
            row.get("latency_p95_ms"),
            row.get("status"),
        ]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    dataset: str,
    model_name: str,
    fetch_k: int,
    k: int,
    args: argparse.Namespace,
    embedder: Any,
    retriever: Any = None,
) -> Dict[str, Any]:
    dataset_stem = Path(dataset).stem
    short_name = short_model_name(model_name)
    method = "baseline" if model_name == "none" else f"reranker_{short_name}_fk{fetch_k}"
    report_path = Path(args.output_dir) / f"report_reranker_{dataset_stem}_{short_name}_fk{fetch_k}_k{k}.json"
    use_reranker = model_name != "none"
    reranker_type = None if not use_reranker else auto_detect_reranker_type(model_name)
    requested_reranker_type = experiment_reranker_type(model_name, args) if use_reranker else "auto"

    if args.dry_run:
        return {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "reranker_model": model_name,
            "reranker_type": reranker_type,
            "fetch_k": fetch_k,
            "final_k": k,
            "method": method,
            "status": "dry_run",
            "report_path": str(report_path),
        }

    cfg = RAGConfig()
    cfg = dataclasses.replace(
        cfg,
        use_reranker=use_reranker,
        reranker_model=model_name if use_reranker else cfg.reranker_model,
        reranker_type=requested_reranker_type,
        reranker_fetch_k=fetch_k,
        reranker_top_k=fetch_k,
        reranker_max_length=args.reranker_max_length or cfg.reranker_max_length,
        reranker_batch_size=args.reranker_batch_size or DEFAULT_BATCH_SIZES.get(model_name, cfg.reranker_batch_size),
        reranker_device=args.reranker_device or cfg.reranker_device,
    )
    own_retriever = retriever is None
    if retriever is None:
        retriever = Retriever(cfg, embedder)
    report = None
    try:
        report = evaluate_dataset(
            dataset,
            k=k,
            use_llm_judge=False,
            use_reranker=use_reranker,
            reranker_model=model_name if use_reranker else None,
            reranker_type=requested_reranker_type,
            reranker_fetch_k=fetch_k,
            reranker_max_length=cfg.reranker_max_length,
            reranker_batch_size=cfg.reranker_batch_size,
            reranker_device=cfg.reranker_device,
            embedder=embedder,
            retriever=retriever,
            method=method,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        metrics = report["metrics"]
        return {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "dataset_stem": dataset_stem,
            "method": method,
            "reranker_model": model_name,
            "reranker_type": report.get("meta", {}).get("reranker_type") if use_reranker else "none",
            "fetch_k": fetch_k,
            "final_k": k,
            "report_path": str(report_path),
            "status": "ok",
            f"recall@{k}": metrics.get(f"recall@{k}"),
            f"precision@{k}": metrics.get(f"precision@{k}"),
            f"mrr@{k}": metrics.get(f"mrr@{k}"),
            f"ndcg@{k}": metrics.get(f"ndcg@{k}"),
            f"hard_negative_rate@{k}": metrics.get(f"hard_negative_rate@{k}"),
            "dense_latency_avg_ms": metrics.get("dense_latency_avg_ms"),
            "rerank_latency_avg_ms": metrics.get("rerank_latency_avg_ms"),
            "latency_p95_ms": metrics.get("latency_p95_ms"),
        }
    except Exception as exc:
        error = str(exc).splitlines()[0] if str(exc).splitlines() else repr(exc)
        error_type = type(exc).__name__
        error_message = str(exc)
        error_traceback = traceback.format_exc()
        failure_report = {
            "meta": {
                "dataset_path": dataset,
                "dataset_stem": dataset_stem,
                "method": method,
                "reranker_model": model_name,
                "reranker_type": reranker_type,
                "reranker_fetch_k": fetch_k,
                "final_k": k,
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "status": "failed",
            "error": error,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": error_traceback,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(failure_report, f, ensure_ascii=False, indent=2)
        return {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "dataset_stem": dataset_stem,
            "method": method,
            "reranker_model": model_name,
            "reranker_type": reranker_type,
            "fetch_k": fetch_k,
            "final_k": k,
            "report_path": str(report_path),
            "status": "failed",
            "error": error,
            "error_type": error_type,
        }
    finally:
        if own_retriever:
            try:
                if getattr(retriever, "_reranker", None) is not None:
                    del retriever._reranker
            except Exception:
                pass
            try:
                del retriever
            except Exception:
                pass
        try:
            del report
        except Exception:
            pass
        if own_retriever:
            cleanup_accelerators()


def run_experiment_isolated(dataset: str, model_name: str, fetch_k: int, k: int, args: argparse.Namespace) -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    result_json = Path(args.output_dir) / "tmp" / (
        f"{Path(dataset).stem}_{short_model_name(model_name)}_fk{fetch_k}_k{k}_{os.getpid()}.json"
    )
    result_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--dataset",
        dataset,
        "--reranker-model",
        model_name,
        "--fetch-k",
        str(fetch_k),
        "--k",
        str(k),
        "--output-dir",
        args.output_dir,
        "--result-json",
        str(result_json),
    ]
    if args.reranker_device:
        cmd.extend(["--reranker-device", args.reranker_device])
    if args.reranker_type:
        cmd.extend(["--reranker-type", args.reranker_type])
    if args.reranker_batch_size is not None:
        cmd.extend(["--reranker-batch-size", str(args.reranker_batch_size)])
    if args.reranker_max_length is not None:
        cmd.extend(["--reranker-max-length", str(args.reranker_max_length)])
    cmd.append("--no-judge")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128",
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    proc = subprocess.run(cmd, env=env)
    if result_json.exists():
        with open(result_json, encoding="utf-8") as f:
            row = json.load(f)
        try:
            result_json.unlink()
        except Exception:
            pass
        return row

    return {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "dataset_stem": Path(dataset).stem,
        "method": "baseline" if model_name == "none" else f"reranker_{short_model_name(model_name)}_fk{fetch_k}",
        "reranker_model": model_name,
        "reranker_type": None if model_name == "none" else auto_detect_reranker_type(model_name),
        "fetch_k": fetch_k,
        "final_k": k,
        "status": "failed",
        "error": f"worker exited with code {proc.returncode}",
    }


def run_experiment_group_isolated(jobs: List[tuple[str, str, int, int]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    script_path = Path(__file__).resolve()
    model_name = jobs[0][1]
    tmp_dir = Path(args.output_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jobs_json = tmp_dir / f"jobs_{short_model_name(model_name)}_{os.getpid()}.json"
    result_json = tmp_dir / f"result_{short_model_name(model_name)}_{os.getpid()}.json"

    with open(jobs_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"dataset": dataset, "reranker_model": model, "fetch_k": fetch_k, "k": k}
                for dataset, model, fetch_k, k in jobs
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--worker-group",
        "--jobs-json",
        str(jobs_json),
        "--output-dir",
        args.output_dir,
        "--result-json",
        str(result_json),
        "--no-judge",
    ]
    if args.reranker_device:
        cmd.extend(["--reranker-device", args.reranker_device])
    if args.reranker_type:
        cmd.extend(["--reranker-type", args.reranker_type])
    if args.reranker_batch_size is not None:
        cmd.extend(["--reranker-batch-size", str(args.reranker_batch_size)])
    if args.reranker_max_length is not None:
        cmd.extend(["--reranker-max-length", str(args.reranker_max_length)])

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128",
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    proc = subprocess.run(cmd, env=env)
    try:
        jobs_json.unlink()
    except Exception:
        pass

    if result_json.exists():
        with open(result_json, encoding="utf-8") as f:
            rows = json.load(f)
        try:
            result_json.unlink()
        except Exception:
            pass
        return rows

    return [
        {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "dataset_stem": Path(dataset).stem,
            "method": "baseline" if model == "none" else f"reranker_{short_model_name(model)}_fk{fetch_k}",
            "reranker_model": model,
            "reranker_type": None if model == "none" else auto_detect_reranker_type(model),
            "fetch_k": fetch_k,
            "final_k": k,
            "status": "failed",
            "error": f"worker group exited with code {proc.returncode}",
        }
        for dataset, model, fetch_k, k in jobs
    ]


def run_worker_group(jobs_path: str, result_path: str, args: argparse.Namespace) -> None:
    jobs_raw = load_jsonl(Path(jobs_path)) if jobs_path.endswith(".jsonl") else json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    jobs = [
        (str(item["dataset"]), str(item["reranker_model"]), int(item["fetch_k"]), int(item["k"]))
        for item in jobs_raw
    ]
    rows: List[Dict[str, Any]] = []
    embedder = STEmbedder(RAGConfig().embed_model_name)
    first_dataset, first_model, first_fetch_k, first_k = jobs[0]
    cfg = dataclasses.replace(
        RAGConfig(),
        use_reranker=first_model != "none",
        reranker_model=first_model if first_model != "none" else RAGConfig().reranker_model,
        reranker_type=experiment_reranker_type(first_model, args),
        reranker_fetch_k=first_fetch_k,
        reranker_top_k=first_fetch_k,
        reranker_max_length=args.reranker_max_length or RAGConfig().reranker_max_length,
        reranker_batch_size=args.reranker_batch_size or DEFAULT_BATCH_SIZES.get(first_model, RAGConfig().reranker_batch_size),
        reranker_device=args.reranker_device or RAGConfig().reranker_device,
    )
    retriever = Retriever(cfg, embedder)
    model_failed = False

    try:
        for dataset, model_name, fetch_k, k in jobs:
            if model_failed and model_name != "none":
                rows.append({
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": dataset,
                    "dataset_stem": Path(dataset).stem,
                    "method": f"reranker_{short_model_name(model_name)}_fk{fetch_k}",
                    "reranker_model": model_name,
                    "reranker_type": auto_detect_reranker_type(model_name),
                    "fetch_k": fetch_k,
                    "final_k": k,
                    "status": "failed",
                    "error": "Skipped because this reranker model failed earlier in the same worker.",
                })
                continue
            row = run_experiment(dataset, model_name, fetch_k, k, args, embedder, retriever=retriever)
            rows.append(row)
            if row.get("status") == "failed" and model_name != "none":
                model_failed = True
    finally:
        try:
            if getattr(retriever, "_reranker", None) is not None:
                del retriever._reranker
        except Exception:
            pass
        try:
            del retriever
            del embedder
        except Exception:
            pass
        cleanup_accelerators()

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-group", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--jobs-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--reranker-model", default=None)
    ap.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--rerankers", nargs="+", default=None)
    ap.add_argument("--fetch-k", nargs="+", type=int, default=[10, 20, 50, 100])
    ap.add_argument("--k", nargs="+", type=int, default=[5])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default="data/metrics")
    ap.add_argument("--no-heavy", action="store_true")
    ap.add_argument("--include-heavy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="Rebuild markdown summary from the existing JSONL log and exit.")
    ap.add_argument("--reranker-device", default=None)
    ap.add_argument("--reranker-type", default=None)
    ap.add_argument("--reranker-batch-size", type=int, default=None)
    ap.add_argument("--reranker-max-length", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    apply_allocator_env()

    if args.worker:
        if not args.dataset or not args.reranker_model or not args.result_json:
            raise SystemExit("--worker requires --dataset, --reranker-model, and --result-json")
        embedder = STEmbedder(RAGConfig().embed_model_name)
        row = run_experiment(args.dataset, args.reranker_model, args.fetch_k[0], args.k[0], args, embedder)
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
        return

    if args.worker_group:
        if not args.jobs_json or not args.result_json:
            raise SystemExit("--worker-group requires --jobs-json and --result-json")
        run_worker_group(args.jobs_json, args.result_json, args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "reranker_experiments.jsonl"
    md_path = output_dir / "reranker_experiments.md"

    if args.summary_only:
        write_markdown_summary(log_path, md_path)
        print(f"Saved summary: {md_path}")
        return

    rerankers = args.rerankers or list(DEFAULT_RERANKERS)
    if args.include_heavy:
        rerankers.extend([m for m in HEAVY_RERANKERS if m not in rerankers])
    elif args.no_heavy:
        rerankers = [m for m in rerankers if m not in HEAVY_RERANKERS]

    datasets = [d for d in args.datasets if Path(d).exists()]
    missing = [d for d in args.datasets if not Path(d).exists()]
    for path in missing:
        print(f"[skip] dataset not found: {path}")

    jobs = [
        (dataset, model_name, fetch_k, k)
        for model_name in rerankers
        for dataset in datasets
        for k in args.k
        for fetch_k in ([k] if model_name == "none" else args.fetch_k)
    ]
    if args.limit is not None:
        jobs = jobs[:args.limit]

    done = existing_successes(log_path) if args.skip_existing and not args.force else set()
    pending_jobs = [job for job in jobs if job not in done]

    if args.dry_run:
        for i, (dataset, model_name, fetch_k, k) in enumerate(jobs, start=1):
            if (dataset, model_name, fetch_k, k) in done:
                print(f"[{i}/{len(jobs)}] skip existing {Path(dataset).name} {model_name} fk{fetch_k}")
                continue
            print(f"[{i}/{len(jobs)}] {Path(dataset).name} {model_name} fk{fetch_k}")
            row = run_experiment(dataset, model_name, fetch_k, k, args, None)
            print(f"  status={row.get('status')} ndcg@{k}={row.get(f'ndcg@{k}')}")
        write_markdown_summary(log_path, md_path)
        print(f"Saved summary: {md_path}")
        return

    groups: List[List[tuple[str, str, int, int]]] = []
    current_group: List[tuple[str, str, int, int]] = []
    current_model: str | None = None
    for job in pending_jobs:
        model_name = job[1]
        if current_group and model_name != current_model:
            groups.append(current_group)
            current_group = []
        current_group.append(job)
        current_model = model_name
    if current_group:
        groups.append(current_group)

    for group_i, group in enumerate(groups, start=1):
        model_name = group[0][1]
        print(f"[model {group_i}/{len(groups)}] {model_name}: {len(group)} run(s)")
        rows = run_experiment_group_isolated(group, args)
        for row in rows:
            append_jsonl(log_path, row)
            dataset = row.get("dataset", "")
            fetch_k = row.get("fetch_k")
            k = row.get("final_k")
            status = row.get("status")
            if status == "failed":
                print(
                    f"  {Path(dataset).name} fk{fetch_k}: failed: "
                    f"{row.get('error_type') or 'Error'}: {row.get('error')} "
                    f"(report: {row.get('report_path')})"
                )
            else:
                print(f"  {Path(dataset).name} fk{fetch_k}: status={status} ndcg@{k}={row.get(f'ndcg@{k}')}")
        write_markdown_summary(log_path, md_path)

    write_markdown_summary(log_path, md_path)
    print(f"Saved log: {log_path}")
    print(f"Saved summary: {md_path}")


if __name__ == "__main__":
    main()
