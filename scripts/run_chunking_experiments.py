#!/usr/bin/env python3
from __future__ import annotations

"""
Run Chonkie chunking experiments over local corpus and eval datasets.

Smoke eval:
python -m src.eval.eval \
  --dataset data/eval/eval_hard_v2.jsonl \
  --k 5 \
  --no-judge \
  --qdrant-collection rag_chunks_bge_m3_token_1024_128_smoke \
  --model chunk_token_1024_smoke

Smoke parent-child eval:
python -m src.eval.eval \
  --dataset data/eval/eval_hard_v2.jsonl \
  --k 5 \
  --no-judge \
  --qdrant-collection rag_chunks_bge_m3_parent3072_child768_smoke \
  --reranker \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --reranker-fetch-k 50 \
  --model parent_child_smoke

Stage 1 full chunking test on hard dataset:
python scripts/run_chunking_experiments.py \
  --stage hard_only \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --datasets data/eval/eval_hard_v2.jsonl \
  --methods baseline reranker \
  --k 5 \
  --embed-model BAAI/bge-m3 \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --reranker-fetch-k 50 \
  --skip-existing

Fast chunking experiment without reranker:
uv run python scripts/run_chunking_experiments.py \
  --stage hard_only \
  --datasets eval_hard_dataset.jsonl \
  --methods baseline \
  --k 5 \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --embed-model BAAI/bge-m3 \
  --embed-device cuda \
  --chunk-workers 8 \
  --embed-batch-size 64 \
  --upsert-batch-size 256 \
  --match-mode doc_overlap \
  --skip-existing
"""

import argparse
import datetime as dt
import dataclasses
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
METRICS_DIR = ROOT / "data" / "metrics"
SUMMARY_JSONL = METRICS_DIR / "chunking_experiments.jsonl"
SUMMARY_MD = METRICS_DIR / "chunking_experiments.md"


DEFAULT_DATASETS = [
    "data/eval/eval_easy_v2.jsonl",
    "data/eval/eval_hard_v2.jsonl",
    "data/eval/eval_superhard_v2.jsonl",
]
FALLBACK_DATASETS = {
    "data/eval/eval_easy_v2.jsonl": "eval_dataset.jsonl",
    "data/eval/eval_hard_v2.jsonl": "eval_hard_dataset.jsonl",
    "data/eval/eval_superhard_v2.jsonl": "eval_superhard_dataset.jsonl",
}


VARIANTS: dict[str, dict[str, Any]] = {
    "token_512_64": {"chunk_method": "token", "chunk_size": 512, "chunk_overlap": 64, "collection": "rag_chunks_bge_m3_token_512_64"},
    "token_1024_64": {"chunk_method": "token", "chunk_size": 1024, "chunk_overlap": 64, "collection": "rag_chunks_bge_m3_token_1024_64"},
    "token_1024_128": {"chunk_method": "token", "chunk_size": 1024, "chunk_overlap": 128, "collection": "rag_chunks_bge_m3_token_1024_128"},
    "token_1024_256": {"chunk_method": "token", "chunk_size": 1024, "chunk_overlap": 256, "collection": "rag_chunks_bge_m3_token_1024_256"},
    "token_1536_192": {"chunk_method": "token", "chunk_size": 1536, "chunk_overlap": 192, "collection": "rag_chunks_bge_m3_token_1536_192"},
    "token_2048_256": {"chunk_method": "token", "chunk_size": 2048, "chunk_overlap": 256, "collection": "rag_chunks_bge_m3_token_2048_256"},
    "sentence_1024_128": {"chunk_method": "sentence", "chunk_size": 1024, "chunk_overlap": 128, "chunk_min_sentences": 2, "collection": "rag_chunks_bge_m3_sentence_1024_128"},
    "sentence_1536_192": {"chunk_method": "sentence", "chunk_size": 1536, "chunk_overlap": 192, "chunk_min_sentences": 2, "collection": "rag_chunks_bge_m3_sentence_1536_192"},
    "recursive_1024": {"chunk_method": "recursive", "chunk_size": 1024, "chunk_overlap": 128, "collection": "rag_chunks_bge_m3_recursive_1024"},
    "recursive_1536": {"chunk_method": "recursive", "chunk_size": 1536, "chunk_overlap": 192, "collection": "rag_chunks_bge_m3_recursive_1536"},
    "recursive_legal_1536": {"chunk_method": "recursive_legal", "chunk_size": 1536, "chunk_overlap": 192, "collection": "rag_chunks_bge_m3_recursive_legal_1536"},
    "semantic_1024_t08": {"chunk_method": "semantic", "chunk_size": 1024, "chunk_overlap": 0, "semantic_threshold": 0.8, "semantic_similarity_window": 3, "semantic_skip_window": 0, "collection": "rag_chunks_bge_m3_semantic_1024_t08"},
    "semantic_1536_t075_skip1": {"chunk_method": "semantic", "chunk_size": 1536, "chunk_overlap": 0, "semantic_threshold": 0.75, "semantic_similarity_window": 3, "semantic_skip_window": 1, "collection": "rag_chunks_bge_m3_semantic_1536_t075_skip1"},
    "parent_2048_child_512": {"chunk_method": "parent_child", "parent_chunk_size": 2048, "parent_chunk_overlap": 192, "child_chunk_size": 512, "child_chunk_overlap": 64, "parent_chunker_method": "recursive_legal", "child_chunker_method": "sentence", "collection": "rag_chunks_bge_m3_parent2048_child512"},
    "parent_3072_child_768": {"chunk_method": "parent_child", "parent_chunk_size": 3072, "parent_chunk_overlap": 256, "child_chunk_size": 768, "child_chunk_overlap": 96, "parent_chunker_method": "recursive_legal", "child_chunker_method": "sentence", "collection": "rag_chunks_bge_m3_parent3072_child768"},
    "parent_4096_child_1024": {"chunk_method": "parent_child", "parent_chunk_size": 4096, "parent_chunk_overlap": 384, "child_chunk_size": 1024, "child_chunk_overlap": 128, "parent_chunker_method": "recursive_legal", "child_chunker_method": "sentence", "collection": "rag_chunks_bge_m3_parent4096_child1024"},
}


def _arg_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def _resolve_datasets(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        p = ROOT / value
        if p.exists():
            out.append(value)
            continue
        fallback = FALLBACK_DATASETS.get(value)
        if fallback and (ROOT / fallback).exists():
            out.append(fallback)
    return out


def _load_top3() -> list[str]:
    if not SUMMARY_JSONL.exists():
        return list(VARIANTS)[:3]
    rows = []
    with SUMMARY_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok":
                rows.append(row)
    rows.sort(key=lambda r: (r.get("mrr") or 0.0, r.get("ndcg") or 0.0), reverse=True)
    names = []
    for row in rows:
        if row.get("variant") not in names:
            names.append(row["variant"])
        if len(names) == 3:
            break
    return names or list(VARIANTS)[:3]


def _append_jsonl(row: dict[str, Any]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown() -> None:
    if not SUMMARY_JSONL.exists():
        return
    rows = []
    with SUMMARY_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows = rows[-500:]
    header = [
        "| time | variant | dataset | method | match | k | Hit@k | Doc@k | Overlap@k | Precision@k | MRR@k | nDCG@k | dense ms | rerank ms | p95 ms | chunk workers | embed batch | upsert batch | points | avg chars | status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines = header[:]
    for r in rows:
        lines.append(
            "| {timestamp} | {variant} | {dataset} | {method} | {match_mode} | {k} | {hit_rate} | {doc_hit} | {overlap_hit} | {precision} | {mrr} | {ndcg} | {dense_latency_avg_ms} | {rerank_latency_avg_ms} | {total_latency_p95_ms} | {chunk_workers} | {embed_batch_size} | {upsert_batch_size} | {points_count} | {avg_chunk_chars} | {status} |".format(
                **{k: _fmt(r.get(k)) for k in [
                    "timestamp", "variant", "dataset", "method", "match_mode", "k", "hit_rate",
                    "doc_hit", "overlap_hit", "precision", "mrr", "ndcg",
                    "dense_latency_avg_ms", "rerank_latency_avg_ms", "total_latency_p95_ms",
                    "chunk_workers", "embed_batch_size", "upsert_batch_size",
                    "points_count", "avg_chunk_chars", "status",
                ]}
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", "utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _report_row(report_path: Path, base: dict[str, Any], status: str, error: str | None = None) -> dict[str, Any]:
    row = {
        **base,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "error": error,
    }
    if report_path.exists():
        report = json.loads(report_path.read_text("utf-8"))
        meta = report.get("meta") or {}
        metrics = report.get("metrics") or {}
        k = base["k"]
        row.update(
            {
                "hit_rate": metrics.get(f"recall@{k}"),
                "precision": metrics.get(f"precision@{k}"),
                "mrr": metrics.get(f"mrr@{k}"),
                "ndcg": metrics.get(f"ndcg@{k}"),
                "hard_negative_rate": metrics.get(f"hard_negative_rate@{k}"),
                "doc_hit": metrics.get(f"doc_hit@{k}"),
                "overlap_hit": metrics.get(f"overlap_hit@{k}"),
                "dense_latency_avg_ms": metrics.get("dense_latency_avg_ms"),
                "rerank_latency_avg_ms": metrics.get("rerank_latency_avg_ms"),
                "total_latency_p95_ms": metrics.get("latency_p95_ms"),
                "avg_chunk_chars": metrics.get("avg_chunk_chars"),
                "avg_chunk_tokens": metrics.get("avg_chunk_tokens"),
                "chunk_method": meta.get("chunk_method"),
                "chunk_size": meta.get("chunk_size"),
                "chunk_overlap": meta.get("chunk_overlap"),
                "match_mode": meta.get("match_mode"),
                "overlap_threshold": meta.get("overlap_threshold"),
            }
        )
    return row


def _report_path(name: str, dataset: str, method: str, k: int) -> Path:
    return METRICS_DIR / f"chunking_{name}_{Path(dataset).stem}_{method}_k{k}.json"


def _valid_report(report_path: Path, *, args: argparse.Namespace) -> bool:
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text("utf-8"))
    except Exception:
        return False
    meta = report.get("meta") or {}
    metrics = report.get("metrics") or {}
    detailed = report.get("detailed") or []
    if meta.get("match_mode") != args.match_mode:
        return False
    if float(meta.get("overlap_threshold") or 0.0) != float(args.overlap_threshold):
        return False
    required = [f"recall@{args.k}", f"precision@{args.k}", f"mrr@{args.k}", f"ndcg@{args.k}"]
    if any(metrics.get(key) is None for key in required):
        return False
    if not detailed:
        return False
    # A previous CUDA failure can leave an empty collection; eval then completes
    # with all retrieved_count=0 and zero metrics. Treat that report as stale.
    if not any(int(item.get("retrieved_count") or 0) > 0 for item in detailed):
        return False
    return True


def _variant_reports_complete(name: str, datasets: list[str], methods: list[str], args: argparse.Namespace) -> bool:
    return all(
        _valid_report(_report_path(name, dataset, method, args.k), args=args)
        for dataset in datasets
        for method in methods
    )


def _variant_args(args: argparse.Namespace, variant: dict[str, Any], collection: str) -> argparse.Namespace:
    values = {
        "qdrant_collection": collection,
        "embed_model": args.embed_model,
        "embed_backend": args.embed_backend,
        "embed_device": args.embed_device,
        "embed_batch_size": args.embed_batch_size,
        "embed_max_seq_length": args.embed_max_seq_length,
        "upsert_batch_size": args.upsert_batch_size,
        "cuda_cleanup_every_batches": args.cuda_cleanup_every_batches,
        "chunk_workers": args.chunk_workers,
        "prefetch_docs": 100,
        "parallel_chunking": True,
        "recreate_collection": args.recreate_collection or args.force,
        "skip_existing": args.skip_existing,
        "chunk_method": variant["chunk_method"],
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "chunk_tokenizer": "character",
        "chunk_min_sentences": 2,
        "chunk_min_characters_per_sentence": 12,
        "semantic_threshold": 0.8,
        "semantic_similarity_window": 3,
        "semantic_skip_window": 0,
        "semantic_embedding_model": "minishlab/potion-base-32M",
        "parent_chunk_size": 3072,
        "parent_chunk_overlap": 256,
        "child_chunk_size": 768,
        "child_chunk_overlap": 96,
        "parent_chunker_method": "recursive_legal",
        "child_chunker_method": "sentence",
    }
    values.update({k: v for k, v in variant.items() if k != "collection"})
    return argparse.Namespace(**values)


def _inprocess_reindex_variant(
    *,
    args: argparse.Namespace,
    variant_name: str,
    variant: dict[str, Any],
    collection: str,
    docs: list[Any],
    embedder: Any,
    client: Any,
    recreate: bool | None = None,
) -> dict[str, Any]:
    from qdrant_client.http.models import PointStruct

    from scripts.reindex_local_corpus import (
        _cfg_from_args,
        _chunk_docs,
        _collection_exists,
        _doc_metadata,
        _ensure_collection,
        _flush_points,
        _iter_batches,
        _print_chunk_stats,
    )
    from src.rag.chunking import stable_chunk_point_id

    total_t0 = time.perf_counter()
    v_args = _variant_args(args, variant, collection)
    if recreate is not None:
        v_args.recreate_collection = recreate
    if v_args.chunk_method == "semantic" and v_args.chunk_workers > 1:
        print("[warn] Semantic chunking may use an embedding model inside Chonkie; forcing chunk_workers=1.")
        v_args.chunk_workers = 1

    if args.skip_existing and not v_args.recreate_collection and _collection_exists(client, collection):
        info = client.get_collection(collection)
        points = int(getattr(info, "points_count", 0) or 0)
        if points > 0:
            print(f"[skip] collection exists: {collection} points={points}")
            return {
                "variant": variant_name,
                "collection": collection,
                "status": "skipped_existing",
                "points_count": points,
                "total_indexing_time_sec": time.perf_counter() - total_t0,
            }
        print(f"[warn] Collection {collection!r} exists but has no points; reindexing it instead of skipping.")

    cfg = _cfg_from_args(v_args)
    chunk_t0 = time.perf_counter()
    chunks = _chunk_docs(
        docs,
        cfg,
        workers=v_args.chunk_workers,
        parallel=v_args.parallel_chunking,
        prefetch_docs=v_args.prefetch_docs,
    )
    chunking_time_sec = time.perf_counter() - chunk_t0
    _print_chunk_stats(chunks, len(docs), collection)
    print(f"Chunking time: {chunking_time_sec:.2f}s")

    _ensure_collection(client, collection, embedder.dim, v_args.recreate_collection)

    pending: list[PointStruct] = []
    upserted = 0
    encoded = 0
    embedding_time_sec = 0.0
    upsert_time_sec = 0.0
    for batch_no, batch in enumerate(_iter_batches(chunks, v_args.embed_batch_size), start=1):
        texts = [c.get("child_text") or c.get("text") or "" for c in batch]
        _log_cuda_memory(f"before embedding batch {batch_no} / {variant_name}")
        emb_t0 = time.perf_counter()
        vectors = embedder.embed_passages(texts, batch_size=v_args.embed_batch_size)
        embedding_time_sec += time.perf_counter() - emb_t0
        _log_cuda_memory(f"after embedding batch {batch_no} / {variant_name}")
        encoded += len(batch)
        print(f"Embedded batch {batch_no}: {encoded}/{len(chunks)} chunks")
        for chunk, vector in zip(batch, vectors):
            point_id = stable_chunk_point_id(chunk)
            payload = {
                **chunk,
                "point_id": point_id,
                "embed_model": args.embed_model,
                "embed_backend": args.embed_backend,
                "embed_dim": embedder.dim,
                "embed_metadata": getattr(embedder, "metadata", {}),
                "chunk_workers": v_args.chunk_workers,
                "embed_batch_size": v_args.embed_batch_size,
                "upsert_batch_size": v_args.upsert_batch_size,
            }
            pending.append(PointStruct(id=point_id, vector=vector, payload=payload))
            if len(pending) >= v_args.upsert_batch_size:
                up_t0 = time.perf_counter()
                upserted += _flush_points(client, collection, pending)
                upsert_time_sec += time.perf_counter() - up_t0
                print(f"Upserted {upserted}/{len(chunks)} points")
        del vectors
        del texts
        if v_args.cuda_cleanup_every_batches > 0 and batch_no % v_args.cuda_cleanup_every_batches == 0:
            _cleanup_cuda_cache(f"embedding batch {batch_no} / {variant_name}")
    if pending:
        up_t0 = time.perf_counter()
        upserted += _flush_points(client, collection, pending)
        upsert_time_sec += time.perf_counter() - up_t0
        print(f"Upserted {upserted}/{len(chunks)} points")

    info = client.get_collection(collection)
    total_indexing_time_sec = time.perf_counter() - total_t0
    return {
        "variant": variant_name,
        "collection": collection,
        "status": "ok",
        "vector_dim": embedder.dim,
        "points_count": getattr(info, "points_count", None),
        "chunking_time_sec": chunking_time_sec,
        "embedding_time_sec": embedding_time_sec,
        "upsert_time_sec": upsert_time_sec,
        "total_indexing_time_sec": total_indexing_time_sec,
    }


def _save_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _cleanup_cuda_cache(label: str) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _log_cuda_memory(label: str) -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"CUDA memory {label}: "
            f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB max_allocated={max_allocated:.2f}GB"
        )
    except Exception:
        pass
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"CUDA memory after {label}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    except Exception:
        pass


def _run_eval_inprocess(
    *,
    args: argparse.Namespace,
    dataset: str,
    method: str,
    collection: str,
    embedder: Any,
    retriever: Any,
    report_path: Path,
) -> dict[str, Any]:
    from src.eval.eval import evaluate_dataset

    report = evaluate_dataset(
        dataset,
        k=args.k,
        use_llm_judge=False,
        use_reranker=method == "reranker",
        reranker_model=args.reranker_model if method == "reranker" else None,
        reranker_fetch_k=args.reranker_fetch_k if method == "reranker" else None,
        embed_model=args.embed_model,
        embed_backend=args.embed_backend,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
        embed_max_seq_length=args.embed_max_seq_length,
        qdrant_collection=collection,
        match_mode=args.match_mode,
        overlap_threshold=args.overlap_threshold,
        method=f"chunk_{method}",
        embedder=embedder,
        retriever=retriever,
        llm=None,
    )
    _save_report(report, report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run local Chonkie chunking experiment matrix.", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["hard_only", "full", "top3"], default="hard_only")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["baseline", "reranker"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--input-dir", default="data/text")
    ap.add_argument("--fallback-raw-dir", default="data/raw")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--embed-backend", default="auto")
    ap.add_argument("--embed-device", default="cuda")
    ap.add_argument("--embed-max-seq-length", type=int, default=2048)
    ap.add_argument("--chunk-workers", type=int, default=4)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--upsert-batch-size", type=int, default=256)
    ap.add_argument(
        "--cuda-cleanup-every-batches",
        type=int,
        default=0,
        help="0 keeps PyTorch CUDA cache for reuse; positive values call empty_cache every N embedding batches.",
    )
    ap.add_argument("--match-mode", choices=["strict", "doc", "doc_overlap", "hybrid"], default="doc_overlap")
    ap.add_argument("--overlap-threshold", type=float, default=0.25)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--reranker-fetch-k", type=int, default=50)
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--recreate-collection", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subprocess-reindex", action="store_true", help="Use legacy one-subprocess-per-variant reindexing.")
    ap.add_argument("--subprocess-eval", action="store_true", help="Use legacy eval subprocesses instead of reusing the loaded embedder.")
    ap.add_argument(
        "--include-semantic",
        action="store_true",
        help="Include semantic chunker variants. They may load an extra embedding model and consume GPU memory.",
    )
    ap.add_argument(
        "--parallel-variants-cpu-only",
        action="store_true",
        help="Reserved for CPU-only runs; ignored when --embed-device starts with cuda to avoid parallel GPU embedders.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    variant_names = args.variants or (_load_top3() if args.stage == "top3" else list(VARIANTS))
    if not args.include_semantic:
        before = len(variant_names)
        variant_names = [name for name in variant_names if VARIANTS[name].get("chunk_method") != "semantic"]
        skipped = before - len(variant_names)
        if skipped:
            print(
                f"[warn] Skipping {skipped} semantic chunking variants by default. "
                "SemanticChunker can load an extra embedding model and hold GPU memory. "
                "Use --include-semantic to run them explicitly."
            )
    datasets = args.datasets
    if datasets is None:
        datasets = ["data/eval/eval_hard_v2.jsonl"] if args.stage == "hard_only" else DEFAULT_DATASETS
    datasets = _resolve_datasets(datasets)
    if not datasets:
        raise SystemExit("No eval datasets found.")
    if args.parallel_variants_cpu_only and str(args.embed_device).startswith("cuda"):
        print("[warn] --parallel-variants-cpu-only ignored for CUDA embedding; variants will run sequentially.")

    inprocess_reindex = not args.dry_run and not args.subprocess_reindex
    inprocess_eval = not args.dry_run and not args.subprocess_eval
    docs = None
    client = None
    embedder = None
    retriever = None

    if inprocess_reindex:
        from qdrant_client import QdrantClient
        from scripts.local_corpus_loader import load_local_documents

        docs = load_local_documents(
            ROOT / args.input_dir,
            ROOT / args.fallback_raw_dir if args.fallback_raw_dir else None,
            max_docs=args.max_docs,
        )
        print(f"Documents loaded once for all variants: {len(docs)}")
        client = QdrantClient(url=args.qdrant_url)

    if inprocess_reindex or inprocess_eval:
        from src.rag.core import RAGConfig, Retriever, STEmbedder

        print(f"Loading embedder once: {args.embed_model!r} on {args.embed_device!r}")
        embedder = STEmbedder(
            args.embed_model,
            device=args.embed_device,
            batch_size=args.embed_batch_size,
            max_seq_length=args.embed_max_seq_length,
            backend=args.embed_backend,
        )
        if inprocess_eval:
            cfg = dataclasses.replace(
                RAGConfig(),
                qdrant_url=args.qdrant_url,
                embed_model_name=args.embed_model,
                embed_device=args.embed_device,
                embed_batch_size=args.embed_batch_size,
                embed_max_seq_length=args.embed_max_seq_length,
                embed_backend=args.embed_backend,
            )
            retriever = Retriever(cfg, embedder)

    for name in variant_names:
        variant = VARIANTS[name]
        collection = variant["collection"]
        try:
            reports_complete = _variant_reports_complete(name, datasets, args.methods, args)
            needs_reindex = not reports_complete
            if reports_complete:
                print(f"[skip] metrics already exist for variant {name}; skipping reindex and eval.")
            if not args.skip_ingest:
                if not needs_reindex:
                    pass
                elif inprocess_reindex:
                    assert docs is not None and client is not None and embedder is not None
                    index_meta = _inprocess_reindex_variant(
                        args=args,
                        variant_name=name,
                        variant=variant,
                        collection=collection,
                        docs=docs,
                        embedder=embedder,
                        client=client,
                        recreate=True,
                    )
                    print(json.dumps(index_meta, ensure_ascii=False))
                else:
                    cmd = [
                        sys.executable, "scripts/reindex_local_corpus.py",
                        "--input-dir", args.input_dir,
                        "--fallback-raw-dir", args.fallback_raw_dir,
                        "--qdrant-url", args.qdrant_url,
                        "--qdrant-collection", collection,
                        "--embed-model", args.embed_model,
                        "--embed-backend", args.embed_backend,
                        "--embed-device", args.embed_device,
                        "--chunk-workers", str(args.chunk_workers),
                        "--embed-batch-size", str(args.embed_batch_size),
                        "--embed-max-seq-length", str(args.embed_max_seq_length),
                        "--upsert-batch-size", str(args.upsert_batch_size),
                        "--cuda-cleanup-every-batches", str(args.cuda_cleanup_every_batches),
                        "--embedding-workers", "1",
                        "--chunk-method", variant["chunk_method"],
                        "--no-network",
                    ]
                    for key, value in variant.items():
                        if key in ("collection", "chunk_method"):
                            continue
                        cmd.extend([_arg_name(key), str(value)])
                    if args.max_docs:
                        cmd.extend(["--max-docs", str(args.max_docs)])
                    if args.recreate_collection or args.force or needs_reindex:
                        cmd.append("--recreate-collection")
                    if args.skip_existing and not needs_reindex:
                        cmd.append("--skip-existing")
                    _run(cmd, dry_run=args.dry_run)

            for dataset in datasets:
                for method in args.methods:
                    report_path = _report_path(name, dataset, method, args.k)
                    if _valid_report(report_path, args=args):
                        print(f"[skip] metrics already exist: {report_path}")
                        row = _report_row(report_path, {
                            "variant": name,
                            "dataset": Path(dataset).name,
                            "method": method,
                            "k": args.k,
                            "qdrant_collection": collection,
                            "embed_model": args.embed_model,
                            "chunk_workers": args.chunk_workers,
                            "embed_batch_size": args.embed_batch_size,
                            "upsert_batch_size": args.upsert_batch_size,
                            "match_mode": args.match_mode,
                            "overlap_threshold": args.overlap_threshold,
                            "reranker_model": args.reranker_model if method == "reranker" else None,
                            "reranker_fetch_k": args.reranker_fetch_k if method == "reranker" else None,
                        }, "skipped_metrics")
                        _append_jsonl(row)
                        _write_markdown()
                        continue
                    if inprocess_eval:
                        assert embedder is not None and retriever is not None
                        _run_eval_inprocess(
                            args=args,
                            dataset=dataset,
                            method=method,
                            collection=collection,
                            embedder=embedder,
                            retriever=retriever,
                            report_path=report_path,
                        )
                        if method != "reranker" and hasattr(retriever, "_reranker"):
                            retriever._reranker = None
                        _cleanup_cuda_cache(f"eval {name}/{Path(dataset).name}/{method}")
                    else:
                        eval_cmd = [
                            sys.executable, "-m", "src.eval.eval",
                            "--dataset", dataset,
                            "--k", str(args.k),
                            "--no-judge",
                            "--qdrant-url", args.qdrant_url,
                            "--qdrant-collection", collection,
                            "--match-mode", args.match_mode,
                            "--overlap-threshold", str(args.overlap_threshold),
                            "--embed-model", args.embed_model,
                            "--embed-backend", args.embed_backend,
                            "--embed-device", args.embed_device,
                            "--embed-batch-size", str(args.embed_batch_size),
                            "--embed-max-seq-length", str(args.embed_max_seq_length),
                            "--model", f"chunk_{name}_{method}",
                            "--out", str(report_path),
                        ]
                        if method == "reranker":
                            eval_cmd.extend([
                                "--reranker",
                                "--reranker-model", args.reranker_model,
                                "--reranker-fetch-k", str(args.reranker_fetch_k),
                            ])
                        _run(eval_cmd, dry_run=args.dry_run)
                    row = _report_row(report_path, {
                        "variant": name,
                        "dataset": Path(dataset).name,
                        "method": method,
                        "k": args.k,
                        "qdrant_collection": collection,
                        "embed_model": args.embed_model,
                        "chunk_workers": args.chunk_workers,
                        "embed_batch_size": args.embed_batch_size,
                        "upsert_batch_size": args.upsert_batch_size,
                        "match_mode": args.match_mode,
                        "overlap_threshold": args.overlap_threshold,
                        "reranker_model": args.reranker_model if method == "reranker" else None,
                        "reranker_fetch_k": args.reranker_fetch_k if method == "reranker" else None,
                    }, "dry_run" if args.dry_run else "ok")
                    _append_jsonl(row)
                    _write_markdown()
            _cleanup_cuda_cache(f"variant {name}")
        except Exception as exc:
            _cleanup_cuda_cache(f"failed variant {name}")
            row = {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "variant": name,
                "dataset": "-",
                "method": "-",
                "k": args.k,
                "qdrant_collection": collection,
                "embed_model": args.embed_model,
                "chunk_workers": args.chunk_workers,
                "embed_batch_size": args.embed_batch_size,
                "upsert_batch_size": args.upsert_batch_size,
                "match_mode": args.match_mode,
                "overlap_threshold": args.overlap_threshold,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
            _append_jsonl(row)
            _write_markdown()
            print(f"[failed] {name}: {exc}")
            continue


if __name__ == "__main__":
    main()
