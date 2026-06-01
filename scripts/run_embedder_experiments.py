#!/usr/bin/env python3
from __future__ import annotations

"""
Run embedding-model experiments over the local corpus and golden eval datasets.

Dry run:
python scripts/run_embedder_experiments.py \
  --dry-run \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --datasets eval_hard_dataset.jsonl \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --no-heavy

Smoke reindex one model:
python scripts/run_embedder_experiments.py \
  --embedders intfloat/multilingual-e5-base \
  --datasets eval_hard_dataset.jsonl \
  --methods baseline \
  --k 5 \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --max-docs 50 \
  --recreate-collection \
  --force

BGE-M3 dense vs BM25 vs hybrid:
python scripts/run_embedder_experiments.py \
  --embedders BAAI/bge-m3 \
  --datasets eval_dataset.jsonl eval_hard_dataset.jsonl eval_superhard_dataset.jsonl \
  --methods baseline bm25 hybrid \
  --k 5 \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --match-mode doc_overlap \
  --overlap-threshold 0.25 \
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

from src.rag.embedders import (
    EMBEDDER_REGISTRY,
    HEAVY_EMBEDDERS,
    collection_name_for_embedder,
    embedder_short_name,
    registry_entry,
)

METRICS_DIR = ROOT / "data" / "metrics"
SUMMARY_JSONL = METRICS_DIR / "embedder_experiments.jsonl"
SUMMARY_MD = METRICS_DIR / "embedder_experiments.md"
DEFAULT_DATASETS = ["eval_dataset.jsonl", "eval_hard_dataset.jsonl", "eval_superhard_dataset.jsonl"]
DEFAULT_METHODS = ["baseline", "reranker"]
METHOD_CHOICES = ["baseline", "bm25", "hybrid", "reranker"]


def _arg_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _run(cmd: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    print(" ".join(cmd))
    if dry_run:
        return None
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    rc = proc.wait()
    output = "".join(lines)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=output)
    return subprocess.CompletedProcess(cmd, rc, stdout=output, stderr="")


def _cleanup_cuda_cache(label: str | None = None) -> None:
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
    if label:
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"CUDA memory after {label}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
        except Exception:
            pass


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
    rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("method")), -(r.get("ndcg") or 0.0), -(r.get("mrr") or 0.0)))
    header = [
        "| dataset | method | embedder | k | Hit@k | Precision@k | MRR@k | nDCG@k | HN@k | embed ms | dense ms | bm25 ms | fusion ms | rerank ms | p95 ms | points | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines = header[:]
    for r in rows[-800:]:
        lines.append(
            "| {dataset} | {method} | {embedder_short} | {k} | {hit_rate} | {precision} | {mrr} | {ndcg} | {hard_negative_rate} | {embedding_query_latency_avg_ms} | {dense_search_latency_avg_ms} | {bm25_search_latency_avg_ms} | {hybrid_fusion_latency_avg_ms} | {rerank_latency_avg_ms} | {total_latency_p95_ms} | {points_count} | {status} |".format(
                **{k: _fmt(r.get(k)) for k in [
                    "dataset", "method", "embedder_short", "k", "hit_rate", "precision", "mrr", "ndcg",
                    "hard_negative_rate", "embedding_query_latency_avg_ms", "dense_search_latency_avg_ms",
                    "bm25_search_latency_avg_ms", "hybrid_fusion_latency_avg_ms",
                    "rerank_latency_avg_ms", "total_latency_p95_ms", "points_count", "status",
                ]}
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", "utf-8")


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
                "vector_dim": meta.get("embed_dim"),
                "hit_rate": metrics.get(f"recall@{k}"),
                "precision": metrics.get(f"precision@{k}"),
                "mrr": metrics.get(f"mrr@{k}"),
                "ndcg": metrics.get(f"ndcg@{k}"),
                "hard_negative_rate": metrics.get(f"hard_negative_rate@{k}"),
                "embedding_query_latency_avg_ms": metrics.get("embedding_query_latency_avg_ms"),
                "dense_search_latency_avg_ms": metrics.get("dense_search_latency_avg_ms"),
                "bm25_search_latency_avg_ms": metrics.get("bm25_search_latency_avg_ms"),
                "hybrid_fusion_latency_avg_ms": metrics.get("hybrid_fusion_latency_avg_ms"),
                "rerank_latency_avg_ms": metrics.get("rerank_latency_avg_ms"),
                "total_latency_p95_ms": metrics.get("latency_p95_ms"),
                "chunk_method": meta.get("chunk_method"),
                "chunk_size": meta.get("chunk_size"),
                "chunk_overlap": meta.get("chunk_overlap"),
            }
        )
    return row


def _collection_points(qdrant_url: str, collection: str) -> int | None:
    try:
        from qdrant_client import QdrantClient

        info = QdrantClient(url=qdrant_url).get_collection(collection)
        return getattr(info, "points_count", None)
    except Exception:
        return None


def _save_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _inprocess_reindex_model(
    *,
    args: argparse.Namespace,
    model_name: str,
    collection: str,
    docs: list[Any],
    embedder: Any,
    client: Any,
    force_recreate: bool,
) -> dict[str, Any]:
    from qdrant_client.http.models import PointStruct

    from scripts.reindex_local_corpus import (
        _cfg_from_args,
        _chunk_docs,
        _collection_exists,
        _ensure_collection,
        _flush_points,
        _iter_batches,
        _print_chunk_stats,
    )
    from src.rag.chunking import stable_chunk_point_id

    total_t0 = time.perf_counter()
    batch_size = args.embed_batch_size or registry_entry(model_name).get("batch_size") or 32
    v_args = argparse.Namespace(
        chunk_method=args.chunk_method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_tokenizer="character",
        chunk_min_sentences=2,
        chunk_min_characters_per_sentence=12,
        semantic_threshold=0.8,
        semantic_similarity_window=3,
        semantic_skip_window=0,
        semantic_embedding_model="minishlab/potion-base-32M",
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        parent_chunker_method=args.parent_chunker_method,
        child_chunker_method=args.child_chunker_method,
        chunk_workers=args.chunk_workers,
        parallel_chunking=args.parallel_chunking,
        prefetch_docs=args.prefetch_docs,
        embed_batch_size=batch_size,
        upsert_batch_size=args.upsert_batch_size,
        recreate_collection=args.recreate_collection or args.force or force_recreate,
        cuda_cleanup_every_batches=args.cuda_cleanup_every_batches,
    )
    if v_args.chunk_method == "semantic" and v_args.chunk_workers > 1:
        print("[warn] Semantic chunking may use an embedding model inside Chonkie; forcing chunk_workers=1.")
        v_args.chunk_workers = 1

    if args.skip_existing and not v_args.recreate_collection and _collection_exists(client, collection):
        info = client.get_collection(collection)
        points = int(getattr(info, "points_count", 0) or 0)
        if points > 0:
            print(f"[skip] collection exists: {collection} points={points}")
            return {
                "status": "skipped_existing",
                "collection": collection,
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
    embed_dim = embedder.dim
    embed_metadata = getattr(embedder, "metadata", {})
    try:
        for batch_no, batch in enumerate(_iter_batches(chunks, v_args.embed_batch_size), start=1):
            texts = [c.get("child_text") or c.get("text") or "" for c in batch]
            emb_t0 = time.perf_counter()
            vectors = embedder.embed_passages(texts, batch_size=v_args.embed_batch_size)
            embedding_time_sec += time.perf_counter() - emb_t0
            encoded += len(batch)
            print(f"Embedded batch {batch_no}: {encoded}/{len(chunks)} chunks")
            for chunk, vector in zip(batch, vectors):
                point_id = stable_chunk_point_id(chunk)
                payload = {
                    **chunk,
                    "point_id": point_id,
                    "embed_model": model_name,
                    "embed_backend": args.embed_backend,
                    "embed_dim": embed_dim,
                    "embed_metadata": embed_metadata,
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
            del batch
            if v_args.cuda_cleanup_every_batches > 0 and batch_no % v_args.cuda_cleanup_every_batches == 0:
                _cleanup_cuda_cache(f"embedding batch {batch_no} / {embedder_short_name(model_name)}")
        if pending:
            up_t0 = time.perf_counter()
            upserted += _flush_points(client, collection, pending)
            upsert_time_sec += time.perf_counter() - up_t0
            print(f"Upserted {upserted}/{len(chunks)} points")
    finally:
        pending.clear()
        _cleanup_cuda_cache(f"reindex {embedder_short_name(model_name)}")

    info = client.get_collection(collection)
    total_indexing_time_sec = time.perf_counter() - total_t0
    meta = {
        "status": "ok",
        "collection_name": collection,
        "vector_dim": embed_dim,
        "points_count": getattr(info, "points_count", None),
        "chunk_workers": v_args.chunk_workers,
        "embed_batch_size": v_args.embed_batch_size,
        "upsert_batch_size": v_args.upsert_batch_size,
        "chunking_time_sec": chunking_time_sec,
        "embedding_time_sec": embedding_time_sec,
        "upsert_time_sec": upsert_time_sec,
        "indexing_time_sec": total_indexing_time_sec,
        "total_indexing_time_sec": total_indexing_time_sec,
        "embed_metadata": embed_metadata,
    }
    print(json.dumps(meta, ensure_ascii=False))
    return meta


def _inprocess_eval(
    *,
    args: argparse.Namespace,
    model_name: str,
    collection: str,
    dataset: str,
    method: str,
    report_path: Path,
    embedder: Any,
    retriever: Any,
) -> None:
    from src.eval.eval import evaluate_dataset

    batch_size = args.embed_batch_size or registry_entry(model_name).get("batch_size") or 32
    retrieval_mode = "bm25" if method == "bm25" else "hybrid" if method == "hybrid" else "dense"
    report = evaluate_dataset(
        dataset,
        k=args.k,
        use_llm_judge=False,
        use_reranker=method == "reranker",
        reranker_model=args.reranker_model if method == "reranker" else None,
        reranker_fetch_k=args.reranker_fetch_k if method == "reranker" else None,
        embed_model=model_name,
        embed_backend=args.embed_backend,
        embed_device=args.embed_device,
        embed_batch_size=batch_size,
        embed_max_seq_length=args.embed_max_seq_length,
        qdrant_collection=collection,
        retrieval_mode=retrieval_mode,
        bm25_fetch_k=args.bm25_fetch_k,
        hybrid_fetch_k=args.hybrid_fetch_k,
        hybrid_rrf_k=args.hybrid_rrf_k,
        match_mode=args.match_mode,
        overlap_threshold=args.overlap_threshold,
        method=f"embedder_{embedder_short_name(model_name)}_{method}",
        embedder=embedder,
        retriever=retriever,
    )
    _save_report(report, report_path)
    _cleanup_cuda_cache(f"eval {embedder_short_name(model_name)}/{Path(dataset).name}/{method}")


def _report_has_metrics(report_path: Path, k: int) -> bool:
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text("utf-8"))
    except Exception:
        return False
    metrics = report.get("metrics") or {}
    required = [f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}"]
    return all(metrics.get(key) is not None for key in required)


def _summary_has_metrics(*, dataset: str, embedder_short: str, method: str, k: int, collection: str) -> bool:
    if not SUMMARY_JSONL.exists():
        return False
    try:
        lines = SUMMARY_JSONL.read_text("utf-8").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not (
            row.get("dataset") == Path(dataset).name
            and row.get("embedder_short") == embedder_short
            and row.get("method") == method
            and int(row.get("k") or 0) == int(k)
            and row.get("collection") == collection
        ):
            continue

        metric_keys = ("hit_rate", "precision", "mrr", "ndcg")
        has_values = all(row.get(key) is not None for key in metric_keys)
        has_positive_metric = any(float(row.get(key) or 0.0) > 0.0 for key in metric_keys)
        return bool(row.get("status") in ("ok", "skipped_existing") and has_values and has_positive_metric)
    return False


def _run_has_metrics(report_path: Path, *, dataset: str, embedder_short: str, method: str, k: int, collection: str) -> bool:
    return _report_has_metrics(report_path, k) and _summary_has_metrics(
        dataset=dataset,
        embedder_short=embedder_short,
        method=method,
        k=k,
        collection=collection,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run embedding model experiment matrix.", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedders", nargs="+", default=list(EMBEDDER_REGISTRY))
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=METHOD_CHOICES)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--input-dir", default="data/text")
    ap.add_argument("--fallback-raw-dir", default="data/raw")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--chunk-method", default="token",
                    choices=["token", "sentence", "recursive", "recursive_legal", "semantic", "parent_child"])
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--chunk-overlap", type=int, default=128)
    ap.add_argument("--chunk-workers", type=int, default=4)
    ap.add_argument("--upsert-batch-size", type=int, default=256)
    ap.add_argument("--prefetch-docs", type=int, default=100)
    chunking_group = ap.add_mutually_exclusive_group()
    chunking_group.add_argument("--parallel-chunking", dest="parallel_chunking", action="store_true", default=True)
    chunking_group.add_argument("--no-parallel-chunking", dest="parallel_chunking", action="store_false")
    ap.add_argument("--parent-chunk-size", type=int, default=3072)
    ap.add_argument("--parent-chunk-overlap", type=int, default=256)
    ap.add_argument("--child-chunk-size", type=int, default=768)
    ap.add_argument("--child-chunk-overlap", type=int, default=96)
    ap.add_argument("--parent-chunker-method", default="recursive_legal")
    ap.add_argument("--child-chunker-method", default="sentence")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--reranker-fetch-k", type=int, default=12)
    ap.add_argument("--bm25-fetch-k", type=int, default=50)
    ap.add_argument("--hybrid-fetch-k", type=int, default=50)
    ap.add_argument("--hybrid-rrf-k", type=int, default=60)
    ap.add_argument("--embed-device", default="cuda")
    ap.add_argument("--embed-backend", default="auto")
    ap.add_argument("--embed-batch-size", type=int, default=None)
    ap.add_argument("--embed-max-seq-length", type=int, default=None)
    ap.add_argument("--cuda-cleanup-every-batches", type=int, default=1)
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--recreate-collection", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-heavy", action="store_true")
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--inprocess", dest="inprocess", action="store_true", default=True,
                            help="Load each embedder once and run reindex/eval in this process.")
    mode_group.add_argument("--subprocess", dest="inprocess", action="store_false",
                            help="Use subprocesses for reindex/eval.")
    ap.add_argument("--match-mode", choices=["strict", "doc", "doc_overlap", "hybrid"], default="strict")
    ap.add_argument("--overlap-threshold", type=float, default=0.25)
    return ap


def _reindex_cmd(
    args: argparse.Namespace,
    model_name: str,
    collection: str,
    *,
    force_recreate: bool = False,
    allow_skip_existing: bool = True,
) -> list[str]:
    entry = registry_entry(model_name)
    batch_size = args.embed_batch_size or entry.get("batch_size") or 32
    cmd = [
        sys.executable, "scripts/reindex_local_corpus.py",
        "--input-dir", args.input_dir,
        "--fallback-raw-dir", args.fallback_raw_dir,
        "--qdrant-url", args.qdrant_url,
        "--qdrant-collection", collection,
        "--embed-model", model_name,
        "--embed-backend", args.embed_backend,
        "--embed-device", args.embed_device,
        "--batch-size", str(batch_size),
        "--embed-batch-size", str(batch_size),
        "--cuda-cleanup-every-batches", str(args.cuda_cleanup_every_batches),
        "--chunk-workers", str(args.chunk_workers),
        "--upsert-batch-size", str(args.upsert_batch_size),
        "--prefetch-docs", str(args.prefetch_docs),
        "--chunk-method", args.chunk_method,
        "--no-network",
    ]
    if not args.parallel_chunking:
        cmd.append("--no-parallel-chunking")
    for key in [
        "chunk_size", "chunk_overlap", "parent_chunk_size", "parent_chunk_overlap",
        "child_chunk_size", "child_chunk_overlap", "parent_chunker_method", "child_chunker_method",
    ]:
        cmd.extend([_arg_name(key), str(getattr(args, key))])
    if args.embed_max_seq_length:
        cmd.extend(["--embed-max-seq-length", str(args.embed_max_seq_length)])
    if args.max_docs:
        cmd.extend(["--max-docs", str(args.max_docs)])
    if args.recreate_collection or args.force or force_recreate:
        cmd.append("--recreate-collection")
    if args.skip_existing and allow_skip_existing and not force_recreate:
        cmd.append("--skip-existing")
    return cmd


def _eval_cmd(args: argparse.Namespace, model_name: str, collection: str, dataset: str, method: str, report_path: Path) -> list[str]:
    entry = registry_entry(model_name)
    batch_size = args.embed_batch_size or entry.get("batch_size") or 32
    cmd = [
        sys.executable, "-m", "src.eval.eval",
        "--dataset", dataset,
        "--k", str(args.k),
        "--no-judge",
        "--qdrant-url", args.qdrant_url,
        "--qdrant-collection", collection,
        "--embed-model", model_name,
        "--embed-backend", args.embed_backend,
        "--embed-device", args.embed_device,
        "--embed-batch-size", str(batch_size),
        "--model", f"embedder_{embedder_short_name(model_name)}_{method}",
        "--out", str(report_path),
        "--match-mode", args.match_mode,
        "--overlap-threshold", str(args.overlap_threshold),
    ]
    if method == "bm25":
        cmd.extend(["--retrieval-mode", "bm25", "--bm25-fetch-k", str(args.bm25_fetch_k)])
    elif method == "hybrid":
        cmd.extend([
            "--retrieval-mode", "hybrid",
            "--bm25-fetch-k", str(args.bm25_fetch_k),
            "--hybrid-fetch-k", str(args.hybrid_fetch_k),
            "--hybrid-rrf-k", str(args.hybrid_rrf_k),
        ])
    else:
        cmd.extend(["--retrieval-mode", "dense"])
    if args.embed_max_seq_length:
        cmd.extend(["--embed-max-seq-length", str(args.embed_max_seq_length)])
    if method == "reranker":
        cmd.extend([
            "--reranker",
            "--reranker-model", args.reranker_model,
            "--reranker-fetch-k", str(args.reranker_fetch_k),
        ])
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    embedders = [m for m in args.embedders if not (args.no_heavy and m in HEAVY_EMBEDDERS)]
    datasets = [d for d in args.datasets if (ROOT / d).exists()]
    if not datasets:
        raise SystemExit("No eval datasets found. Use root golden datasets: eval_dataset.jsonl eval_hard_dataset.jsonl eval_superhard_dataset.jsonl")
    use_inprocess = bool(args.inprocess and not args.dry_run)
    docs = None
    client = None
    if use_inprocess and not args.skip_ingest:
        from qdrant_client import QdrantClient
        from scripts.local_corpus_loader import load_local_documents

        docs = load_local_documents(
            ROOT / args.input_dir,
            ROOT / args.fallback_raw_dir if args.fallback_raw_dir else None,
            max_docs=args.max_docs,
        )
        print(f"Documents loaded once for all embedder runs: {len(docs)}")
        client = QdrantClient(url=args.qdrant_url)

    for model_name in embedders:
        short = embedder_short_name(model_name)
        collection = collection_name_for_embedder(
            model_name,
            chunk_method=args.chunk_method,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            parent_chunk_size=args.parent_chunk_size,
            child_chunk_size=args.child_chunk_size,
        )
        points_count = None
        indexing_time_sec = None
        embedder = None
        retriever = None
        try:
            requested_reports = [
                (dataset, method, METRICS_DIR / f"report_embedder_{Path(dataset).stem}_{short}_{method}_k{args.k}.json")
                for dataset in datasets
                for method in args.methods
            ]
            missing_or_invalid_reports = [
                path for dataset, method, path in requested_reports
                if not _run_has_metrics(
                    path,
                    dataset=dataset,
                    embedder_short=short,
                    method=method,
                    k=args.k,
                    collection=collection,
                )
            ]
            force_reindex_for_metrics = bool(args.skip_existing and missing_or_invalid_reports and not args.force)
            if force_reindex_for_metrics:
                print(
                    f"[metrics-missing] {model_name}: {len(missing_or_invalid_reports)} report(s) missing metrics; "
                    f"recreating {collection} and recomputing requested eval matrix."
                )
            if use_inprocess:
                from src.rag.core import RAGConfig, Retriever, STEmbedder

                batch_size = args.embed_batch_size or registry_entry(model_name).get("batch_size") or 32
                print(f"Loading embedder once: {model_name!r} on {args.embed_device!r}")
                embedder = STEmbedder(
                    model_name,
                    device=args.embed_device,
                    batch_size=batch_size,
                    max_seq_length=args.embed_max_seq_length,
                    normalize_embeddings=None,
                    backend=args.embed_backend,
                    trust_remote_code=None,
                )
                cfg = dataclasses.replace(
                    RAGConfig(),
                    qdrant_url=args.qdrant_url,
                    qdrant_collection=collection,
                    embed_model_name=model_name,
                    embed_device=args.embed_device,
                    embed_batch_size=batch_size,
                    embed_max_seq_length=args.embed_max_seq_length,
                    embed_backend=args.embed_backend,
                )
                retriever = Retriever(cfg, embedder)
            if not args.skip_ingest:
                if use_inprocess:
                    assert docs is not None and client is not None and embedder is not None
                    index_meta = _inprocess_reindex_model(
                        args=args,
                        model_name=model_name,
                        collection=collection,
                        docs=docs,
                        embedder=embedder,
                        client=client,
                        force_recreate=force_reindex_for_metrics,
                    )
                    indexing_time_sec = index_meta.get("total_indexing_time_sec")
                    points_count = index_meta.get("points_count")
                else:
                    t0 = time.perf_counter()
                    cp = _run(
                        _reindex_cmd(
                            args,
                            model_name,
                            collection,
                            force_recreate=force_reindex_for_metrics,
                            allow_skip_existing=not force_reindex_for_metrics,
                        ),
                        dry_run=args.dry_run,
                    )
                    indexing_time_sec = 0.0 if args.dry_run else time.perf_counter() - t0
                    points_count = _collection_points(args.qdrant_url, collection) if not args.dry_run else None
            elif force_reindex_for_metrics:
                print(
                    f"[warn] --skip-ingest is set, cannot recreate {collection}; "
                    "will recompute eval reports against the existing collection."
                )

            for dataset, method, report_path in requested_reports:
                if (
                    args.skip_existing
                    and _run_has_metrics(
                        report_path,
                        dataset=dataset,
                        embedder_short=short,
                        method=method,
                        k=args.k,
                        collection=collection,
                    )
                    and not args.force
                    and not force_reindex_for_metrics
                ):
                    row = _report_row(report_path, {
                        "dataset": Path(dataset).name,
                        "embedder": model_name,
                        "embedder_short": short,
                        "collection": collection,
                        "chunk_method": args.chunk_method,
                        "chunk_size": args.chunk_size,
                        "chunk_overlap": args.chunk_overlap,
                        "method": method,
                        "reranker_model": args.reranker_model if method == "reranker" else None,
                        "reranker_fetch_k": args.reranker_fetch_k if method == "reranker" else None,
                        "k": args.k,
                        "points_count": points_count,
                        "indexing_time_sec": indexing_time_sec,
                    }, "skipped_existing")
                    _append_jsonl(row)
                    _write_markdown()
                    continue
                if use_inprocess:
                    assert embedder is not None and retriever is not None
                    _inprocess_eval(
                        args=args,
                        model_name=model_name,
                        collection=collection,
                        dataset=dataset,
                        method=method,
                        report_path=report_path,
                        embedder=embedder,
                        retriever=retriever,
                    )
                    if method == "reranker" and hasattr(retriever, "_reranker"):
                        retriever._reranker = None
                        _cleanup_cuda_cache(f"reranker {short}/{Path(dataset).name}")
                else:
                    cp = _run(_eval_cmd(args, model_name, collection, dataset, method, report_path), dry_run=args.dry_run)
                row = _report_row(report_path, {
                    "dataset": Path(dataset).name,
                    "embedder": model_name,
                    "embedder_short": short,
                    "collection": collection,
                    "chunk_method": args.chunk_method,
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "method": method,
                    "reranker_model": args.reranker_model if method == "reranker" else None,
                    "reranker_fetch_k": args.reranker_fetch_k if method == "reranker" else None,
                    "k": args.k,
                    "points_count": points_count,
                    "indexing_time_sec": indexing_time_sec,
                }, "dry_run" if args.dry_run else "ok")
                _append_jsonl(row)
                _write_markdown()
            if use_inprocess:
                del retriever
                del embedder
                retriever = None
                embedder = None
                _cleanup_cuda_cache(f"embedder {short}")
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or exc.stdout or str(exc))[-4000:]
            print(error)
            row = {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": "-",
                "embedder": model_name,
                "embedder_short": short,
                "collection": collection,
                "chunk_method": args.chunk_method,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "method": "-",
                "k": args.k,
                "points_count": points_count,
                "indexing_time_sec": indexing_time_sec,
                "status": "failed",
                "error": error,
            }
            _append_jsonl(row)
            _write_markdown()
            continue
        except Exception as exc:
            row = {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": "-",
                "embedder": model_name,
                "embedder_short": short,
                "collection": collection,
                "chunk_method": args.chunk_method,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "method": "-",
                "k": args.k,
                "points_count": points_count,
                "indexing_time_sec": indexing_time_sec,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
            _append_jsonl(row)
            _write_markdown()
            print(f"[failed] {model_name}: {exc}")
            continue
        finally:
            if use_inprocess:
                try:
                    if retriever is not None:
                        del retriever
                    if embedder is not None:
                        del embedder
                except Exception:
                    pass
                _cleanup_cuda_cache(f"finalize {short}")


if __name__ == "__main__":
    main()
