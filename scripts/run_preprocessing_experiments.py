#!/usr/bin/env python3
from __future__ import annotations

"""
Run preprocessing-profile experiments over the local corpus and eval datasets.

Dry run:
uv run python scripts/run_preprocessing_experiments.py \
  --dry-run \
  --profiles raw clean_basic clean_legal \
  --datasets eval_hard_dataset.jsonl \
  --methods baseline \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128

Smoke one profile:
uv run python scripts/reindex_local_corpus.py \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection rag_chunks_bge_m3_token1024_clean_legal_smoke \
  --embed-model BAAI/bge-m3 \
  --embed-device cuda \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --cleaning-profile clean_legal \
  --max-docs 30 \
  --recreate-collection \
  --save-cleaned-preview \
  --no-network
"""

import argparse
import datetime as dt
import dataclasses
import gc
import json
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.embedders import embedder_short_name
from src.rag.text_cleaning import SUPPORTED_CLEANING_PROFILES


DEFAULT_PROFILES = list(SUPPORTED_CLEANING_PROFILES)
DEFAULT_DATASETS = [
    "eval_dataset.jsonl",
    "eval_hard_dataset.jsonl",
    "eval_superhard_dataset.jsonl",
]
DEFAULT_METHODS = ["baseline", "reranker"]

METRICS_DIR = ROOT / "data" / "metrics"
SUMMARY_JSONL = METRICS_DIR / "preprocessing_experiments.jsonl"
SUMMARY_MD = METRICS_DIR / "preprocessing_experiments.md"


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _python_cmd(args: argparse.Namespace) -> list[str]:
    cmd = shlex.split(args.python_cmd.strip())
    if not cmd:
        raise ValueError("--python-cmd must not be empty")
    return cmd


def _run(cmd: list[str], *, dry_run: bool) -> str:
    print(" ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return ""
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output, stderr=output)
    return output


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_markdown() -> None:
    rows = _load_jsonl(SUMMARY_JSONL)
    rows = rows[-1000:]
    lines = [
        "| Profile | Dataset | Method | Hit@5 | Precision@5 | MRR@5 | nDCG@5 | HardNeg@5 | Avg removed % | Warnings docs | Collection | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        cleaning_stats = row.get("cleaning_stats") or {}
        k = int(row.get("k") or 5)
        hit = metrics.get(f"hit_at_{k}")
        if hit is None:
            hit = metrics.get(f"recall_at_{k}")
        lines.append(
            "| {profile} | {dataset} | {method} | {hit} | {precision} | {mrr} | {ndcg} | {hardneg} | {removed} | {warnings} | {collection} | {status} |".format(
                profile=_fmt(row.get("profile")),
                dataset=_fmt(row.get("dataset")),
                method=_fmt(row.get("method")),
                hit=_fmt(hit),
                precision=_fmt(metrics.get(f"precision_at_{k}")),
                mrr=_fmt(metrics.get(f"mrr_at_{k}")),
                ndcg=_fmt(metrics.get(f"ndcg_at_{k}")),
                hardneg=_fmt(metrics.get(f"hard_negative_rate_at_{k}")),
                removed=_fmt(cleaning_stats.get("avg_removed_char_ratio")),
                warnings=_fmt(cleaning_stats.get("docs_with_cleaning_warnings")),
                collection=_fmt(row.get("collection")),
                status=_fmt(row.get("status")),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_last_json_object(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _report_path(dataset: str, profile: str, method: str, k: int) -> Path:
    return METRICS_DIR / f"report_preprocessing_{Path(dataset).stem}_{profile}_{method}_k{k}.json"


def _valid_report(report_path: Path, k: int) -> bool:
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text("utf-8"))
    except Exception:
        return False
    metrics = report.get("metrics") or {}
    detailed = report.get("detailed") or []
    required = [f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}"]
    if any(metrics.get(key) is None for key in required):
        return False
    if not detailed:
        return False
    return True


def _chunk_tag(args: argparse.Namespace) -> str:
    if args.chunk_method == "parent_child":
        return f"parent{args.parent_chunk_size}_child{args.child_chunk_size}"
    if args.chunk_method == "recursive_legal":
        return f"recursive_legal_{args.chunk_size}"
    return f"{args.chunk_method}{args.chunk_size}"


def _collection_name(profile: str, args: argparse.Namespace) -> str:
    profile_tag = profile if profile.startswith("clean_") else f"clean_{profile}"
    return f"rag_chunks_{embedder_short_name(args.embed_model)}_{_chunk_tag(args)}_{profile_tag}"


def _reindex_profile(profile: str, collection: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_ingest:
        return {"status": "skipped_ingest", "collection_name": collection, "cleaning_profile": profile}
    cmd = [
        *_python_cmd(args),
        "scripts/reindex_local_corpus.py",
        "--input-dir",
        args.input_dir,
        "--fallback-raw-dir",
        args.fallback_raw_dir,
        "--qdrant-url",
        args.qdrant_url,
        "--qdrant-collection",
        collection,
        "--embed-model",
        args.embed_model,
        "--embed-backend",
        args.embed_backend,
        "--embed-device",
        args.embed_device,
        "--embed-batch-size",
        str(args.embed_batch_size),
        "--cuda-cleanup-every-batches",
        str(args.cuda_cleanup_every_batches),
        "--chunk-workers",
        str(args.chunk_workers),
        "--upsert-batch-size",
        str(args.upsert_batch_size),
        "--prefetch-docs",
        str(args.prefetch_docs),
        "--chunk-method",
        args.chunk_method,
        "--chunk-size",
        str(args.chunk_size),
        "--chunk-overlap",
        str(args.chunk_overlap),
        "--parent-chunk-size",
        str(args.parent_chunk_size),
        "--parent-chunk-overlap",
        str(args.parent_chunk_overlap),
        "--child-chunk-size",
        str(args.child_chunk_size),
        "--child-chunk-overlap",
        str(args.child_chunk_overlap),
        "--parent-chunker-method",
        args.parent_chunker_method,
        "--child-chunker-method",
        args.child_chunker_method,
        "--cleaning-profile",
        profile,
        "--max-cleaning-removed-ratio",
        str(args.max_cleaning_removed_ratio),
        "--no-network",
    ]
    if args.embed_max_seq_length is not None:
        cmd.extend(["--embed-max-seq-length", str(args.embed_max_seq_length)])
    if not args.parallel_chunking:
        cmd.append("--no-parallel-chunking")
    if args.max_docs is not None:
        cmd.extend(["--max-docs", str(args.max_docs)])
    if args.recreate_collection or args.force:
        cmd.append("--recreate-collection")
    if args.skip_existing and not args.force:
        cmd.append("--skip-existing")
    if args.save_cleaned_preview:
        cmd.append("--save-cleaned-preview")
    if args.dry_run:
        cmd.append("--dry-run")
    output = _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return {"status": "dry_run", "collection_name": collection, "cleaning_profile": profile}
    parsed = _extract_last_json_object(output)
    if parsed is None:
        return {
            "status": "failed",
            "collection_name": collection,
            "cleaning_profile": profile,
            "error": "Could not parse reindex JSON output.",
        }
    return parsed


def _save_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _inprocess_reindex_profile(
    *,
    profile: str,
    collection: str,
    docs: list[Any],
    original_docs: list[Any],
    embedder: Any,
    client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_ingest:
        return {"status": "skipped_ingest", "collection_name": collection, "cleaning_profile": profile}

    from qdrant_client.http.models import PointStruct
    from scripts.reindex_local_corpus import (
        _apply_cleaning,
        _cfg_from_args,
        _chunk_docs,
        _collection_exists,
        _doc_key,
        _ensure_collection,
        _flush_points,
        _iter_batches,
        _print_chunk_stats,
        _save_cleaned_previews,
    )
    from src.rag.chunking import stable_chunk_point_id

    total_t0 = time.perf_counter()
    if args.skip_existing and not args.force and not args.recreate_collection and _collection_exists(client, collection):
        info = client.get_collection(collection)
        points_count = int(getattr(info, "points_count", 0) or 0)
        if points_count > 0:
            print(
                json.dumps(
                    {
                        "status": "skipped_existing",
                        "collection_name": collection,
                        "cleaning_profile": profile,
                        "points_count": points_count,
                        "total_indexing_time_sec": time.perf_counter() - total_t0,
                    },
                    ensure_ascii=False,
                )
            )
            return {
                "status": "skipped_existing",
                "collection_name": collection,
                "cleaning_profile": profile,
                "points_count": points_count,
                "total_indexing_time_sec": time.perf_counter() - total_t0,
            }
        print(f"[warn] Collection {collection!r} exists but has no points; reindexing it instead of skipping.")

    cleaned_docs, cleaning_results, cleaning_summary = _apply_cleaning(
        docs,
        profile=profile,
        max_removed_ratio=args.max_cleaning_removed_ratio,
    )
    print(
        "Cleaning profile={profile} avg_removed={avg:.4f} p95_removed={p95:.4f} warnings_docs={warn}".format(
            profile=profile,
            avg=cleaning_summary["avg_removed_char_ratio"],
            p95=cleaning_summary["p95_removed_char_ratio"],
            warn=cleaning_summary["docs_with_cleaning_warnings"],
        )
    )
    if args.save_cleaned_preview:
        saved = _save_cleaned_previews(
            original_docs,
            cleaning_results,
            profile=profile,
            output_dir=Path(args.cleaned_preview_dir),
            max_docs=20,
        )
        print(f"Saved cleaned previews: {saved} documents -> {Path(args.cleaned_preview_dir) / profile}")

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
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        recreate_collection=args.recreate_collection or args.force,
        cuda_cleanup_every_batches=args.cuda_cleanup_every_batches,
    )
    if v_args.chunk_method == "semantic" and v_args.chunk_workers > 1:
        print("[warn] Semantic chunking may use an embedding model inside Chonkie; forcing chunk_workers=1.")
        v_args.chunk_workers = 1

    cfg = _cfg_from_args(v_args)
    chunk_t0 = time.perf_counter()
    chunks = _chunk_docs(
        cleaned_docs,
        cfg,
        workers=v_args.chunk_workers,
        parallel=v_args.parallel_chunking,
        prefetch_docs=v_args.prefetch_docs,
    )
    for chunk in chunks:
        key = (str(chunk.get("source_code") or ""), str(chunk.get("external_id") or ""))
        result = cleaning_results.get(key)
        if result is None:
            continue
        chunk["cleaning_profile"] = result.profile
        chunk["original_char_len"] = result.original_char_len
        chunk["cleaned_char_len"] = result.cleaned_char_len
        chunk["cleaning_removed_ratio"] = result.removed_char_ratio
        chunk["cleaning_removed_line_count"] = result.removed_line_count
        chunk["cleaning_warning_count"] = result.warning_count
        chunk["cleaning_warnings"] = list(result.warnings)
    chunking_time_sec = time.perf_counter() - chunk_t0
    chunk_stats = _print_chunk_stats(chunks, len(cleaned_docs), collection)
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
                    "embed_model": args.embed_model,
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
                _cleanup_cuda_cache(f"embedding batch {batch_no} / {profile}")
        if pending:
            up_t0 = time.perf_counter()
            upserted += _flush_points(client, collection, pending)
            upsert_time_sec += time.perf_counter() - up_t0
            print(f"Upserted {upserted}/{len(chunks)} points")
    finally:
        pending.clear()
        _cleanup_cuda_cache(f"reindex {profile}")

    info = client.get_collection(collection)
    total_indexing_time_sec = time.perf_counter() - total_t0
    meta = {
        "status": "ok",
        "collection_name": collection,
        "cleaning_profile": profile,
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
        "avg_chunk_char_len": chunk_stats.get("avg_chunk_char_len"),
        "avg_chunk_tokens": chunk_stats.get("avg_chunk_tokens"),
        "avg_chunks_per_doc": chunk_stats.get("avg_chunks_per_doc"),
        "unique_parent_count": chunk_stats.get("unique_parent_count"),
        "avg_removed_char_ratio": cleaning_summary["avg_removed_char_ratio"],
        "p95_removed_char_ratio": cleaning_summary["p95_removed_char_ratio"],
        "docs_with_cleaning_warnings": cleaning_summary["docs_with_cleaning_warnings"],
        "avg_original_char_len": cleaning_summary["avg_original_char_len"],
        "avg_cleaned_char_len": cleaning_summary["avg_cleaned_char_len"],
        "corpus_boilerplate_line_count": cleaning_summary["corpus_boilerplate_line_count"],
        "embed_metadata": embed_metadata,
    }
    print(json.dumps(meta, ensure_ascii=False))
    return meta


def _write_failed_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _eval_run(
    *,
    profile: str,
    dataset: str,
    method: str,
    collection: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report_path = _report_path(dataset, profile, method, args.k)
    if args.skip_existing and not args.force and _valid_report(report_path, args.k):
        report = json.loads(report_path.read_text("utf-8"))
        return {"status": "skipped_existing", "report_path": str(report_path), "report": report}

    cmd = [
        *_python_cmd(args),
        "-m",
        "src.eval.eval",
        "--dataset",
        dataset,
        "--k",
        str(args.k),
        "--no-judge",
        "--qdrant-url",
        args.qdrant_url,
        "--qdrant-collection",
        collection,
        "--embed-model",
        args.embed_model,
        "--embed-backend",
        args.embed_backend,
        "--embed-device",
        args.embed_device,
        "--embed-batch-size",
        str(args.embed_batch_size),
        "--model",
        f"preprocessing_{profile}_{method}",
        "--out",
        str(report_path),
    ]
    if args.embed_max_seq_length is not None:
        cmd.extend(["--embed-max-seq-length", str(args.embed_max_seq_length)])
    cmd.extend(["--match-mode", args.match_mode, "--overlap-threshold", str(args.overlap_threshold)])
    if method == "reranker":
        cmd.extend(
            [
                "--reranker",
                "--reranker-model",
                args.reranker_model,
                "--reranker-fetch-k",
                str(args.reranker_fetch_k),
            ]
        )
    output = _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return {"status": "dry_run", "report_path": str(report_path), "report": None}
    if not report_path.exists():
        _write_failed_report(
            report_path,
            {
                "status": "failed",
                "error": "Eval command finished without report output.",
                "stdout_stderr": output[-6000:],
            },
        )
        return {"status": "failed", "report_path": str(report_path), "report": None}
    report = json.loads(report_path.read_text("utf-8"))
    return {"status": "ok", "report_path": str(report_path), "report": report}


def _eval_run_inprocess(
    *,
    profile: str,
    dataset: str,
    method: str,
    collection: str,
    args: argparse.Namespace,
    embedder: Any,
    retriever: Any,
) -> dict[str, Any]:
    report_path = _report_path(dataset, profile, method, args.k)
    if args.skip_existing and not args.force and _valid_report(report_path, args.k):
        report = json.loads(report_path.read_text("utf-8"))
        return {"status": "skipped_existing", "report_path": str(report_path), "report": report}

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
        method=f"preprocessing_{profile}_{method}",
        embedder=embedder,
        retriever=retriever,
    )
    _save_report(report, report_path)
    _cleanup_cuda_cache(f"eval {profile}/{Path(dataset).name}/{method}")
    return {"status": "ok", "report_path": str(report_path), "report": report}


def _build_row(
    *,
    profile: str,
    dataset: str,
    method: str,
    collection: str,
    reindex_meta: dict[str, Any],
    eval_meta: dict[str, Any],
    args: argparse.Namespace,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    report = eval_meta.get("report") if isinstance(eval_meta, dict) else None
    metrics_raw = (report or {}).get("metrics") or {}
    k = args.k
    metrics = {
        f"hit_at_{k}": metrics_raw.get(f"recall@{k}"),
        f"recall_at_{k}": metrics_raw.get(f"recall@{k}"),
        f"precision_at_{k}": metrics_raw.get(f"precision@{k}"),
        f"mrr_at_{k}": metrics_raw.get(f"mrr@{k}"),
        f"ndcg_at_{k}": metrics_raw.get(f"ndcg@{k}"),
        f"hard_negative_rate_at_{k}": metrics_raw.get(f"hard_negative_rate@{k}"),
        "dense_latency_avg_ms": metrics_raw.get("dense_latency_avg_ms"),
        "rerank_latency_avg_ms": metrics_raw.get("rerank_latency_avg_ms"),
        "latency_p95_ms": metrics_raw.get("latency_p95_ms"),
    }
    cleaning_stats = {
        "avg_removed_char_ratio": reindex_meta.get("avg_removed_char_ratio"),
        "p95_removed_char_ratio": reindex_meta.get("p95_removed_char_ratio"),
        "docs_with_cleaning_warnings": reindex_meta.get("docs_with_cleaning_warnings"),
        "avg_original_char_len": reindex_meta.get("avg_original_char_len"),
        "avg_cleaned_char_len": reindex_meta.get("avg_cleaned_char_len"),
    }
    return {
        "timestamp": _timestamp(),
        "profile": profile,
        "dataset": Path(dataset).name,
        "dataset_path": dataset,
        "method": method,
        "k": k,
        "collection": collection,
        "embed_model": args.embed_model,
        "chunk_method": args.chunk_method,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "parent_chunk_size": args.parent_chunk_size if args.chunk_method == "parent_child" else None,
        "parent_chunk_overlap": args.parent_chunk_overlap if args.chunk_method == "parent_child" else None,
        "child_chunk_size": args.child_chunk_size if args.chunk_method == "parent_child" else None,
        "child_chunk_overlap": args.child_chunk_overlap if args.chunk_method == "parent_child" else None,
        "parent_chunker_method": args.parent_chunker_method if args.chunk_method == "parent_child" else None,
        "child_chunker_method": args.child_chunker_method if args.chunk_method == "parent_child" else None,
        "reranker_model": args.reranker_model if method == "reranker" else None,
        "reranker_fetch_k": args.reranker_fetch_k if method == "reranker" else None,
        "index_points_count": reindex_meta.get("points_count"),
        "avg_chunk_char_len": reindex_meta.get("avg_chunk_char_len"),
        "avg_chunks_per_doc": reindex_meta.get("avg_chunks_per_doc"),
        "metrics": metrics,
        "cleaning_stats": cleaning_stats,
        "report_path": eval_meta.get("report_path") if isinstance(eval_meta, dict) else None,
        "status": status,
        "error": error,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run preprocessing profile matrix experiments over local corpus and eval datasets.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES, choices=DEFAULT_PROFILES)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["baseline", "reranker"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--input-dir", default="data/text")
    ap.add_argument("--fallback-raw-dir", default="data/raw")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--embed-backend", default="auto")
    ap.add_argument("--embed-device", default="cuda")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--embed-max-seq-length", type=int, default=None)
    ap.add_argument("--cuda-cleanup-every-batches", type=int, default=1)
    ap.add_argument("--chunk-method", default="token", choices=["token", "sentence", "recursive", "recursive_legal", "semantic", "parent_child"])
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
    ap.add_argument("--reranker-fetch-k", type=int, default=50)
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--recreate-collection", action="store_true")
    ap.add_argument("--save-cleaned-preview", action="store_true")
    ap.add_argument("--max-cleaning-removed-ratio", type=float, default=0.7)
    ap.add_argument("--cleaned-preview-dir", default="data/metrics/cleaned_previews")
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--inprocess", dest="inprocess", action="store_true", default=True)
    mode_group.add_argument("--subprocess", dest="inprocess", action="store_false")
    ap.add_argument("--match-mode", choices=["strict", "doc", "doc_overlap", "hybrid"], default="strict")
    ap.add_argument("--overlap-threshold", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python-cmd", default="uv run python")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [dataset for dataset in args.datasets if (ROOT / dataset).exists()]
    if not datasets:
        raise SystemExit("No eval datasets found.")

    use_inprocess = bool(args.inprocess and not args.dry_run)
    docs = None
    original_docs = None
    client = None
    embedder = None
    retriever = None
    if use_inprocess:
        from qdrant_client import QdrantClient
        from scripts.local_corpus_loader import load_local_documents
        from src.rag.core import RAGConfig, Retriever, STEmbedder

        docs = load_local_documents(
            ROOT / args.input_dir,
            ROOT / args.fallback_raw_dir if args.fallback_raw_dir else None,
            max_docs=args.max_docs,
        )
        original_docs = docs
        print(f"Documents loaded once for all preprocessing profiles: {len(docs)}")
        client = QdrantClient(url=args.qdrant_url)
        print(f"Loading embedder once: {args.embed_model!r} on {args.embed_device!r}")
        embedder = STEmbedder(
            args.embed_model,
            device=args.embed_device,
            batch_size=args.embed_batch_size,
            max_seq_length=args.embed_max_seq_length,
            backend=args.embed_backend,
        )
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

    for profile in args.profiles:
        collection = _collection_name(profile, args)
        try:
            if use_inprocess:
                assert docs is not None and original_docs is not None and client is not None and embedder is not None
                reindex_meta = _inprocess_reindex_profile(
                    profile=profile,
                    collection=collection,
                    docs=docs,
                    original_docs=original_docs,
                    embedder=embedder,
                    client=client,
                    args=args,
                )
            else:
                reindex_meta = _reindex_profile(profile, collection, args)
            reindex_status = str(reindex_meta.get("status") or "ok")
            if reindex_status == "failed":
                raise RuntimeError(str(reindex_meta.get("error") or "reindex failed"))

            for dataset in datasets:
                for method in args.methods:
                    try:
                        if use_inprocess:
                            assert embedder is not None and retriever is not None
                            retriever.cfg = dataclasses.replace(retriever.cfg, qdrant_collection=collection)
                            eval_meta = _eval_run_inprocess(
                                profile=profile,
                                dataset=dataset,
                                method=method,
                                collection=collection,
                                args=args,
                                embedder=embedder,
                                retriever=retriever,
                            )
                            if method == "reranker" and hasattr(retriever, "_reranker"):
                                retriever._reranker = None
                                _cleanup_cuda_cache(f"reranker {profile}/{Path(dataset).name}")
                        else:
                            eval_meta = _eval_run(
                                profile=profile,
                                dataset=dataset,
                                method=method,
                                collection=collection,
                                args=args,
                            )
                        status = str(eval_meta.get("status") or "ok")
                        row = _build_row(
                            profile=profile,
                            dataset=dataset,
                            method=method,
                            collection=collection,
                            reindex_meta=reindex_meta,
                            eval_meta=eval_meta,
                            args=args,
                            status=status,
                        )
                    except Exception as exc:
                        report_path = _report_path(dataset, profile, method, args.k)
                        _write_failed_report(
                            report_path,
                            {
                                "status": "failed",
                                "error": str(exc),
                                "traceback": traceback.format_exc()[-6000:],
                            },
                        )
                        row = _build_row(
                            profile=profile,
                            dataset=dataset,
                            method=method,
                            collection=collection,
                            reindex_meta=reindex_meta,
                            eval_meta={"status": "failed", "report_path": str(report_path), "report": None},
                            args=args,
                            status="failed",
                            error=str(exc),
                        )
                    _append_jsonl(row)
                    _write_markdown()
            if use_inprocess:
                _cleanup_cuda_cache(f"profile {profile}")
        except Exception as exc:
            for dataset in datasets:
                for method in args.methods:
                    report_path = _report_path(dataset, profile, method, args.k)
                    _write_failed_report(
                        report_path,
                        {
                            "status": "failed",
                            "error": str(exc),
                            "traceback": traceback.format_exc()[-6000:],
                        },
                    )
                    row = _build_row(
                        profile=profile,
                        dataset=dataset,
                        method=method,
                        collection=collection,
                        reindex_meta={
                            "status": "failed",
                            "points_count": None,
                            "avg_chunk_char_len": None,
                            "avg_chunks_per_doc": None,
                            "avg_removed_char_ratio": None,
                            "p95_removed_char_ratio": None,
                            "docs_with_cleaning_warnings": None,
                            "avg_original_char_len": None,
                            "avg_cleaned_char_len": None,
                        },
                        eval_meta={"status": "failed", "report_path": str(report_path), "report": None},
                        args=args,
                        status="failed",
                        error=str(exc),
                    )
                    _append_jsonl(row)
            _write_markdown()
            print(f"[failed] profile={profile}: {exc}")
            continue
    if use_inprocess:
        try:
            if retriever is not None:
                del retriever
            if embedder is not None:
                del embedder
        except Exception:
            pass
        _cleanup_cuda_cache("preprocessing experiments finished")


if __name__ == "__main__":
    main()
