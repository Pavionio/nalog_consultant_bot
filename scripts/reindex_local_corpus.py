#!/usr/bin/env python3
from __future__ import annotations

"""
Reindex already extracted local documents with experimental Chonkie chunkers.

Smoke reindex local corpus:
python scripts/reindex_local_corpus.py \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection rag_chunks_bge_m3_token_1024_128_smoke \
  --embed-model BAAI/bge-m3 \
  --embed-device cuda \
  --chunk-method token \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --chunk-workers 8 \
  --embed-batch-size 64 \
  --upsert-batch-size 256 \
  --max-docs 20 \
  --recreate-collection \
  --no-network

Smoke parent-child:
python scripts/reindex_local_corpus.py \
  --input-dir data/text \
  --fallback-raw-dir data/raw \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection rag_chunks_bge_m3_parent3072_child768_smoke \
  --embed-model BAAI/bge-m3 \
  --embed-device cuda \
  --chunk-method parent_child \
  --parent-chunk-size 3072 \
  --parent-chunk-overlap 256 \
  --child-chunk-size 768 \
  --child-chunk-overlap 96 \
  --max-docs 20 \
  --recreate-collection \
  --no-network

Smoke reindex with cleaning:
python scripts/reindex_local_corpus.py \
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
import concurrent.futures as cf
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from scripts.local_corpus_loader import LocalDocument, load_local_documents
from src.rag.chunking import ChunkingConfig, chunk_document, stable_chunk_point_id
from src.rag.text_cleaning import (
    CleaningResult,
    SUPPORTED_CLEANING_PROFILES,
    build_corpus_boilerplate_stats,
    clean_text,
)


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reindex local data/text or data/raw documents into a Chonkie experiment Qdrant collection.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-dir", default="data/text")
    ap.add_argument("--fallback-raw-dir", default="data/raw")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-collection", required=False, default="rag_chunks_local_experiment")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--embed-backend", default=os.getenv("EMBED_BACKEND", "auto"))
    ap.add_argument("--embed-device", default=os.getenv("EMBED_DEVICE", "cuda"))
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--embed-max-seq-length", type=int, default=None)
    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument("--embed-normalize", dest="embed_normalize", action="store_true", default=None)
    norm_group.add_argument("--no-embed-normalize", dest="embed_normalize", action="store_false")
    trust_group = ap.add_mutually_exclusive_group()
    trust_group.add_argument("--embed-trust-remote-code", dest="embed_trust_remote_code", action="store_true", default=None)
    trust_group.add_argument("--no-embed-trust-remote-code", dest="embed_trust_remote_code", action="store_false")
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--passage-prefix", default=None)
    ap.add_argument("--query-instruction", default=None)
    ap.add_argument("--chunk-method", required=False, default="token",
                    choices=["token", "sentence", "recursive", "recursive_legal", "semantic", "parent_child"])
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--chunk-overlap", type=int, default=128)
    ap.add_argument("--chunk-tokenizer", default="character")
    ap.add_argument("--chunk-min-sentences", type=int, default=2)
    ap.add_argument("--chunk-min-characters-per-sentence", type=int, default=12)
    ap.add_argument("--semantic-threshold", type=float, default=0.8)
    ap.add_argument("--semantic-similarity-window", type=int, default=3)
    ap.add_argument("--semantic-skip-window", type=int, default=0)
    ap.add_argument("--semantic-embedding-model", default="minishlab/potion-base-32M")
    ap.add_argument("--parent-chunk-size", type=int, default=3072)
    ap.add_argument("--parent-chunk-overlap", type=int, default=256)
    ap.add_argument("--child-chunk-size", type=int, default=768)
    ap.add_argument("--child-chunk-overlap", type=int, default=96)
    ap.add_argument("--parent-chunker-method", default="recursive_legal")
    ap.add_argument("--child-chunker-method", default="sentence")
    ap.add_argument("--batch-size", type=int, default=None, help="Backward-compatible alias for --embed-batch-size")
    ap.add_argument("--upsert-batch-size", type=int, default=256)
    ap.add_argument("--prefetch-docs", type=int, default=100)
    parallel_group = ap.add_mutually_exclusive_group()
    parallel_group.add_argument("--parallel-chunking", dest="parallel_chunking", action="store_true", default=True)
    parallel_group.add_argument("--no-parallel-chunking", dest="parallel_chunking", action="store_false")
    ap.add_argument("--chunk-workers", type=int, default=4)
    ap.add_argument("--embedding-workers", type=int, default=1)
    ap.add_argument("--allow-multiple-gpu-embedders", action="store_true")
    ap.add_argument(
        "--cuda-cleanup-every-batches",
        type=int,
        default=1,
        help="0 keeps PyTorch CUDA cache for reuse; positive values call empty_cache every N embedding batches.",
    )
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--cleaning-profile", choices=SUPPORTED_CLEANING_PROFILES, default=None)
    ap.add_argument("--save-cleaned-preview", action="store_true")
    ap.add_argument("--cleaned-preview-dir", default="data/metrics/cleaned_previews")
    ap.add_argument("--max-cleaning-removed-ratio", type=float, default=0.7)
    ap.add_argument("--recreate-collection", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-network", type=_bool_arg, nargs="?", const=True, default=True)
    return ap


def _cfg_from_args(args: argparse.Namespace) -> ChunkingConfig:
    return ChunkingConfig(
        chunk_method=args.chunk_method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_tokenizer=args.chunk_tokenizer,
        chunk_min_sentences=args.chunk_min_sentences,
        chunk_min_characters_per_sentence=args.chunk_min_characters_per_sentence,
        semantic_threshold=args.semantic_threshold,
        semantic_similarity_window=args.semantic_similarity_window,
        semantic_skip_window=args.semantic_skip_window,
        semantic_embedding_model=args.semantic_embedding_model,
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        parent_chunker_method=args.parent_chunker_method,
        child_chunker_method=args.child_chunker_method,
    )


def _doc_metadata(doc: LocalDocument) -> dict[str, Any]:
    return {
        "source_code": doc.source_code,
        "external_id": doc.external_id,
        "title": doc.title,
        "canonical_url": doc.canonical_url,
        "document_date": doc.document_date,
        "publication_date": doc.publication_date,
        "document_number": doc.document_number,
        "local_metadata": doc.metadata or {},
    }


def _collection_exists(client: QdrantClient, collection: str) -> bool:
    names = {c.name for c in client.get_collections().collections}
    return collection in names


def _ensure_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool) -> None:
    if recreate and _collection_exists(client, collection):
        print(f"Deleting collection {collection!r}...")
        client.delete_collection(collection)
    if not _collection_exists(client, collection):
        print(f"Creating collection {collection!r}, vector_size={vector_size}")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    existing_size = getattr(vectors, "size", None)
    if existing_size is None and isinstance(vectors, dict):
        existing_size = next(iter(vectors.values())).size
    if int(existing_size) != int(vector_size):
        raise RuntimeError(
            f"Collection {collection!r} has vector size {existing_size}, but embedder returns {vector_size}. "
            "Use --recreate-collection or another collection name."
        )
    print(f"Collection {collection!r} exists, vector_size={existing_size}")


def _iter_batches(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _chunk_one_doc(args: tuple[int, int, LocalDocument, ChunkingConfig]) -> tuple[int, list[dict[str, Any]]]:
    doc_i, _total, doc, cfg = args
    return doc_i, chunk_document(doc.text, _doc_metadata(doc), cfg)


def _chunk_docs(
    docs: list[LocalDocument],
    cfg: ChunkingConfig,
    *,
    workers: int,
    parallel: bool,
    prefetch_docs: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    total = len(docs)
    log_every = max(1, int(prefetch_docs))
    if not parallel or workers <= 1 or total <= 1:
        for doc_i, doc in enumerate(docs, start=1):
            _, doc_chunks = _chunk_one_doc((doc_i, total, doc, cfg))
            chunks.extend(doc_chunks)
            if doc_i % log_every == 0 or doc_i == total:
                print(f"Chunked documents: {doc_i}/{total} chunks={len(chunks)}")
        return chunks

    print(f"Chunking documents in parallel: workers={workers}, prefetch_docs={prefetch_docs}")
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_chunk_one_doc, (doc_i, total, doc, cfg))
            for doc_i, doc in enumerate(docs, start=1)
        ]
        done = 0
        for fut in cf.as_completed(futures):
            _doc_i, doc_chunks = fut.result()
            chunks.extend(doc_chunks)
            done += 1
            if done % log_every == 0 or done == total:
                print(f"Chunked documents: {done}/{total} chunks={len(chunks)}")
    return chunks


def _print_chunk_stats(chunks: list[dict[str, Any]], docs_count: int, collection: str) -> dict[str, Any]:
    lengths = [int(c.get("chunk_char_len") or len(c.get("text") or "")) for c in chunks]
    token_counts = [c.get("chunk_token_count") for c in chunks if c.get("chunk_token_count") is not None]
    parent_ids = {c.get("parent_id") for c in chunks if c.get("parent_id")}
    print(f"Qdrant collection: {collection}")
    print(f"Chunks count: {len(chunks)}")
    print(f"Average chunk length: {mean(lengths):.1f}" if lengths else "Average chunk length: 0")
    print(f"Average chunk tokens: {mean(token_counts):.1f}" if token_counts else "Average chunk tokens: n/a")
    print(f"Chunks per doc avg: {len(chunks) / max(1, docs_count):.2f}")
    if parent_ids:
        print(f"Unique parent count: {len(parent_ids)}")
    return {
        "avg_chunk_char_len": mean(lengths) if lengths else 0.0,
        "avg_chunk_tokens": mean(token_counts) if token_counts else None,
        "avg_chunks_per_doc": len(chunks) / max(1, docs_count),
        "unique_parent_count": len(parent_ids) if parent_ids else None,
    }


def _flush_points(client: QdrantClient, collection: str, pending: list[PointStruct]) -> int:
    if not pending:
        return 0
    client.upsert(collection_name=collection, points=list(pending))
    count = len(pending)
    pending.clear()
    return count


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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _doc_key(doc: LocalDocument) -> tuple[str, str]:
    return (doc.source_code, doc.external_id)


def _ensure_warning(result: CleaningResult, message: str) -> CleaningResult:
    if message in result.warnings:
        return result
    warnings = [*result.warnings, message]
    return CleaningResult(
        text=result.text,
        profile=result.profile,
        original_char_len=result.original_char_len,
        cleaned_char_len=result.cleaned_char_len,
        removed_char_ratio=result.removed_char_ratio,
        removed_line_count=result.removed_line_count,
        warning_count=len(warnings),
        warnings=warnings,
        stats=dict(result.stats),
    )


def _apply_cleaning(
    docs: list[LocalDocument],
    *,
    profile: str,
    max_removed_ratio: float,
) -> tuple[list[LocalDocument], dict[tuple[str, str], CleaningResult], dict[str, Any]]:
    corpus_stats = build_corpus_boilerplate_stats(docs) if profile == "clean_no_boilerplate" else None
    cleaned_docs: list[LocalDocument] = []
    by_doc_key: dict[tuple[str, str], CleaningResult] = {}
    ratios: list[float] = []
    original_lens: list[int] = []
    cleaned_lens: list[int] = []
    docs_with_warnings = 0

    for doc in docs:
        result = clean_text(doc.text, profile=profile, corpus_stats=corpus_stats)
        if result.removed_char_ratio > max_removed_ratio:
            result = _ensure_warning(
                result,
                f"Removed ratio exceeds --max-cleaning-removed-ratio={max_removed_ratio:.2f}.",
            )
        if result.warning_count > 0:
            docs_with_warnings += 1
        ratios.append(result.removed_char_ratio)
        original_lens.append(result.original_char_len)
        cleaned_lens.append(result.cleaned_char_len)
        key = _doc_key(doc)
        by_doc_key[key] = result
        cleaned_docs.append(LocalDocument(**{**doc.__dict__, "text": result.text}))

    summary = {
        "cleaning_profile": profile,
        "avg_removed_char_ratio": mean(ratios) if ratios else 0.0,
        "p95_removed_char_ratio": _percentile(ratios, 0.95),
        "docs_with_cleaning_warnings": docs_with_warnings,
        "avg_original_char_len": mean(original_lens) if original_lens else 0.0,
        "avg_cleaned_char_len": mean(cleaned_lens) if cleaned_lens else 0.0,
        "corpus_boilerplate_line_count": int((corpus_stats or {}).get("boilerplate_line_count") or 0),
    }
    return cleaned_docs, by_doc_key, summary


def _safe_file_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "doc"))
    return cleaned.strip("._") or "doc"


def _save_cleaned_previews(
    docs: list[LocalDocument],
    cleaning_map: dict[tuple[str, str], CleaningResult],
    *,
    profile: str,
    output_dir: Path,
    max_docs: int = 20,
) -> int:
    profile_dir = output_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for doc in docs:
        if saved >= max_docs:
            break
        result = cleaning_map.get(_doc_key(doc))
        if result is None:
            continue
        doc_stem = _safe_file_stem(f"{doc.source_code}_{doc.external_id}")
        path = profile_dir / f"{doc_stem}.md"
        warnings = ", ".join(result.warnings) if result.warnings else "-"
        body = "\n".join(
            [
                "# Документ",
                f"source_code: {doc.source_code}",
                f"external_id: {doc.external_id}",
                f"title: {doc.title or '-'}",
                "",
                "## Статистика",
                f"original_char_len: {result.original_char_len}",
                f"cleaned_char_len: {result.cleaned_char_len}",
                f"removed_char_ratio: {result.removed_char_ratio:.6f}",
                f"warnings: {warnings}",
                "",
                "## Original preview",
                "",
                (doc.text or "")[:3000],
                "",
                "## Cleaned preview",
                "",
                (result.text or "")[:3000],
                "",
            ]
        )
        path.write_text(body, encoding="utf-8")
        saved += 1
    return saved


def main() -> None:
    total_t0 = time.perf_counter()
    args = build_arg_parser().parse_args()
    if args.batch_size is not None:
        args.embed_batch_size = args.batch_size
    if args.embedding_workers > 1 and str(args.embed_device).startswith("cuda") and not args.allow_multiple_gpu_embedders:
        print("[warn] Multiple CUDA embedding workers can cause OOM/device not ready. Use one GPU embedding worker.")
        print("[warn] Forcing --embedding-workers=1. Use --allow-multiple-gpu-embedders to override.")
        args.embedding_workers = 1
    elif args.embedding_workers > 1 and str(args.embed_device).startswith("cuda"):
        print("[warn] Multiple CUDA embedding workers can cause OOM/device not ready. Use one GPU embedding worker.")
    if args.chunk_method == "semantic" and args.parallel_chunking and args.chunk_workers > 1:
        print("[warn] Semantic chunking may use an embedding model inside Chonkie; forcing --chunk-workers=1.")
        args.chunk_workers = 1
    if not args.no_network:
        print("[warn] --no-network=false was supplied, but this script still does not crawl/fetch/discover URLs.")
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    os.environ["EMBED_MODEL"] = args.embed_model
    os.environ["EMBED_DEVICE"] = args.embed_device
    os.environ["EMBED_BACKEND"] = args.embed_backend

    client: QdrantClient | None = None
    if args.skip_existing and not args.dry_run:
        client = QdrantClient(url=args.qdrant_url)
    if client is not None and args.skip_existing and _collection_exists(client, args.qdrant_collection) and not args.recreate_collection:
        info = client.get_collection(args.qdrant_collection)
        points_count = int(getattr(info, "points_count", 0) or 0)
        active_cleaning_profile = args.cleaning_profile or "raw"
        if points_count <= 0:
            print(f"[warn] Collection {args.qdrant_collection!r} exists but has no points; reindexing it instead of skipping.")
        else:
            print(
                json.dumps(
                    {
                        "status": "skipped_existing",
                        "collection_name": args.qdrant_collection,
                        "cleaning_profile": active_cleaning_profile,
                        "points_count": points_count,
                        "chunk_workers": args.chunk_workers,
                        "embed_batch_size": args.embed_batch_size,
                        "upsert_batch_size": args.upsert_batch_size,
                        "total_indexing_time_sec": time.perf_counter() - total_t0,
                    },
                    ensure_ascii=False,
                )
            )
            return

    docs = load_local_documents(
        Path(args.input_dir),
        Path(args.fallback_raw_dir) if args.fallback_raw_dir else None,
        max_docs=args.max_docs,
    )
    original_docs = docs
    print(f"Documents loaded: {len(docs)}")
    if args.max_docs:
        print(f"Limited documents by --max-docs: {len(docs)}")

    cleaning_profile = args.cleaning_profile or "raw"
    cleaned_docs, cleaning_results, cleaning_summary = _apply_cleaning(
        docs,
        profile=cleaning_profile,
        max_removed_ratio=args.max_cleaning_removed_ratio,
    )
    docs = cleaned_docs
    print(
        "Cleaning profile={profile} avg_removed={avg:.4f} p95_removed={p95:.4f} warnings_docs={warn}".format(
            profile=cleaning_profile,
            avg=cleaning_summary["avg_removed_char_ratio"],
            p95=cleaning_summary["p95_removed_char_ratio"],
            warn=cleaning_summary["docs_with_cleaning_warnings"],
        )
    )
    if args.save_cleaned_preview:
        saved = _save_cleaned_previews(
            original_docs,
            cleaning_results,
            profile=cleaning_profile,
            output_dir=Path(args.cleaned_preview_dir),
            max_docs=20,
        )
        print(f"Saved cleaned previews: {saved} documents -> {Path(args.cleaned_preview_dir) / cleaning_profile}")

    cfg = _cfg_from_args(args)
    chunk_t0 = time.perf_counter()
    chunks = _chunk_docs(
        docs,
        cfg,
        workers=args.chunk_workers,
        parallel=args.parallel_chunking,
        prefetch_docs=args.prefetch_docs,
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
    chunk_stats = _print_chunk_stats(chunks, len(docs), args.qdrant_collection)
    print(f"Chunking time: {chunking_time_sec:.2f}s")

    if args.dry_run:
        preview = {
            "used_input_dir": args.input_dir if Path(args.input_dir).exists() else args.fallback_raw_dir,
            "docs_loaded": len(docs),
            "cleaning_profile": cleaning_profile,
            "cleaning_stats": cleaning_summary,
            "chunks_count": len(chunks),
            "chunk_workers": args.chunk_workers,
            "embed_batch_size": args.embed_batch_size,
            "upsert_batch_size": args.upsert_batch_size,
            "chunking_time_sec": chunking_time_sec,
            "total_indexing_time_sec": time.perf_counter() - total_t0,
            "first_chunk": chunks[0] if chunks else None,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:4000])
        return

    print(f"Loading embedding model {args.embed_model!r} on {args.embed_device!r}...")
    from src.rag.core import STEmbedder

    if client is None:
        client = QdrantClient(url=args.qdrant_url)
    embedder = STEmbedder(
        args.embed_model,
        device=args.embed_device,
        batch_size=args.embed_batch_size,
        max_seq_length=args.embed_max_seq_length,
        normalize_embeddings=args.embed_normalize,
        backend=args.embed_backend,
        trust_remote_code=args.embed_trust_remote_code,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        query_instruction=args.query_instruction,
    )
    _ensure_collection(client, args.qdrant_collection, embedder.dim, args.recreate_collection)

    pending_points: list[PointStruct] = []
    upserted = 0
    encoded = 0
    embedding_time_sec = 0.0
    upsert_time_sec = 0.0
    embed_dim = embedder.dim
    embed_metadata = getattr(embedder, "metadata", {})
    try:
        for batch_no, batch in enumerate(_iter_batches(chunks, args.embed_batch_size), start=1):
            texts = [c.get("child_text") or c.get("text") or "" for c in batch]
            emb_t0 = time.perf_counter()
            vectors = embedder.embed_passages(texts, batch_size=args.embed_batch_size)
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
                    "chunk_workers": args.chunk_workers,
                    "embed_batch_size": args.embed_batch_size,
                    "upsert_batch_size": args.upsert_batch_size,
                }
                pending_points.append(PointStruct(id=point_id, vector=vector, payload=payload))
                if len(pending_points) >= args.upsert_batch_size:
                    up_t0 = time.perf_counter()
                    upserted += _flush_points(client, args.qdrant_collection, pending_points)
                    upsert_time_sec += time.perf_counter() - up_t0
                    print(f"Upserted {upserted}/{len(chunks)} points")
            del vectors
            del texts
            del batch
            if args.cuda_cleanup_every_batches > 0 and batch_no % args.cuda_cleanup_every_batches == 0:
                _cleanup_cuda_cache(f"embedding batch {batch_no}")

        if pending_points:
            up_t0 = time.perf_counter()
            upserted += _flush_points(client, args.qdrant_collection, pending_points)
            upsert_time_sec += time.perf_counter() - up_t0
            print(f"Upserted {upserted}/{len(chunks)} points")
    finally:
        pending_points.clear()
        del embedder
        _cleanup_cuda_cache("reindex finished")

    info = client.get_collection(args.qdrant_collection)
    total_indexing_time_sec = time.perf_counter() - total_t0
    print(
        json.dumps(
            {
                "status": "ok",
                "collection_name": args.qdrant_collection,
                "cleaning_profile": cleaning_profile,
                "vector_dim": embed_dim,
                "points_count": getattr(info, "points_count", None),
                "chunk_workers": args.chunk_workers,
                "embed_batch_size": args.embed_batch_size,
                "upsert_batch_size": args.upsert_batch_size,
                "embedding_workers": args.embedding_workers,
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
            },
            ensure_ascii=False,
        )
    )
    print("Done.")


if __name__ == "__main__":
    main()
