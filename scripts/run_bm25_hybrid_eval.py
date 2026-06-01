"""Retrieval-only eval of dense / bm25 / hybrid(RRF) modes — NO reranker.

Loads the embedder + retriever once and evaluates every (dataset x mode x k)
in-process, mirroring run_eval_matrix's harness (match_mode="strict", default
collection rag_chunks) so the numbers are directly comparable to the published
baseline-dense retrieval tables.

Run:
    python scripts/run_bm25_hybrid_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.core import RAGConfig, STEmbedder, Retriever  # noqa: E402
from src.eval.eval import evaluate_dataset  # noqa: E402

DATASETS = [
    ("easy", "eval_dataset.jsonl"),
    ("hard", "eval_hard_dataset.jsonl"),
    ("superhard", "eval_superhard_dataset.jsonl"),
]
MODES = ["dense", "bm25", "hybrid"]
KS = [1, 5, 10]


def main() -> None:
    cfg = RAGConfig()
    print(
        f"collection={cfg.qdrant_collection} embed={cfg.embed_model_name} "
        f"device={cfg.embed_device} bm25_fetch_k={cfg.bm25_fetch_k} "
        f"hybrid_fetch_k={cfg.hybrid_fetch_k} hybrid_rrf_k={cfg.hybrid_rrf_k}",
        flush=True,
    )
    embedder = STEmbedder(
        cfg.embed_model_name,
        device=cfg.embed_device,
        batch_size=cfg.embed_batch_size,
        max_seq_length=cfg.embed_max_seq_length,
        normalize_embeddings=cfg.embed_normalize,
        backend=cfg.embed_backend,
        trust_remote_code=cfg.embed_trust_remote_code,
        query_prefix=cfg.embed_query_prefix,
        passage_prefix=cfg.embed_passage_prefix,
        query_instruction=cfg.embed_query_instruction,
        passage_instruction=cfg.embed_passage_instruction,
    )
    retriever = Retriever(cfg, embedder)

    rows = []
    for ds_label, ds_path in DATASETS:
        if not Path(ds_path).exists():
            print(f"SKIP missing dataset {ds_path}", flush=True)
            continue
        for mode in MODES:
            for k in KS:
                rep = evaluate_dataset(
                    ds_path,
                    k=k,
                    use_llm_judge=False,
                    retrieval_mode=mode,
                    embedder=embedder,
                    retriever=retriever,
                    match_mode="strict",
                    method=mode,
                )
                m = rep["metrics"]
                row = {
                    "dataset": ds_label,
                    "mode": mode,
                    "k": k,
                    "hit": m.get(f"recall@{k}"),
                    "mrr": m.get(f"mrr@{k}"),
                    "ndcg": m.get(f"ndcg@{k}"),
                    "precision": m.get(f"precision@{k}"),
                    "hardneg": m.get(f"hard_negative_rate@{k}"),
                    "n_docs_bm25": getattr(retriever, "_bm25_index", {}) and retriever._bm25_index.get("n_docs"),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path("data/metrics/bm25_hybrid_results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
