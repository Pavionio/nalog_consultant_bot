"""
Run full eval matrix in a single process — models loaded once.

Two-phase approach:
  Phase 1 (LLM): precompute all query transformations (hyde/rewrite) while llama.cpp is running
  Phase 2 (retrieval): run search + rerank after llama.cpp is stopped (frees GPU for reranker)

Usage:
    python scripts/run_eval_matrix.py --k 5
    python scripts/run_eval_matrix.py --k 5 --methods baseline reranker
    python scripts/run_eval_matrix.py --k 5 10 --datasets eval_dataset.jsonl eval_hard_dataset.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import subprocess
import time
from tqdm import tqdm
from src.rag.core import RAGConfig, STEmbedder, Retriever, LlamaCppChatClient, rewrite_query, hyde_query
from src.eval.eval import evaluate_dataset, _append_log, _print_log_table, EVAL_LOG, EVAL_LOG_SOURCE, load_jsonl

DOCKER_COMPOSE  = "docker/docker-compose.yml"
LLAMA_SERVICE   = "llamacpp"
PRECOMPUTE_CACHE = "data/metrics/precomputed_queries.json"


def _docker(action: str) -> None:
    cmd = ["docker", "compose", "-f", DOCKER_COMPOSE, "--env-file", ".env", action, LLAMA_SERVICE]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if action == "start":
        print("  waiting for llama.cpp to be ready...")
        time.sleep(10)


DEFAULT_DATASETS = [
    "eval_dataset.jsonl",
    "eval_hard_dataset.jsonl",
    "eval_superhard_dataset.jsonl",
]

METHODS = [
    # (label,          use_rewrite, use_hyde, use_reranker)
    ("baseline",       False,       False,    False),
    ("rewrite",        True,        False,    False),
    ("hyde",           False,       True,     False),
    ("reranker",       False,       False,    True),
    ("hyde+reranker",  False,       True,     True),
]


def _needs_llm(method: Tuple) -> bool:
    _, use_rewrite, use_hyde, _ = method
    return use_rewrite or use_hyde


def _needs_reranker(method: Tuple) -> bool:
    return method[3]


def precompute_queries(
    llm: LlamaCppChatClient,
    datasets: List[str],
    methods_to_run: List[Tuple],
) -> Dict[str, Dict[str, str]]:
    """
    Returns {(dataset, label): {original_query: transformed_query}}
    Only for methods that use LLM transformation.
    """
    llm_methods = [m for m in methods_to_run if _needs_llm(m)]
    if not llm_methods:
        return {}

    results: Dict[str, Dict[str, str]] = {}

    for dataset in datasets:
        if not Path(dataset).exists():
            continue
        ds = load_jsonl(dataset)
        queries = [str(item["query"]) for item in ds]

        for label, use_rewrite, use_hyde, _ in llm_methods:
            key = f"{dataset}|{label}"
            transform_fn = hyde_query if use_hyde else rewrite_query
            transformed = {}

            desc = f"LLM precompute [{label}] {Path(dataset).stem}"
            for q in tqdm(queries, desc=desc, unit="q"):
                transformed[q] = transform_fn(llm, q)

            results[key] = transformed
            tqdm.write(f"  precomputed {len(transformed)} queries for {label} / {Path(dataset).name}")

    return results


def _save_run(report, out_dir, label, dataset_label, k, log_path):
    out_path = Path(out_dir) / f"report_{label}_{dataset_label}_k{k}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    m = report["metrics"]
    n = len(report["detailed"])
    model_label = f"{label} {dataset_label} k{k}"

    log_row = {
        "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":       model_label,
        "dataset":     f"{dataset_label}.jsonl",
        "k":           k,
        "n":           n,
        "hit_rate":    m.get(f"recall@{k}"),
        "mrr":         m.get(f"mrr@{k}"),
        "ndcg":        m.get(f"ndcg@{k}"),
        "precision":   m.get(f"precision@{k}"),
        "judge_faith": None,
        "judge_rel":   None,
    }
    _append_log(log_row, log_path)

    if m.get("per_source"):
        source_log = Path(log_path).parent / Path(EVAL_LOG_SOURCE).name
        for sc, sm in m["per_source"].items():
            _append_log({
                "timestamp":   log_row["timestamp"],
                "model":       model_label,
                "dataset":     log_row["dataset"],
                "k":           k,
                "source_code": sc,
                **sm,
            }, str(source_log))

    hit = m.get(f"recall@{k}", 0)
    mrr = m.get(f"mrr@{k}", 0)
    tqdm.write(f"  {model_label}: hit@{k}={hit:.3f}  mrr@{k}={mrr:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--k", nargs="+", type=int, default=[5])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Methods: baseline rewrite hyde reranker hyde+reranker")
    ap.add_argument("--log", default=EVAL_LOG)
    ap.add_argument("--out-dir", default="data/metrics")
    ap.add_argument("--skip-precompute", action="store_true",
                    help="Skip LLM phase, load precomputed queries from cache")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cfg = RAGConfig()
    methods_to_run = [m for m in METHODS if args.methods is None or m[0] in args.methods]
    datasets = [d for d in args.datasets if Path(d).exists()]

    # ─────────────────────────────────────────────
    # Phase 1: LLM precomputation (llama.cpp ON)
    # ─────────────────────────────────────────────
    precomputed: Dict[str, Dict[str, str]] = {}
    cache_path = Path(args.out_dir) / Path(PRECOMPUTE_CACHE).name

    if args.skip_precompute:
        if cache_path.exists():
            print(f"\n=== Loading precomputed queries from cache: {cache_path} ===")
            with open(cache_path, encoding="utf-8") as f:
                precomputed = json.load(f)
            print(f"  loaded {len(precomputed)} entries")
        else:
            print(f"[warn] cache not found at {cache_path}, running LLM phase anyway")
            args.skip_precompute = False

    if not args.skip_precompute and any(_needs_llm(m) for m in methods_to_run):
        print("\n=== Phase 1: LLM precomputation (llama.cpp should be running) ===")
        llm = LlamaCppChatClient(cfg)
        precomputed = precompute_queries(llm, datasets, methods_to_run)
        del llm
        # save cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(precomputed, f, ensure_ascii=False)
        print(f"LLM precomputation done. Cache saved to {cache_path}")

    # ─────────────────────────────────────────────
    # Stop llama.cpp if reranker needed
    # ─────────────────────────────────────────────
    llama_stopped = False
    if any(_needs_reranker(m) for m in methods_to_run):
        print("\n=== Stopping llama.cpp to free GPU memory for reranker ===")
        _docker("stop")
        llama_stopped = True

    # ─────────────────────────────────────────────
    # Load embedder + reranker
    # ─────────────────────────────────────────────
    print("\nLoading embedder...")
    embedder = STEmbedder(cfg.embed_model_name)
    retriever = Retriever(cfg, embedder)

    if any(_needs_reranker(m) for m in methods_to_run):
        print("Loading reranker...")
        retriever._get_reranker()

    # ─────────────────────────────────────────────
    # Phase 2: retrieval + rerank
    # ─────────────────────────────────────────────
    print("\n=== Retrieval phase ===")
    runs = [
        (dataset, k, method)
        for dataset in datasets
        for k in args.k
        for method in methods_to_run
    ]

    pbar = tqdm(runs, desc="matrix", unit="run")
    for dataset, k, (label, use_rewrite, use_hyde, use_reranker) in pbar:
        dataset_label = Path(dataset).stem.replace("eval_", "").replace("_dataset", "")
        pbar.set_postfix_str(f"{label} {dataset_label} k{k}")

        precomputed_queries = precomputed.get(f"{dataset}|{label}")

        report = evaluate_dataset(
            dataset,
            k=k,
            use_llm_judge=False,
            use_rewrite=use_rewrite,
            use_hyde=use_hyde,
            use_reranker=use_reranker,
            embedder=embedder,
            retriever=retriever,
            llm=None,
            precomputed_queries=precomputed_queries,
        )

        _save_run(report, args.out_dir, label, dataset_label, k, args.log)

    # ─────────────────────────────────────────────
    # Restart llama.cpp if it was stopped
    # ─────────────────────────────────────────────
    if llama_stopped:
        print("\n=== Restarting llama.cpp ===")
        _docker("start")

    print("\n")
    _print_log_table(args.log)


if __name__ == "__main__":
    main()
