"""
Precompute rewrite/HyDE queries for all eval datasets while the LLM server is on.

Default target is LM Studio/OpenAI-compatible server:
    http://localhost:1234, model openai/gpt-oss-20b

After this script finishes, the LLM server can be stopped and cached
experiments can be run with scripts/run_cached_experiments.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
DEFAULT_LM_STUDIO_MODEL = "openai/gpt-oss-20b"
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

LLM_PRECOMPUTE_METHODS = {"rewrite", "hyde"}


def discover_datasets() -> List[str]:
    ordered = [p for p in DEFAULT_DATASETS if Path(p).exists()]
    seen = set(ordered)
    extra = sorted(
        str(p)
        for p in Path(".").glob("eval*_dataset.jsonl")
        if str(p) not in seen
    )
    return ordered + extra


def configure_no_proxy_for_url(url: str) -> None:
    host = url.removeprefix("http://").removeprefix("https://").split("/", 1)[0].split(":", 1)[0]
    if host in {"", "localhost", "127.0.0.1"}:
        hosts = ["localhost", "127.0.0.1"]
    else:
        hosts = [host]

    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [part.strip() for part in current.split(",") if part.strip()]
    for item in hosts:
        if item not in parts:
            parts.append(item)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def strict_transform(llm, label: str, query: str, rewrite_prompt: str, hyde_prompt: str) -> str:
    if label == "rewrite":
        prompt = rewrite_prompt.format(query=query)
        result = llm.chat([{"role": "user", "content": prompt}], max_tokens=350).strip()
        if len(result) < 10:
            raise RuntimeError(f"rewrite returned too short result for query: {query[:120]}")
        return result

    if label == "hyde":
        prompt = hyde_prompt.format(query=query)
        result = llm.chat([{"role": "user", "content": prompt}], max_tokens=300).strip()
        if len(result) < 20:
            raise RuntimeError(f"HyDE returned too short result for query: {query[:120]}")
        return result

    raise ValueError(f"Unsupported LLM transform method: {label}")


def precompute_queries_strict(
    llm,
    datasets: List[str],
    methods_to_run: List[Tuple],
    load_jsonl,
    rewrite_prompt: str,
    hyde_prompt: str,
) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}

    for dataset in datasets:
        ds = load_jsonl(dataset)
        queries = [str(item["query"]) for item in ds]

        for label, use_rewrite, use_hyde, _ in methods_to_run:
            if label not in LLM_PRECOMPUTE_METHODS:
                continue

            key = f"{dataset}|{label}"
            transformed: Dict[str, str] = {}
            changed = 0
            examples: List[Tuple[str, str]] = []

            desc = f"LLM precompute [{label}] {Path(dataset).stem}"
            for q in tqdm(queries, desc=desc, unit="q"):
                out = strict_transform(llm, label, q, rewrite_prompt, hyde_prompt)
                transformed[q] = out
                if out.strip() != q.strip():
                    changed += 1
                    if len(examples) < 2:
                        examples.append((q, out))

            results[key] = transformed
            unchanged = len(transformed) - changed
            change_rate = changed / max(1, len(transformed))
            tqdm.write(
                f"  precomputed {len(transformed)} queries for {label} / {Path(dataset).name}; "
                f"changed={changed}, unchanged={unchanged}, change_rate={change_rate:.1%}"
            )
            for before, after in examples:
                tqdm.write(f"    example before: {before[:160]}")
                tqdm.write(f"    example after:  {after[:160]}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute rewrite/HyDE query cache for eval matrix")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["rewrite", "hyde"],
                    help="Methods that need LLM transforms")
    ap.add_argument("--out-dir", default="data/metrics")
    ap.add_argument("--llm-base-url", default=DEFAULT_LM_STUDIO_URL)
    ap.add_argument("--llm-model", default=DEFAULT_LM_STUDIO_MODEL)
    args = ap.parse_args()

    datasets: List[str] = args.datasets or discover_datasets()
    missing = [d for d in datasets if not Path(d).exists()]
    if missing:
        raise SystemExit(f"Dataset files not found: {', '.join(missing)}")
    if not datasets:
        raise SystemExit("No eval datasets found. Expected files like eval_dataset.jsonl.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.out_dir) / Path(PRECOMPUTE_CACHE).name

    configure_no_proxy_for_url(args.llm_base_url)
    from dotenv import load_dotenv
    load_dotenv()

    from src.rag.core import RAGConfig, LlamaCppChatClient, REWRITE_PROMPT, HYDE_PROMPT
    from src.eval.eval import load_jsonl
    from scripts.run_eval_matrix import METHODS

    methods_to_run = [m for m in METHODS if m[0] in set(args.methods)]
    if not methods_to_run:
        raise SystemExit(f"No known LLM methods selected. Available: {', '.join(ALL_METHODS)}")

    cfg = dataclasses.replace(
        RAGConfig(),
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
    )

    print("Precomputing rewrite/HyDE queries:")
    print(f"  datasets: {', '.join(datasets)}")
    print(f"  methods: {', '.join(m[0] for m in methods_to_run)}")
    print(f"  llm_base_url: {args.llm_base_url}")
    print(f"  llm_model: {args.llm_model}")
    print(f"  cache: {cache_path}")

    llm = LlamaCppChatClient(cfg)
    try:
        probe = llm.chat([{"role": "user", "content": "Ответь одним словом: OK"}], max_tokens=8).strip()
    except Exception as exc:
        raise SystemExit(
            f"LLM server is not reachable at {args.llm_base_url} with model {args.llm_model}: {exc}"
        ) from exc
    print(f"  llm_probe: {probe[:80]}")

    precomputed = precompute_queries_strict(
        llm,
        datasets,
        methods_to_run,
        load_jsonl,
        REWRITE_PROMPT,
        HYDE_PROMPT,
    )
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(precomputed, f, ensure_ascii=False)

    print(f"Saved {len(precomputed)} dataset/method cache entries to {cache_path}")


if __name__ == "__main__":
    main()
