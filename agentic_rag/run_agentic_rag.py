"""CLI: run the agentic RAG on a single question and print the full trace.

Example:
  python -m agentic_rag.run_agentic_rag \
    --question "у нас ооо на усн, можно ли учесть в расходах штраф контрагенту по суду?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic_rag.llm_client import GPTOSSChatClient, DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402
from agentic_rag.search_tool import SearchTool, DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL  # noqa: E402
from agentic_rag.agent import AgenticRAG  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic RAG single-question runner")
    ap.add_argument("--question", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", default="low", choices=["off", "low", "medium", "high"])
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--per-call-top-k", type=int, default=4)
    ap.add_argument("--snippet-chars", type=int, default=600)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--json-out", action="store_true", help="Print the full result as JSON instead of pretty text")
    args = ap.parse_args()

    client = GPTOSSChatClient(base_url=args.base_url, model=args.model, reasoning_effort=args.reasoning_effort)
    tool = SearchTool(collection=args.collection, embed_model=args.embed_model, snippet_chars=args.snippet_chars)
    agent = AgenticRAG(client, tool, max_iterations=args.max_iters, per_call_top_k=args.per_call_top_k)

    result = agent.answer(args.question)

    if args.json_out:
        # Drop heavy raw chunk payloads from the JSON dump.
        slim = {k: v for k, v in result.items() if k != "all_chunks"}
        print(json.dumps(slim, ensure_ascii=False, indent=2))
        return

    print("=" * 70)
    print("ВОПРОС:", result["question"])
    print("=" * 70)
    for step in result["trace"]:
        if step["action"] == "search":
            tag = " [повтор/пропуск]" if step.get("skipped") else ""
            print(f"  [шаг {step['iter']}] SEARCH{tag}: {step.get('query')!r}"
                  + (f"  source={step['source_code']}" if step.get("source_code") else "")
                  + (f"  -> {step.get('n_results')} рез., новых {step.get('n_new')}, [n]={step.get('global_indices')}"
                     if not step.get("skipped") else ""))
        elif step["action"] == "answer":
            print(f"  [шаг {step['iter']}] ANSWER (inline)")
        else:
            print(f"  [шаг {step['iter']}] MALFORMED JSON")
    print("-" * 70)
    print(f"поисков: {result['n_searches']} | уникальных чанков: {result['n_unique_chunks']} | latency: {result['latency_s']}с")
    print("-" * 70)
    print("ИТОГОВЫЙ ОТВЕТ:\n")
    print(result["answer"])
    print("-" * 70)
    print("ИСТОЧНИКИ:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s.get('source_code')} / {s.get('external_id')} / chunk {s.get('chunk_i')}"
              + (f"  {s.get('url')}" if s.get("url") else ""))


if __name__ == "__main__":
    main()
