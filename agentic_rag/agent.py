"""AgenticRAG: a JSON-ReAct loop where gpt-oss-20b iteratively searches Qdrant.

The agent emits one JSON action per turn ({"action":"search"|"answer", ...}).
Search results are deduped and given stable global [n] numbers. The final answer
is always regenerated from the deduped chunk set via build_context, so citation
numbering and strict-citation formatting are consistent regardless of how the
agent phrased its inline answer.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag.core import build_context  # noqa: E402
from agentic_rag.llm_client import GPTOSSChatClient  # noqa: E402
from agentic_rag.search_tool import SearchTool, chunk_identity, format_snippet  # noqa: E402
from agentic_rag.prompts import AGENT_SYSTEM_PROMPT, FINAL_ANSWER_PROMPT, JSON_REPAIR_HINT  # noqa: E402

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def parse_json_action(raw: str) -> Optional[Dict[str, Any]]:
    """Extract one {"action": ...} object from the model output, or None."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    for candidate in (text, (_JSON_OBJ_RE.search(text).group(0) if _JSON_OBJ_RE.search(text) else None)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("action") in {"search", "answer"}:
            return obj
    return None


class AgenticRAG:
    def __init__(
        self,
        client: GPTOSSChatClient,
        search_tool: SearchTool,
        *,
        max_iterations: int = 6,
        per_call_top_k: int = 4,
        max_unique_chunks: int = 24,
        max_json_repairs: int = 2,
        agent_max_tokens: int = 1400,
        final_max_tokens: int = 1500,
    ) -> None:
        self.client = client
        self.tool = search_tool
        self.cfg = search_tool.cfg
        self.max_iterations = max_iterations
        self.per_call_top_k = per_call_top_k
        self.max_unique_chunks = max_unique_chunks
        self.max_json_repairs = max_json_repairs
        self.agent_max_tokens = agent_max_tokens
        self.final_max_tokens = final_max_tokens

    def answer(self, question: str) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос пользователя:\n{question}"},
        ]
        unique: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()  # identity -> chunk
        index_of: Dict[tuple, int] = {}  # identity -> global [n]
        seen_queries: set = set()
        trace: List[Dict[str, Any]] = []
        n_searches = 0
        repairs = 0
        inline_answer: Optional[str] = None
        t0 = time.perf_counter()

        for it in range(self.max_iterations):
            raw = self.client.content_only(messages, max_tokens=self.agent_max_tokens)
            action = parse_json_action(raw)

            if action is None:
                repairs += 1
                trace.append({"iter": it, "action": "malformed", "raw": raw[:300]})
                if repairs > self.max_json_repairs:
                    break
                messages.append({"role": "user", "content": JSON_REPAIR_HINT})
                continue

            if action["action"] == "answer":
                inline_answer = str(action.get("content") or "")
                trace.append({"iter": it, "action": "answer", "content": inline_answer[:400]})
                break

            # action == "search"
            query = str(action.get("query") or "").strip()
            source_code = action.get("source_code") or None
            norm = query.lower()
            if not query or norm in seen_queries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Этот запрос уже выполнялся или пуст. Уточни запрос или дай ответ."})
                trace.append({"iter": it, "action": "search", "query": query, "skipped": True})
                continue

            seen_queries.add(norm)
            n_searches += 1
            results = self.tool.run(query, top_k=self.per_call_top_k, source_code=source_code)

            shown: List[tuple] = []  # (chunk, global_n) for this search
            n_new = 0
            for ch in results:
                ident = chunk_identity(ch)
                if ident not in index_of:
                    if len(index_of) >= self.max_unique_chunks:
                        continue  # at cap: don't introduce unnumbered chunks
                    index_of[ident] = len(index_of) + 1
                    unique[ident] = ch
                    n_new += 1
                shown.append((ch, index_of[ident]))

            if shown:
                obs = "\n\n".join(format_snippet(ch, n, self.tool.snippet_chars) for ch, n in shown)
            else:
                obs = "Ничего не найдено."
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Результаты поиска (фрагменты пронумерованы сквозно):\n{obs}"})
            trace.append({
                "iter": it, "action": "search", "query": query, "source_code": source_code,
                "n_results": len(results), "n_new": n_new,
                "global_indices": [n for _, n in shown],
            })

        # Always regenerate a clean, consistently-cited answer from the deduped chunks.
        chunks = list(unique.values())
        if chunks:
            context, sources = build_context(self.cfg, chunks)
            final = self.client.content_only(
                [
                    {"role": "system", "content": FINAL_ANSWER_PROMPT},
                    {"role": "user", "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context}"},
                ],
                max_tokens=self.final_max_tokens,
            )
        else:
            final = "В предоставленных документах нет достаточной информации для ответа."
            sources = []

        return {
            "question": question,
            "answer": final,
            "inline_answer": inline_answer,
            "sources": sources,
            "trace": trace,
            "n_searches": n_searches,
            "n_unique_chunks": len(unique),
            "all_chunks": chunks,
            "latency_s": round(time.perf_counter() - t0, 2),
        }
