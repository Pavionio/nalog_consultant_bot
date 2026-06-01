"""Dense Qdrant semantic-search tool for the agent. NO reranker (embedder only).

Reuses the project building blocks: RAGConfig + STEmbedder + Retriever, forced to
dense / no-reranker so only the e5-large embedder is loaded on the GPU. Returns
raw chunk dicts ({id, score, text, payload}); the agent assigns global [n] numbers
and formats observations via the helpers below.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `src` importable when run as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag.core import RAGConfig, STEmbedder, Retriever  # noqa: E402

# Source codes the agent may filter on (from config/sources.yaml). Shown in the prompt.
KNOWN_SOURCE_CODES = [
    "pravo_nk1", "nalog_letters", "nalog_docs", "nalog_calendar",
    "minfin_commonlaw", "minfin_orgprofit", "minfin_fizprofit", "minfin_property",
    "minfin_indirect", "minfin_international", "minfin_special", "minfin_transfert",
    "minfin_foreign", "minfin_customs_value", "minfin_imposition",
]


def chunk_identity(chunk: Dict[str, Any]) -> tuple:
    p = chunk.get("payload") or {}
    return (p.get("source_code"), p.get("external_id"), p.get("chunk_i"))


def chunk_text(chunk: Dict[str, Any]) -> str:
    p = chunk.get("payload") or {}
    return str(chunk.get("text") or p.get("text") or p.get("chunk") or "")


def source_ref(chunk: Dict[str, Any]) -> str:
    sc, eid, ci = chunk_identity(chunk)
    return f"{sc} / {eid} / chunk {ci}"


def format_snippet(chunk: Dict[str, Any], n: int, snippet_chars: int) -> str:
    text = chunk_text(chunk).strip().replace("\n", " ")
    if len(text) > snippet_chars:
        text = text[:snippet_chars].rstrip() + "…"
    score = chunk.get("score")
    score_s = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
    return f"[{n}] ({source_ref(chunk)}){score_s}\n{text}"


# The agentic experiment runs on the ORIGINAL rag_chunks collection (bge-m3),
# because that is the collection the eval datasets' gold (source_code/external_id/
# chunk_i) was built from — and the same backend as the paper's single-shot
# baseline (retrieval matrix). The re-chunked optimal collection (e5-large) uses a
# different external_id scheme, so gold would not match. Embedder only, no reranker.
DEFAULT_COLLECTION = "rag_chunks"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


class SearchTool:
    def __init__(
        self,
        *,
        collection: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        per_call_top_k_max: int = 6,
        snippet_chars: int = 600,
    ) -> None:
        cfg = RAGConfig()
        # Dense + no reranker: Retriever.search/_qdrant_search never loads the reranker.
        self.cfg = dataclasses.replace(
            cfg, use_reranker=False, retrieval_mode="dense",
            qdrant_collection=collection, embed_model_name=embed_model,
        )
        self.embedder = STEmbedder(
            self.cfg.embed_model_name,
            device=self.cfg.embed_device,
            batch_size=self.cfg.embed_batch_size,
            max_seq_length=self.cfg.embed_max_seq_length,
            normalize_embeddings=self.cfg.embed_normalize,
            backend=self.cfg.embed_backend,
            trust_remote_code=self.cfg.embed_trust_remote_code,
            query_prefix=self.cfg.embed_query_prefix,
            passage_prefix=self.cfg.embed_passage_prefix,
            query_instruction=self.cfg.embed_query_instruction,
            passage_instruction=self.cfg.embed_passage_instruction,
        )
        self.retriever = Retriever(self.cfg, self.embedder)
        self.per_call_top_k_max = per_call_top_k_max
        self.snippet_chars = snippet_chars

    def run(self, query: str, top_k: int = 4, source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Dense search; returns up to top_k chunk dicts (optionally filtered by source)."""
        top_k = max(1, min(int(top_k or 4), self.per_call_top_k_max))
        query = str(query or "").strip()
        if not query:
            return []
        if source_code:
            # Fetch wider, then post-filter by exact-or-prefix source_code (so e.g.
            # "minfin" matches all minfin_* sources). Fall back to unfiltered results
            # if the filter is too narrow / invalid, so the agent never dead-ends.
            sc = str(source_code).strip()
            raw = self.retriever._qdrant_search(query, limit=top_k * 8)
            filtered = [
                r for r in raw
                if str((r.get("payload") or {}).get("source_code") or "").startswith(sc)
            ]
            return (filtered or raw)[:top_k]
        return self.retriever._qdrant_search(query, limit=top_k)


if __name__ == "__main__":  # smoke test
    tool = SearchTool()
    q = sys.argv[1] if len(sys.argv) > 1 else "срок уплаты налога по УСН за первый квартал"
    res = tool.run(q, top_k=4)
    print(f"query={q!r} -> {len(res)} chunks")
    for i, ch in enumerate(res, 1):
        print(format_snippet(ch, i, 200))
        print("-")
