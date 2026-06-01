from __future__ import annotations

import os
import time
import json
import math
import re
import html
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence

from dotenv import load_dotenv
load_dotenv()

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import numpy as np

from src.rag.embedders import EmbedderConfig, build_embedder, config_from_registry


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _default_torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_reranker_fp16() -> bool:
    return _default_torch_device() == "cuda"


_DEFAULT_RERANKER_FETCH_K = int(os.getenv("RAG_RERANKER_FETCH_K", os.getenv("RAG_RERANKER_TOP_K", "12")))
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")
CONTEXT_ORDER_MODES = {"rerank_123", "rerank_132"}


def _bm25_tokens(text: str) -> List[str]:
    return [m.group(0).lower().replace("ё", "е") for m in _TOKEN_RE.finditer(text or "")]

@dataclass(frozen=True)
class RAGConfig:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8080")
    llm_model: str = os.getenv("LLM", "Qwen2.5-7B-Instruct")  # llama.cpp может игнорировать

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # 0.0 = не фильтровать
    retrieval_mode: str = os.getenv("RAG_RETRIEVAL_MODE", "dense")  # dense / bm25 / hybrid
    bm25_fetch_k: int = int(os.getenv("RAG_BM25_FETCH_K", "50"))
    hybrid_fetch_k: int = int(os.getenv("RAG_HYBRID_FETCH_K", "50"))
    hybrid_rrf_k: int = int(os.getenv("RAG_HYBRID_RRF_K", "60"))

    # Prompting
    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "18000"))
    max_chunk_chars: int = int(os.getenv("RAG_MAX_CHUNK_CHARS", "2500"))
    context_order: str = os.getenv("RAG_CONTEXT_ORDER", "rerank_123")

    # LLM generation
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "900"))
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "120"))
    reasoning_effort: Optional[str] = os.getenv("LLM_REASONING_EFFORT") or None

    # Embeddings
    embed_model_name: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    embed_device: str = os.getenv("EMBED_DEVICE", "cuda")  # "cuda" / "cpu"
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "0"))  # 0 = model-registry default
    embed_max_seq_length: Optional[int] = int(os.getenv("EMBED_MAX_SEQ_LENGTH")) if os.getenv("EMBED_MAX_SEQ_LENGTH") else None
    embed_normalize: bool = _env_bool("EMBED_NORMALIZE", True)
    embed_backend: str = os.getenv("EMBED_BACKEND", "auto")
    embed_query_prefix: Optional[str] = os.getenv("EMBED_QUERY_PREFIX") or None
    embed_passage_prefix: Optional[str] = os.getenv("EMBED_PASSAGE_PREFIX") or None
    embed_query_instruction: Optional[str] = os.getenv("EMBED_QUERY_INSTRUCTION") or None
    embed_passage_instruction: Optional[str] = os.getenv("EMBED_PASSAGE_INSTRUCTION") or None
    embed_trust_remote_code: Optional[bool] = _env_bool("EMBED_TRUST_REMOTE_CODE", False) if os.getenv("EMBED_TRUST_REMOTE_CODE") is not None else None

    # Query rewriting / HyDE
    use_rewrite: bool = _env_bool("RAG_USE_REWRITE", False)
    use_hyde: bool = _env_bool("RAG_USE_HYDE", False)

    # Reranker
    use_reranker: bool = _env_bool("RAG_USE_RERANKER", False)
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    reranker_type: str = os.getenv("RAG_RERANKER_TYPE", "auto")
    reranker_fetch_k: int = _DEFAULT_RERANKER_FETCH_K
    reranker_top_k: int = _DEFAULT_RERANKER_FETCH_K  # backwards-compatible alias for reranker_fetch_k
    reranker_max_length: int = int(os.getenv("RAG_RERANKER_MAX_LENGTH", "1024"))
    reranker_batch_size: int = int(os.getenv("RAG_RERANKER_BATCH_SIZE", "8"))
    reranker_device: str = os.getenv("RAG_RERANKER_DEVICE", _default_torch_device())
    reranker_use_fp16: bool = _env_bool("RAG_RERANKER_USE_FP16", _default_reranker_fp16())
    reranker_normalize: bool = _env_bool("RAG_RERANKER_NORMALIZE", False)
    reranker_instruction: Optional[str] = os.getenv("RAG_RERANKER_INSTRUCTION") or None

    def __post_init__(self) -> None:
        if self.context_order not in CONTEXT_ORDER_MODES:
            raise ValueError(
                f"Unsupported context_order={self.context_order!r}; "
                f"expected one of {sorted(CONTEXT_ORDER_MODES)}"
            )
        # Keep old callers that pass reranker_top_k working while making
        # reranker_fetch_k the preferred field.
        if self.reranker_fetch_k != self.reranker_top_k:
            if self.reranker_fetch_k != _DEFAULT_RERANKER_FETCH_K and self.reranker_top_k == _DEFAULT_RERANKER_FETCH_K:
                object.__setattr__(self, "reranker_top_k", self.reranker_fetch_k)
            else:
                object.__setattr__(self, "reranker_fetch_k", self.reranker_top_k)
    
    
def load_reranker(model_name: str, device: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device, token=os.getenv("HF_TOKEN"))


def load_embedder(model_name: str):
    cfg = config_from_registry(
        model_name,
        device=os.getenv("EMBED_DEVICE", "cuda"),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        max_seq_length=int(os.getenv("EMBED_MAX_SEQ_LENGTH")) if os.getenv("EMBED_MAX_SEQ_LENGTH") else None,
        normalize_embeddings=_env_bool("EMBED_NORMALIZE", True),
        trust_remote_code=_env_bool("EMBED_TRUST_REMOTE_CODE", False) if os.getenv("EMBED_TRUST_REMOTE_CODE") is not None else None,
        backend=os.getenv("EMBED_BACKEND", "auto"),
        query_prefix=os.getenv("EMBED_QUERY_PREFIX"),
        passage_prefix=os.getenv("EMBED_PASSAGE_PREFIX"),
        query_instruction=os.getenv("EMBED_QUERY_INSTRUCTION"),
        passage_instruction=os.getenv("EMBED_PASSAGE_INSTRUCTION"),
    )
    return build_embedder(cfg)
    
@dataclass
class STEmbedder:
    model_name: str
    device: Optional[str] = None
    batch_size: Optional[int] = None
    max_seq_length: Optional[int] = None
    normalize_embeddings: Optional[bool] = None
    backend: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    query_prefix: Optional[str] = None
    passage_prefix: Optional[str] = None
    query_instruction: Optional[str] = None
    passage_instruction: Optional[str] = None

    def __post_init__(self) -> None:
        cfg = config_from_registry(
            self.model_name,
            device=self.device or os.getenv("EMBED_DEVICE", "cuda"),
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            normalize_embeddings=self.normalize_embeddings,
            trust_remote_code=self.trust_remote_code,
            backend=self.backend,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
            query_instruction=self.query_instruction,
            passage_instruction=self.passage_instruction,
        )
        self.config: EmbedderConfig = cfg
        self.model = build_embedder(cfg)
        self.metadata = getattr(self.model, "metadata", {})

    @property
    def dim(self) -> int:
        return int(self.model.dim)

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 64) -> List[List[float]]:
        """
        Возвращает list[vector] где vector = list[float], удобно для Qdrant.
        Backward-compatible alias: old callers get passage-mode embeddings.
        """
        return self.embed_passages(texts, batch_size=batch_size)

    def embed_passages(self, texts: Sequence[str], *, batch_size: Optional[int] = None) -> List[List[float]]:
        old_batch = self.model.config.batch_size
        if batch_size is not None:
            self.model.config.batch_size = int(batch_size)
        try:
            embs = self.model.encode_passages([str(t) for t in texts if t is not None])
        finally:
            self.model.config.batch_size = old_batch
        out = np.asarray(embs, dtype=np.float32).tolist()
        del embs
        return out

    def embed_queries(self, texts: Sequence[str], *, batch_size: Optional[int] = None) -> List[List[float]]:
        old_batch = self.model.config.batch_size
        if batch_size is not None:
            self.model.config.batch_size = int(batch_size)
        try:
            embs = self.model.encode_queries([str(t) for t in texts if t is not None])
        finally:
            self.model.config.batch_size = old_batch
        out = np.asarray(embs, dtype=np.float32).tolist()
        del embs
        return out

    def embed_query(self, text: str) -> List[float]:
        text = str(text or "").strip()
        if not text:
            raise ValueError("embed_query: пустая строка запроса")
        return self.embed_queries([text], batch_size=1)[0]
    
    
class Retriever:
    def __init__(self, cfg: RAGConfig, embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.qdrant = QdrantClient(url=cfg.qdrant_url)
        self._reranker = None
        self._bm25_index: Optional[Dict[str, Any]] = None
        self.last_timings: Dict[str, float] = {}

    def _get_reranker(self):
        if self._reranker is None:
            from src.rag.rerankers import build_reranker
            self._reranker = build_reranker(self.cfg)
        return self._reranker

    def _payload_text(self, payload: Dict[str, Any]) -> str:
        return (
            payload.get("child_text")
            or payload.get("text")
            or payload.get("chunk")
            or payload.get("parent_text")
            or ""
        )

    def _load_bm25_index(self) -> Dict[str, Any]:
        if self._bm25_index is not None:
            return self._bm25_index

        docs: List[Dict[str, Any]] = []
        tokenized: List[List[str]] = []
        term_freqs: List[Counter[str]] = []
        doc_freq: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = self.qdrant.scroll(
                collection_name=self.cfg.qdrant_collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                text = self._payload_text(payload)
                if not text:
                    continue
                tokens = _bm25_tokens(text)
                if not tokens:
                    continue
                docs.append({"id": p.id, "text": text, "payload": payload})
                tokenized.append(tokens)
                term_freqs.append(Counter(tokens))
                doc_freq.update(set(tokens))
            if offset is None:
                break

        n_docs = len(docs)
        avgdl = sum(len(toks) for toks in tokenized) / max(1, n_docs)
        idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        self._bm25_index = {
            "docs": docs,
            "tokenized": tokenized,
            "term_freqs": term_freqs,
            "idf": idf,
            "avgdl": avgdl,
            "n_docs": n_docs,
        }
        return self._bm25_index

    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        tb = time.perf_counter()
        index = self._load_bm25_index()
        query_terms = _bm25_tokens(query)
        if not query_terms or not index["docs"]:
            self._last_bm25_search_latency_ms = (time.perf_counter() - tb) * 1000.0
            return []

        k1 = 1.5
        b = 0.75
        avgdl = float(index["avgdl"] or 1.0)
        idf: Dict[str, float] = index["idf"]
        scores: List[Tuple[float, int]] = []
        q_terms = list(dict.fromkeys(query_terms))
        for i, tokens in enumerate(index["tokenized"]):
            counts = index["term_freqs"][i]
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                score += idf.get(term, 0.0) * (tf * (k1 + 1.0) / denom)
            if score > 0.0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, i in scores[:limit]:
            doc = index["docs"][i]
            out.append(
                {
                    "id": doc["id"],
                    "score": float(score),
                    "score_bm25": float(score),
                    "text": doc["text"],
                    "payload": doc["payload"],
                }
            )
        self._last_bm25_search_latency_ms = (time.perf_counter() - tb) * 1000.0
        return out

    def _qdrant_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        te = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        self._last_embedding_query_latency_ms = (time.perf_counter() - te) * 1000.0
        ts = time.perf_counter()
        resp = self.qdrant.query_points(
            collection_name=self.cfg.qdrant_collection,
            query=qvec,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        self._last_dense_search_latency_ms = (time.perf_counter() - ts) * 1000.0
        out = []
        for p in resp.points:
            if p.score < self.cfg.score_threshold:
                continue
            payload = p.payload or {}
            text = payload.get("text") or payload.get("chunk") or ""
            out.append(
                {
                    "id": p.id,
                    "score": float(p.score) if p.score is not None else None,
                    "text": text,
                    "payload": payload,
                }
            )
        return out

    def _is_parent_child(self, item: Dict[str, Any]) -> bool:
        payload = item.get("payload") or {}
        return payload.get("chunk_method") == "parent_child" or payload.get("chunk_role") == "child"

    def _as_parent_child_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(item.get("payload") or {})
        child_text = payload.get("child_text") or payload.get("text") or item.get("text") or ""
        parent_text = payload.get("parent_text") or child_text
        if not payload.get("parent_text"):
            payload["parent_text_missing"] = True
        child_score = item.get("score")
        payload.update(
            {
                "matched_child_text": child_text,
                "matched_child_score": child_score,
                "score_parent": child_score,
                "score_child": child_score,
                "child_text": child_text,
                "parent_text": parent_text,
            }
        )
        result = dict(item)
        result["text"] = parent_text
        result["payload"] = payload
        return result

    def _group_parent_child(self, items: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not items or not any(self._is_parent_child(x) for x in items):
            return items[:limit] if limit is not None else items
        grouped: Dict[str, Dict[str, Any]] = {}
        passthrough: List[Dict[str, Any]] = []
        for item in items:
            if not self._is_parent_child(item):
                passthrough.append(item)
                continue
            payload = item.get("payload") or {}
            parent_id = payload.get("parent_id") or f"{payload.get('source_code')}:{payload.get('external_id')}:{payload.get('parent_i')}"
            current = grouped.get(parent_id)
            if current is None or float(item.get("score") or 0.0) > float(current.get("score") or 0.0):
                grouped[parent_id] = self._as_parent_child_result(item)
        out = list(grouped.values()) + passthrough
        out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return out[:limit] if limit is not None else out

    def search(self, query: str) -> List[Dict[str, Any]]:
        self.last_timings = {}
        t0 = time.perf_counter()
        mode = getattr(self.cfg, "retrieval_mode", "dense")
        if mode == "bm25":
            tb = time.perf_counter()
            out = self._bm25_search(query, limit=max(self.cfg.top_k, self.cfg.bm25_fetch_k))
            out = self._group_parent_child(out, self.cfg.top_k)
            bm25_ms = (time.perf_counter() - tb) * 1000.0
            self.last_timings = {
                "embedding_query_latency_ms": 0.0,
                "dense_search_latency_ms": 0.0,
                "dense_retrieval_latency_ms": 0.0,
                "bm25_search_latency_ms": getattr(self, "_last_bm25_search_latency_ms", bm25_ms),
                "hybrid_fusion_latency_ms": 0.0,
                "rerank_latency_ms": 0.0,
                "total_retrieval_latency_ms": (time.perf_counter() - t0) * 1000.0,
            }
            return out
        if mode == "hybrid":
            out = self._search_hybrid(query)
            self.last_timings["total_retrieval_latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return out
        if self.cfg.use_reranker:
            out = self._search_rerank(query)
            self.last_timings["total_retrieval_latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return out
        td = time.perf_counter()
        out = self._qdrant_search(query, limit=self.cfg.top_k)
        out = self._group_parent_child(out, self.cfg.top_k)
        dense_ms = (time.perf_counter() - td) * 1000.0
        self.last_timings = {
            "embedding_query_latency_ms": getattr(self, "_last_embedding_query_latency_ms", 0.0),
            "dense_search_latency_ms": getattr(self, "_last_dense_search_latency_ms", dense_ms),
            "dense_retrieval_latency_ms": dense_ms,
            "bm25_search_latency_ms": 0.0,
            "hybrid_fusion_latency_ms": 0.0,
            "rerank_latency_ms": 0.0,
            "total_retrieval_latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return out

    def _search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        fetch_k = max(self.cfg.top_k, int(getattr(self.cfg, "hybrid_fetch_k", 50)))
        td = time.perf_counter()
        dense = self._qdrant_search(query, limit=fetch_k)
        dense_ms = (time.perf_counter() - td) * 1000.0
        bm25 = self._bm25_search(query, limit=fetch_k)

        tf = time.perf_counter()
        rrf_k = float(getattr(self.cfg, "hybrid_rrf_k", 60))
        fused: Dict[str, Dict[str, Any]] = {}

        def key_for(item: Dict[str, Any]) -> str:
            payload = item.get("payload") or {}
            return str(item.get("id") or payload.get("point_id") or payload.get("id"))

        for rank, item in enumerate(dense, start=1):
            key = key_for(item)
            cur = fused.setdefault(key, dict(item))
            cur["score_dense"] = item.get("score")
            cur["rank_dense"] = rank
            cur["score_rrf"] = float(cur.get("score_rrf") or 0.0) + 1.0 / (rrf_k + rank)

        for rank, item in enumerate(bm25, start=1):
            key = key_for(item)
            cur = fused.setdefault(key, dict(item))
            cur["score_bm25"] = item.get("score_bm25", item.get("score"))
            cur["rank_bm25"] = rank
            cur["score_rrf"] = float(cur.get("score_rrf") or 0.0) + 1.0 / (rrf_k + rank)

        result = []
        for item in fused.values():
            item["score"] = float(item.get("score_rrf") or 0.0)
            item["retrieval_mode"] = "hybrid"
            item["hybrid_fusion"] = "rrf"
            result.append(item)
        result.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        result = self._group_parent_child(result, self.cfg.top_k)
        fusion_ms = (time.perf_counter() - tf) * 1000.0
        self.last_timings = {
            "embedding_query_latency_ms": getattr(self, "_last_embedding_query_latency_ms", 0.0),
            "dense_search_latency_ms": getattr(self, "_last_dense_search_latency_ms", dense_ms),
            "dense_retrieval_latency_ms": dense_ms,
            "bm25_search_latency_ms": getattr(self, "_last_bm25_search_latency_ms", 0.0),
            "hybrid_fusion_latency_ms": fusion_ms,
            "rerank_latency_ms": 0.0,
            "total_retrieval_latency_ms": dense_ms + getattr(self, "_last_bm25_search_latency_ms", 0.0) + fusion_ms,
        }
        return result

    def _search_rerank(self, query: str) -> List[Dict[str, Any]]:
        # 1. fetch more candidates from Qdrant
        fetch_k = int(getattr(self.cfg, "reranker_fetch_k", self.cfg.reranker_top_k))
        td = time.perf_counter()
        candidates = self._qdrant_search(query, limit=fetch_k)
        dense_ms = (time.perf_counter() - td) * 1000.0
        if not candidates:
            self.last_timings = {
                "embedding_query_latency_ms": getattr(self, "_last_embedding_query_latency_ms", 0.0),
                "dense_search_latency_ms": getattr(self, "_last_dense_search_latency_ms", dense_ms),
                "dense_retrieval_latency_ms": dense_ms,
                "bm25_search_latency_ms": 0.0,
                "hybrid_fusion_latency_ms": 0.0,
                "rerank_latency_ms": 0.0,
                "total_retrieval_latency_ms": dense_ms,
            }
            return []

        # 2. rerank with configured backend
        reranker = self._get_reranker()
        tr = time.perf_counter()
        scores = reranker.score(query, [
            (c.get("payload") or {}).get("child_text") or c["text"] for c in candidates
        ])
        rerank_ms = (time.perf_counter() - tr) * 1000.0

        # 3. sort by reranker score, keep top_k
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = []
        for rerank_score, chunk in ranked:
            chunk = dict(chunk)
            dense_score = chunk.get("score")
            chunk["score_dense"] = float(dense_score) if dense_score is not None else None
            chunk["score_rerank"] = float(rerank_score)
            chunk["rerank_score"] = float(rerank_score)
            chunk["score"] = float(rerank_score)
            chunk["reranker_model"] = self.cfg.reranker_model
            chunk["reranker_type"] = getattr(reranker, "reranker_type", self.cfg.reranker_type)
            chunk["reranker_fetch_k"] = fetch_k
            chunk["reranker_max_length"] = self.cfg.reranker_max_length
            chunk["reranker_effective_max_length"] = getattr(reranker, "max_length", self.cfg.reranker_max_length)
            result.append(chunk)
        result = self._group_parent_child(result, self.cfg.top_k)
        self.last_timings = {
            "embedding_query_latency_ms": getattr(self, "_last_embedding_query_latency_ms", 0.0),
            "dense_search_latency_ms": getattr(self, "_last_dense_search_latency_ms", dense_ms),
            "dense_retrieval_latency_ms": dense_ms,
            "bm25_search_latency_ms": 0.0,
            "hybrid_fusion_latency_ms": 0.0,
            "rerank_latency_ms": rerank_ms,
            "total_retrieval_latency_ms": dense_ms + rerank_ms,
        }
        return result

    
def _chunk_text(ch):
    # поддержка: объект с .text и dict с ["text"]
    if isinstance(ch, dict):
        return ch.get("text") or ch.get("chunk") or ""
    return getattr(ch, "text", "") or ""

def _chunk_payload(ch):
    if isinstance(ch, dict):
        return ch.get("payload") or {}
    return getattr(ch, "payload", {}) or {}


def reorder_context_chunks(chunks: List[Any], order: str) -> List[Any]:
    """
    Reorder retrieved chunks before prompt assembly.

    ``rerank_123`` preserves the retriever/reranker order.
    ``rerank_132`` splits that order into three contiguous parts and moves the
    last third before the middle third to test middle-of-context degradation.
    """
    if order == "rerank_123":
        return list(chunks)
    if order != "rerank_132":
        raise ValueError(f"Unsupported context_order={order!r}; expected one of {sorted(CONTEXT_ORDER_MODES)}")

    n = len(chunks)
    base, rem = divmod(n, 3)
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    first_end = sizes[0]
    second_end = first_end + sizes[1]
    part1 = chunks[:first_end]
    part2 = chunks[first_end:second_end]
    part3 = chunks[second_end:]
    return list(part1) + list(part3) + list(part2)

def build_context(cfg, chunks: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Собирает контекст по лимитам:
      - cfg.max_chunk_chars: обрезка каждого чанка
      - cfg.max_context_chars: общий лимит контекста
    Поддерживает чанки как dict ({"text":..., "payload":...}) и как объекты (ch.text / ch.payload).
    Возвращает (context_text, sources).
    """

    def get_text(ch) -> str:
        if isinstance(ch, dict):
            return (ch.get("text") or ch.get("chunk") or "").strip()
        return (getattr(ch, "text", "") or "").strip()

    def get_payload(ch) -> Dict[str, Any]:
        if isinstance(ch, dict):
            return ch.get("payload") or {}
        return getattr(ch, "payload", {}) or {}

    def get_score(ch) -> Optional[float]:
        if isinstance(ch, dict):
            s = ch.get("score")
            return float(s) if s is not None else None
        s = getattr(ch, "score", None)
        return float(s) if s is not None else None

    ctx_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    total = 0
    sep = "\n\n---\n\n"
    max_total = int(cfg.max_context_chars)
    max_chunk = int(cfg.max_chunk_chars)

    ordered_chunks = reorder_context_chunks(chunks, getattr(cfg, "context_order", "rerank_123"))

    for ch in ordered_chunks:
        t = get_text(ch)
        if not t:
            continue

        if len(t) > max_chunk:
            t = t[:max_chunk].rstrip() + "…"
    
        add_len = len(t) + (len(sep) if ctx_parts else 0)
        if total + add_len > max_total:
            break

        # ВОТ ЗДЕСЬ: нумерация источников внутри контекста
        ctx_parts.append(f"[{len(ctx_parts)+1}] {t}")
        
        payload = get_payload(ch)
        score = get_score(ch)
        total += add_len

        sources.append(
            {
                "score": score,
                "url": payload.get("url"),
                "title": payload.get("title"),
                "doc_id": payload.get("doc_id"),
                "external_id": payload.get("external_id"),
                "chunk_i": payload.get("chunk_i"),
                "source_code": payload.get("source_code"),
                "chunk_method": payload.get("chunk_method"),
                "parent_id": payload.get("parent_id"),
                "parent_i": payload.get("parent_i"),
                "child_i": payload.get("child_i"),
                "score_parent": payload.get("score_parent"),
                "score_child": payload.get("score_child"),
            }
        )

    context = sep.join(ctx_parts)
    return context, sources


class LlamaCppChatClient:
    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.url = cfg.llm_base_url.rstrip("/") + "/v1/chat/completions"

    def _messages_without_qwen_thinking(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        model = str(getattr(self.cfg, "llm_model", "") or "").lower()
        if "qwen3" not in model:
            return messages
        out = [dict(m) for m in messages]
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") != "user":
                continue
            content = str(out[i].get("content") or "")
            if "/no_think" not in content:
                out[i]["content"] = content.rstrip() + "\n\n/no_think"
            break
        return out

    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        reasoning_effort = getattr(self.cfg, "reasoning_effort", None)
        payload = {
            "model": self.cfg.llm_model,
            "messages": self._messages_without_qwen_thinking(messages),
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        if reasoning_effort and str(reasoning_effort).lower() != "off":
            payload["reasoning_effort"] = str(reasoning_effort).lower()
        r = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
        if r.status_code != 200:
            # llama.cpp обычно кладёт причину сюда
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
    
REWRITE_PROMPT = """Ты — эксперт по поиску по базе документов ФНС РФ. Твоя задача — извлечь суть вопроса и переформулировать его для поиска по базе документов ФНС.

Шаг 1. Определи СУТЬ вопроса — что именно хочет узнать пользователь про налоги/отчётность/документы. Игнорируй лишний контекст: тип бизнеса, регион, историю ситуации, детали про сотрудников и т.д.

Шаг 2. Перефразируй суть с профессиональными терминами ФНС: добавь синонимы, раскрой аббревиатуры, добавь возможные названия документов/форм.
Если вопрос не про налоги или кажется нерелевантным, все равно верни максимально близкий поисковый запрос по ключевым словам, не отказывайся и не возвращай пустой ответ.

Примеры:
- "у нас ооо в подмосковье торгуем запчастями, нужно ли сдавать отчёт по счёту в латвийском банке?" → "отчёт о движении денежных средств по счёту в иностранном банке КНД 1112521 обязанность резидента валютный контроль"
- "я ип стригу людей, если найму мастера что будет с патентом" → "патентная система налогообложения найм работников превышение численности утрата права ПСН"
- "пришло требование от налоговой не понимаю как считать штраф" → "требование об уплате налога штраф расчёт пени недоимка"

Верни ТОЛЬКО переформулированный поисковый запрос, без пояснений.

Вопрос: {query}"""


HYDE_PROMPT = """Ты — эксперт по поиску по базе документов ФНС РФ. Напиши короткий фрагмент официального документа ФНС, который может помочь найти ответ на вопрос пользователя.

Требования:
- пиши языком официальных писем/приказов ФНС (сухой, юридический стиль)
- используй профессиональные термины, названия форм, статей НК РФ
- не отвечай на вопрос напрямую — имитируй выдержку из документа
- 1-2 коротких предложения, без воды
- если вопрос не про налоги, все равно верни нейтральный юридический фрагмент по ключевым словам вопроса
- не возвращай пустой ответ

Вопрос: {query}

Верни ТОЛЬКО текст гипотетического фрагмента документа."""


def hyde_query(llm: "LlamaCppChatClient", query: str) -> str:
    """Generate a hypothetical document passage and use it as the search query."""
    try:
        msgs = [{"role": "user", "content": HYDE_PROMPT.format(query=query)}]
        hypothesis = llm.chat(msgs, max_tokens=300).strip()
        if len(hypothesis) < 20:
            return query
        return hypothesis
    except Exception:
        return query


def rewrite_query(llm: "LlamaCppChatClient", query: str) -> str:
    """Expand query with tax-law terminology for better semantic matching."""
    try:
        msgs = [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}]
        rewritten = llm.chat(msgs).strip()
        # Sanity check: if LLM returned something too short or identical, keep original
        if len(rewritten) < 10 or rewritten == query:
            return query
        return rewritten
    except Exception:
        return query


# Промпт генерации (вариант strict_citations с доработками для бота: отказ на
# нерелевантные вопросы с примерами). Модель ставит ссылки [n] на ИСХОДНЫЕ номера
# фрагментов и НЕ пишет раздел «Источники» — он собирается постобработкой
# (_finalize_answer): остаются только использованные ссылки, перенумеровываются
# подряд, и к ним добавляются названия документов из метаданных.
SYSTEM_PROMPT = """Ты — налоговый консультант. Отвечаешь на вопросы по налогам, сборам, страховым взносам, отчётности, проверкам и льготам, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленный контекст — пронумерованные фрагменты документов [1], [2], … .

Не используй внешние знания, не отвечай по памяти, не выдумывай факты, реквизиты, ссылки, сроки, суммы штрафов и номера статей.

Раздел «Источники» сам НЕ пиши — он добавляется автоматически по использованным тобой ссылкам.

Сначала про себя определи, относится ли вопрос к налогам — пользователю об этом не сообщай и НЕ пиши фраз о релевантности вроде «Ваш вопрос релевантен…». Вопрос РЕЛЕВАНТЕН, если в сообщении есть вопрос по налогам, сборам, взносам, отчётности, срокам, проверкам или льготам — даже если он сформулирован общо и содержит лишние бытовые детали (профессия, город, описание ситуации). Такие детали — это просто контекст; для релевантного вопроса сразу переходи к ответу по структуре ниже, не считая его нерелевантным и не предваряя ответ оценкой релевантности.

Вопрос НЕРЕЛЕВАНТЕН, только если он вообще не про налоги (например «Как дела?», приветствия, болтовня, посторонние темы). Только в этом случае начни ответ строго с фразы «Простите, мне кажется, вы задали нерелевантный вопрос.», затем коротко сообщи, что помогаешь только с вопросами по налогам и отчётности, и приведи 2–3 примера вопросов из списка ниже. Структурированный ответ не давай и ссылки [n] не используй.

Если вопрос про налоги, но в контексте нет ответа (или не хватает данных, например неизвестен налоговый режим): ответь «В предоставленных документах нет достаточной информации для ответа.» (без ссылок) и при необходимости попроси уточнить детали (налоговый режим, вид деятельности, период). Не называй такой вопрос нерелевантным.

Примеры вопросов, на которые ты отвечаешь:
• Какой срок уплаты налога по УСН за первый квартал?
• Какие документы нужны для вычета по НДФЛ?
• Можно ли уменьшить налог по УСН на страховые взносы?

Если вопрос релевантный и в контексте есть ответ, структурируй ответ так:
1. Краткий вывод (2–5 предложений).
2. Обоснование по документам — после ключевых утверждений ставь ссылку [n] на номер фрагмента из контекста, на который опираешься. Используй номера фрагментов ровно так, как они даны в контексте, и ссылайся только на те фрагменты, которые действительно использовал."""


# Человекочитаемые названия источников по source_code (метаданные корпуса бедные:
# title часто мусорный, url/номер/дата отсутствуют, поэтому строим имя по коду).
SOURCE_NAMES = {
    "pravo_nk1": "Налоговый кодекс РФ",
    "nalog_letters": "Разъяснение ФНС России",
    "nalog_docs": "Документ ФНС России",
    "nalog_calendar": "Налоговый календарь ФНС",
    "minfin_commonlaw": "Письмо Минфина России",
    "minfin_orgprofit": "Письмо Минфина России",
    "minfin_fizprofit": "Письмо Минфина России",
    "minfin_property": "Письмо Минфина России",
    "minfin_indirect": "Письмо Минфина России",
    "minfin_international": "Письмо Минфина России",
    "minfin_special": "Письмо Минфина России",
    "minfin_transfert": "Письмо Минфина России",
    "minfin_foreign": "Письмо Минфина России",
    "minfin_customs_value": "Письмо Минфина России",
    "minfin_imposition": "Письмо Минфина России",
}

# Базовые URL источников (из config/sources.yaml) — ссылка на раздел источника.
SOURCE_BASE_URLS = {
    "nalog_letters": "https://www.nalog.gov.ru/rn77/about_fts/about_nalog/",
    "nalog_docs": "https://www.nalog.gov.ru/rn77/about_fts/docs/",
    "nalog_calendar": "https://www.nalog.gov.ru/rn77/opendata/7707329152-kalendar/",
    "pravo_nk1": "http://pravo.gov.ru/proxy/ips/?savertf=&nd=102054722&page=all",
    "minfin_commonlaw": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/commonlaw/",
    "minfin_orgprofit": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/orgprofit/",
    "minfin_fizprofit": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/fizprofit/",
    "minfin_property": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/property/",
    "minfin_indirect": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/indirect/",
    "minfin_international": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/international/",
    "minfin_special": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/special/",
    "minfin_transfert": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/transfert/",
    "minfin_foreign": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/foreign/",
    "minfin_customs_value": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/customs_value/",
    "minfin_imposition": "https://minfin.gov.ru/ru/perfomance/tax_relations/Answers/imposition/",
}

# Источники, у которых title — основной идентификатор (регион / событие календаря).
_TITLE_SOURCES = {"nalog_docs", "nalog_calendar"}

_GENERIC_TITLES = {"", "минфин", "российская федерация", "фнс"}
_SLUG_DATE_RE = re.compile(r"ot[_-](\d{2}\.\d{2}\.\d{2,4})")
_CITE_RE = re.compile(r"\[(\d+)\]")
_SOURCES_HEADING_RE = re.compile(
    r"\n[\s*#>\-]*(?:\d+[.)]\s*)?(?:\*\*)?Источник\w*(?:\*\*)?\s*[:.]?\s*(?:\n|$).*$",
    re.S | re.I,
)


def _source_url(source_code: str, external_id: str) -> Optional[str]:
    """URL документа: точный для НК РФ и документов ФНС, иначе — раздел источника."""
    eid = str(external_id or "")
    if source_code == "pravo_nk1":
        nd = eid if eid.isdigit() else "102054722"
        return f"http://pravo.gov.ru/proxy/ips/?savertf=&nd={nd}&page=all"
    if source_code == "nalog_docs" and eid.isdigit():
        return f"https://www.nalog.gov.ru/rn77/about_fts/docs/{eid}/"
    return SOURCE_BASE_URLS.get(source_code)


def _source_name(src: Dict[str, Any]) -> str:
    """Название документа: имя источника + (дата/заголовок), без URL."""
    sc = str(src.get("source_code") or "")
    eid = str(src.get("external_id") or "")
    base = SOURCE_NAMES.get(sc, sc or "Источник")
    extras: List[str] = []
    m = _SLUG_DATE_RE.search(eid)
    if m:
        extras.append(f"от {m.group(1)}")
    if sc in _TITLE_SOURCES:
        title = str(src.get("title") or "").strip()
        if title and title.lower() not in _GENERIC_TITLES and len(title) <= 80:
            extras.append(title)
    return base + (f" ({'; '.join(extras)})" if extras else "")


def _source_line(src: Dict[str, Any], num: int, html_links: bool) -> str:
    """Строка источника. html_links=True → название как HTML-гиперссылка на URL."""
    name = _source_name(src)
    sc = str(src.get("source_code") or "")
    eid = str(src.get("external_id") or "")
    url = src.get("url") or src.get("canonical_url") or _source_url(sc, eid)
    if html_links:
        text = html.escape(name)
        if url:
            return f'[{num}] <a href="{html.escape(str(url), quote=True)}">{text}</a>'
        return f"[{num}] {text}"
    line = f"[{num}] {name}"
    if url:
        line += f"\n   {url}"
    return line


def _finalize_answer(answer: str, sources: List[Dict[str, Any]], html_links: bool = False) -> str:
    """Постобработка ответа модели.

    Модель ссылается на исходные номера фрагментов [n]. Здесь:
      - срезаем раздел «Источники», если модель его всё же написал;
      - оставляем только реально использованные ссылки и перенумеровываем их подряд;
      - заново собираем раздел «Источники» с названиями документов.
    Если использованных ссылок нет (нерелевантный вопрос / отказ) — раздел не добавляется.
    html_links=True → тело экранируется, а источники оформляются как HTML-гиперссылки
    (для отправки в Telegram с parse_mode=HTML).
    """
    if not answer:
        return answer
    answer = _SOURCES_HEADING_RE.sub("", answer).rstrip()

    used: List[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(sources) and n not in used:
            used.append(n)
    if not used:
        return html.escape(answer) if html_links else answer

    remap = {orig: i + 1 for i, orig in enumerate(used)}
    answer = _CITE_RE.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
        answer,
    )
    body = html.escape(answer) if html_links else answer

    lines = ["Источники:"]
    for orig in used:
        lines.append(_source_line(sources[orig - 1], remap[orig], html_links))
    return body.rstrip() + "\n\n" + "\n".join(lines)

def rag_answer(
    cfg: RAGConfig,
    retriever: Retriever,
    llm: LlamaCppChatClient,
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    html_links: bool = False,
) -> Dict[str, Any]:
    if chunks is None:
        if cfg.use_hyde:
            search_query = hyde_query(llm, user_query)
        elif cfg.use_rewrite:
            search_query = rewrite_query(llm, user_query)
        else:
            search_query = user_query
        chunks = retriever.search(search_query)
    context, sources = build_context(cfg, chunks)

    # Сообщения для LLM
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        # Ограничение истории по вкусу; сейчас просто добавляем как есть
        msgs.extend(chat_history[-4:])

    user_prompt = f"""Вопрос пользователя:
{user_query}

Контекст:
{context if context else "(контекст пуст)"}"""

    msgs.append({"role": "user", "content": user_prompt})

    answer = llm.chat(msgs, max_tokens=2000)
    answer = _finalize_answer(answer, sources, html_links=html_links)

    return {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(chunks),
    }
