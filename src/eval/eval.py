from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag.core import CONTEXT_ORDER_MODES, RAGConfig, STEmbedder, Retriever, LlamaCppChatClient, build_context, rag_answer, rewrite_query
from src.rag.embedders import embedder_short_name
# мб не работает, не запускал еще

# ----------------------------
# Dataset loading
# ----------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ----------------------------
# Retrieval metrics
# ----------------------------

MATCH_KEYS = ("point_id", "source_code", "doc_id", "external_id", "url", "chunk_i")
MATCH_MODES = ("strict", "doc", "doc_overlap", "hybrid")
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


def _target_value(target: Dict[str, Any], key: str) -> Any:
    if key == "url":
        return target.get("url") or target.get("source_url") or target.get("doc_url")
    if key == "point_id":
        value = target.get("point_id") or target.get("id")
        return str(value) if value is not None and value != "" else None
    return target.get(key)


def _candidate_identity(candidate: Dict[str, Any]) -> Dict[str, Any]:
    payload = candidate.get("payload") or {}
    point_id = candidate.get("id") or payload.get("point_id") or payload.get("id")
    return {
        "point_id": str(point_id) if point_id is not None and point_id != "" else None,
        "source_code": payload.get("source_code"),
        "doc_id": payload.get("doc_id"),
        "external_id": payload.get("external_id"),
        "url": payload.get("url") or payload.get("source_url") or payload.get("doc_url"),
        "chunk_i": payload.get("chunk_i"),
    }


def _candidate_text(candidate: Dict[str, Any]) -> str:
    payload = candidate.get("payload") or {}
    parts = [
        candidate.get("text"),
        payload.get("text"),
        payload.get("chunk"),
        payload.get("child_text"),
        payload.get("parent_text"),
    ]
    return "\n".join(str(x) for x in parts if x)


def _match_target(target: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    """
    Определяет, совпадает ли retrieved-кандидат с таргетом из датасета.
    Новый формат датасета хранит point_id/source_code/external_id/chunk_i; старый
    мог хранить doc_id/url. Если поле задано в таргете, оно должно совпасть.
    """
    ident = _candidate_identity(candidate)
    target_source = _target_value(target, "source_code")
    target_external_id = _target_value(target, "external_id")
    target_chunk_i = _target_value(target, "chunk_i")
    if target_source and target_external_id and target_chunk_i is not None:
        return (
            ident.get("source_code") == target_source
            and ident.get("external_id") == target_external_id
            and str(ident.get("chunk_i")) == str(target_chunk_i)
        )

    specified = []
    for key in MATCH_KEYS:
        target_value = _target_value(target, key)
        if target_value is None or target_value == "":
            continue
        specified.append(key)
        if key == "point_id":
            if str(ident.get(key)) != str(target_value):
                return False
        elif key == "chunk_i":
            if str(ident.get(key)) != str(target_value):
                return False
        elif ident.get(key) != target_value:
            return False

    if not specified:
        return False

    return True


def _match_doc(target: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    ident = _candidate_identity(candidate)
    source = _target_value(target, "source_code")
    external_id = _target_value(target, "external_id")
    if source and external_id:
        return ident.get("source_code") == source and ident.get("external_id") == external_id
    for key in ("doc_id", "url"):
        target_value = _target_value(target, key)
        if target_value and ident.get(key) == target_value:
            return True
    return False


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")


def _text_tokens(text: str) -> set[str]:
    return {m.group(0).lower().replace("ё", "е") for m in TOKEN_RE.finditer(text or "")}


def _text_overlap_score(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    target_text = str(target.get("text") or target.get("chunk") or "")
    if not target_text:
        return 0.0
    target_tokens = _text_tokens(target_text)
    candidate_tokens = _text_tokens(_candidate_text(candidate))
    if not target_tokens or not candidate_tokens:
        return 0.0
    return len(target_tokens & candidate_tokens) / max(1, min(len(target_tokens), len(candidate_tokens)))


def _match_doc_overlap(target: Dict[str, Any], candidate: Dict[str, Any], *, overlap_threshold: float) -> bool:
    if not _match_doc(target, candidate):
        return False
    if not (target.get("text") or target.get("chunk")):
        return True
    return _text_overlap_score(target, candidate) >= overlap_threshold


def _match_target_mode(
    target: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    match_mode: str,
    overlap_threshold: float,
) -> bool:
    if match_mode == "strict":
        return _match_target(target, candidate)
    if match_mode == "doc":
        return _match_doc(target, candidate)
    if match_mode == "doc_overlap":
        return _match_doc_overlap(target, candidate, overlap_threshold=overlap_threshold)
    if match_mode == "hybrid":
        return _match_target(target, candidate) or _match_doc_overlap(target, candidate, overlap_threshold=overlap_threshold)
    raise ValueError(f"Unsupported match_mode={match_mode!r}; expected one of {MATCH_MODES}")


def _binary_relevance_list(
    retrieved: List[Dict[str, Any]],
    relevant: List[Dict[str, Any]],
    *,
    match_mode: str = "strict",
    overlap_threshold: float = 0.25,
) -> List[int]:
    """
    Returns a list of 0/1 for each retrieved item whether it matches ANY relevant target.
    """
    rels: List[int] = []
    for r in retrieved:
        hit = any(_match_target_mode(gt, r, match_mode=match_mode, overlap_threshold=overlap_threshold) for gt in relevant)
        rels.append(1 if hit else 0)
    return rels


def _binary_target_list(
    retrieved: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    *,
    match_mode: str = "strict",
    overlap_threshold: float = 0.25,
) -> List[int]:
    vals: List[int] = []
    for r in retrieved:
        hit = any(_match_target_mode(target, r, match_mode=match_mode, overlap_threshold=overlap_threshold) for target in targets)
        vals.append(1 if hit else 0)
    return vals


def precision_at_k(bin_rels: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = bin_rels[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def recall_at_k(bin_rels: List[int], relevant_total: int, k: int) -> float:
    if relevant_total <= 0:
        return 0.0
    # В новых датасетах один вопрос может иметь несколько acceptable relevant
    # chunks: исходный chunk плюс похожие валидированные chunks. Для hit-rate
    # достаточно найти любой из них.
    return 1.0 if any(bin_rels[:k]) else 0.0


def mrr_at_k(bin_rels: List[int], k: int) -> float:
    for i, rel in enumerate(bin_rels[:k], start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


def ndcg_at_k(bin_rels: List[int], k: int) -> float:
    """
    Binary nDCG@k with log2 discount.
    """
    def dcg(rels: List[int]) -> float:
        s = 0.0
        for i, r in enumerate(rels[:k], start=1):
            if r:
                s += 1.0 / math.log2(i + 1)
        return s

    actual = dcg(bin_rels)
    ideal = dcg(sorted(bin_rels, reverse=True))
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def hard_negative_count_at_k(bin_hard_negatives: List[int], k: int) -> int:
    return sum(bin_hard_negatives[:k])


def hard_negative_rate_at_k(bin_hard_negatives: List[int], k: int) -> float:
    return 1.0 if any(bin_hard_negatives[:k]) else 0.0


def first_hard_negative_rank(bin_hard_negatives: List[int], k: int) -> Optional[int]:
    for i, val in enumerate(bin_hard_negatives[:k], start=1):
        if val == 1:
            return i
    return None


# ----------------------------
# Generation metrics (heuristics + LLM judge)
# ----------------------------

CIT_PATTERN = re.compile(r"\[(\d+)\]")

def extract_citations(answer: str) -> List[int]:
    return [int(x) for x in CIT_PATTERN.findall(answer)]


def citation_validity(answer: str, n_sources: int) -> float:
    cites = extract_citations(answer)
    if not cites:
        return 0.0
    ok = sum(1 for c in cites if 1 <= c <= n_sources)
    return ok / len(cites)


def citation_presence(answer: str) -> float:
    # simple: 1 if has any [n], else 0
    return 1.0 if extract_citations(answer) else 0.0


def citation_density(answer: str) -> float:
    # citations per 1000 chars
    cites = extract_citations(answer)
    if not answer:
        return 0.0
    return len(cites) * 1000.0 / max(1, len(answer))


JUDGE_SYSTEM = """Ты — строгий оценщик качества ответа RAG-системы.
Оценивай ТОЛЬКО по вопросу пользователя и предоставленному контексту (выдержки источников).
Нельзя домысливать факты вне контекста.

Верни JSON строго следующего вида:
{
  "faithfulness": <0|1|2>,     // 0 = есть недоказанные утверждения; 1 = сомнительно; 2 = все утверждения опираются на контекст
  "relevance": <0|1|2>,        // 0 = не отвечает; 1 = частично; 2 = отвечает по сути
  "notes": "<коротко, 1-3 предложения>"
}
"""

def judge_with_llm(
    llm: LlamaCppChatClient,
    question: str,
    context: str,
    answer: str,
) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context}\n\nОтвет:\n{answer}\n\nВерни JSON:"},
    ]
    raw = llm.chat(msgs)

    # попытка вытащить JSON, даже если LLM обрамил текстом
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"faithfulness": None, "relevance": None, "notes": f"Judge parse error: {raw[:200]}"}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {"faithfulness": None, "relevance": None, "notes": f"Judge JSON error: {m.group(0)[:200]}"}


# ----------------------------
# Runner
# ----------------------------

@dataclass
class EvalResult:
    id: str
    query: str

    # retrieval
    p_at_k: float
    r_at_k: float
    mrr: float
    ndcg: float
    hard_negative_count: int
    hard_negative_rate: Optional[float]
    first_hard_negative_rank: Optional[int]
    has_hard_negatives: bool
    doc_hit_at_k: float
    overlap_hit_at_k: float
    embedding_query_latency_ms: float
    dense_search_latency_ms: float
    dense_retrieval_latency_ms: float
    bm25_search_latency_ms: float
    hybrid_fusion_latency_ms: float
    rerank_latency_ms: float
    total_retrieval_latency_ms: float
    transform_latency_ms: float
    total_query_latency_ms: float

    # generation
    cite_presence: float
    cite_validity: float
    cite_density: float
    judge_faithfulness: Optional[int]
    judge_relevance: Optional[int]
    judge_notes: str

    retrieved_count: int
    source_code: str = ""


def evaluate_dataset(
    dataset_path: str,
    *,
    k: Optional[int] = None,
    use_llm_judge: bool = True,
    use_rewrite: bool = False,
    use_hyde: bool = False,
    use_reranker: bool = False,
    reranker_model: Optional[str] = None,
    reranker_type: Optional[str] = None,
    reranker_fetch_k: Optional[int] = None,
    reranker_max_length: Optional[int] = None,
    reranker_batch_size: Optional[int] = None,
    reranker_device: Optional[str] = None,
    reranker_use_fp16: Optional[bool] = None,
    reranker_normalize: Optional[bool] = None,
    reranker_instruction: Optional[str] = None,
    embed_model: Optional[str] = None,
    embed_backend: Optional[str] = None,
    embed_device: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    embed_max_seq_length: Optional[int] = None,
    embed_normalize: Optional[bool] = None,
    embed_trust_remote_code: Optional[bool] = None,
    embed_query_prefix: Optional[str] = None,
    embed_passage_prefix: Optional[str] = None,
    embed_query_instruction: Optional[str] = None,
    qdrant_collection: Optional[str] = None,
    retrieval_mode: str = "dense",
    bm25_fetch_k: Optional[int] = None,
    hybrid_fetch_k: Optional[int] = None,
    hybrid_rrf_k: Optional[int] = None,
    context_order: Optional[str] = None,
    match_mode: str = "strict",
    overlap_threshold: float = 0.25,
    config_overrides: Optional[Dict[str, Any]] = None,
    method: Optional[str] = None,
    # pre-loaded components (pass to avoid reloading across runs)
    embedder=None,
    retriever=None,
    llm=None,
    # precomputed query transformations {original_query: transformed_query}
    precomputed_queries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    import dataclasses
    if match_mode not in MATCH_MODES:
        raise ValueError(f"Unsupported match_mode={match_mode!r}; expected one of {MATCH_MODES}")
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval_mode={retrieval_mode!r}; expected one of {RETRIEVAL_MODES}")
    cfg = RAGConfig()
    overrides: Dict[str, Any] = {
        "use_rewrite": use_rewrite,
        "use_hyde": use_hyde,
        "use_reranker": use_reranker,
        "retrieval_mode": retrieval_mode,
    }
    if reranker_model is not None:
        overrides["reranker_model"] = reranker_model
    if reranker_type is not None:
        overrides["reranker_type"] = reranker_type
    if reranker_fetch_k is not None:
        overrides["reranker_fetch_k"] = reranker_fetch_k
        overrides["reranker_top_k"] = reranker_fetch_k
    if reranker_max_length is not None:
        overrides["reranker_max_length"] = reranker_max_length
    if reranker_batch_size is not None:
        overrides["reranker_batch_size"] = reranker_batch_size
    if reranker_device is not None:
        overrides["reranker_device"] = reranker_device
    if reranker_use_fp16 is not None:
        overrides["reranker_use_fp16"] = reranker_use_fp16
    if reranker_normalize is not None:
        overrides["reranker_normalize"] = reranker_normalize
    if reranker_instruction is not None:
        overrides["reranker_instruction"] = reranker_instruction
    if embed_model is not None:
        overrides["embed_model_name"] = embed_model
    if embed_backend is not None:
        overrides["embed_backend"] = embed_backend
    if embed_device is not None:
        overrides["embed_device"] = embed_device
    if embed_batch_size is not None:
        overrides["embed_batch_size"] = embed_batch_size
    if embed_max_seq_length is not None:
        overrides["embed_max_seq_length"] = embed_max_seq_length
    if embed_normalize is not None:
        overrides["embed_normalize"] = embed_normalize
    if embed_trust_remote_code is not None:
        overrides["embed_trust_remote_code"] = embed_trust_remote_code
    if embed_query_prefix is not None:
        overrides["embed_query_prefix"] = embed_query_prefix
    if embed_passage_prefix is not None:
        overrides["embed_passage_prefix"] = embed_passage_prefix
    if embed_query_instruction is not None:
        overrides["embed_query_instruction"] = embed_query_instruction
    if qdrant_collection is not None:
        overrides["qdrant_collection"] = qdrant_collection
    if bm25_fetch_k is not None:
        overrides["bm25_fetch_k"] = bm25_fetch_k
    if hybrid_fetch_k is not None:
        overrides["hybrid_fetch_k"] = hybrid_fetch_k
    if hybrid_rrf_k is not None:
        overrides["hybrid_rrf_k"] = hybrid_rrf_k
    if context_order is not None:
        overrides["context_order"] = context_order
    if config_overrides:
        overrides.update(config_overrides)
    cfg = dataclasses.replace(cfg, **overrides)
    if k is None:
        k = cfg.top_k

    if embedder is None and retrieval_mode != "bm25":
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
    if retriever is None:
        retriever = Retriever(cfg, embedder)
    else:
        # update flags on existing retriever's config
        old_key = (
            getattr(retriever.cfg, "reranker_model", None),
            getattr(retriever.cfg, "reranker_type", None),
            getattr(retriever.cfg, "reranker_device", None),
            getattr(retriever.cfg, "reranker_use_fp16", None),
            getattr(retriever.cfg, "reranker_max_length", None),
        )
        retriever.cfg = cfg
        new_key = (
            cfg.reranker_model,
            cfg.reranker_type,
            cfg.reranker_device,
            cfg.reranker_use_fp16,
            cfg.reranker_max_length,
        )
        if old_key != new_key and hasattr(retriever, "_reranker"):
            retriever._reranker = None
    from tqdm import tqdm

    ds = load_jsonl(dataset_path)

    per_item: List[EvalResult] = []
    retrieved_payload_samples: List[Dict[str, Any]] = []
    retrieved_context_chars: List[int] = []

    for item in tqdm(ds, desc="eval", unit="q"):
        qid = str(item.get("id", ""))
        query = str(item["query"])
        relevant = item.get("relevant") or []
        if not relevant and item.get("source_chunk"):
            relevant = [item["source_chunk"]]
        hard_negatives = item.get("hard_negatives") or []
        relevant_total = len(relevant)
        source_chunk = item.get("source_chunk") or {}
        source_code = source_chunk.get("source_code") or (relevant[0].get("source_code", "") if relevant else "")

        query_t0 = time.perf_counter()
        transform_ms = 0.0
        # retrieval only
        from src.rag.core import hyde_query
        if precomputed_queries and query in precomputed_queries:
            search_query = precomputed_queries[query]
        elif use_hyde:
            if llm is None:
                llm = LlamaCppChatClient(cfg)
            tt = time.perf_counter()
            search_query = hyde_query(llm, query)
            transform_ms = (time.perf_counter() - tt) * 1000.0
        elif use_rewrite:
            if llm is None:
                llm = LlamaCppChatClient(cfg)
            tt = time.perf_counter()
            search_query = rewrite_query(llm, query)
            transform_ms = (time.perf_counter() - tt) * 1000.0
        else:
            search_query = query
        retrieved = retriever.search(search_query)
        for r in retrieved:
            payload = dict(r.get("payload") or {})
            for key in (
                "reranker_model",
                "reranker_type",
                "reranker_fetch_k",
                "reranker_max_length",
                "reranker_effective_max_length",
            ):
                if key in r:
                    payload[key] = r.get(key)
            if len(retrieved_payload_samples) < 200:
                retrieved_payload_samples.append(payload)
            retrieved_context_chars.append(len(str(r.get("text") or "")))
        timings = getattr(retriever, "last_timings", {}) or {}
        embed_ms = float(timings.get("embedding_query_latency_ms", 0.0))
        dense_search_ms = float(timings.get("dense_search_latency_ms", 0.0))
        dense_ms = float(timings.get("dense_retrieval_latency_ms", 0.0))
        bm25_ms = float(timings.get("bm25_search_latency_ms", 0.0))
        fusion_ms = float(timings.get("hybrid_fusion_latency_ms", 0.0))
        rerank_ms = float(timings.get("rerank_latency_ms", 0.0))
        retrieval_ms = float(timings.get("total_retrieval_latency_ms", dense_ms + rerank_ms))
        total_query_ms = (time.perf_counter() - query_t0) * 1000.0
        bin_rels = _binary_relevance_list(
            retrieved,
            relevant,
            match_mode=match_mode,
            overlap_threshold=overlap_threshold,
        )
        bin_doc_rels = _binary_relevance_list(
            retrieved,
            relevant,
            match_mode="doc",
            overlap_threshold=overlap_threshold,
        )
        bin_overlap_rels = _binary_relevance_list(
            retrieved,
            relevant,
            match_mode="doc_overlap",
            overlap_threshold=overlap_threshold,
        )
        bin_hard_negatives = _binary_target_list(
            retrieved,
            hard_negatives,
            match_mode=match_mode,
            overlap_threshold=overlap_threshold,
        )
        has_hard_negatives = bool(hard_negatives)

        p = precision_at_k(bin_rels, k)
        r = recall_at_k(bin_rels, relevant_total, k)
        mrr = mrr_at_k(bin_rels, k)
        nd = ndcg_at_k(bin_rels, k)
        doc_hit = recall_at_k(bin_doc_rels, relevant_total, k)
        overlap_hit = recall_at_k(bin_overlap_rels, relevant_total, k)
        hn_count = hard_negative_count_at_k(bin_hard_negatives, k)
        hn_rate = hard_negative_rate_at_k(bin_hard_negatives, k) if has_hard_negatives else None
        hn_rank = first_hard_negative_rank(bin_hard_negatives, k)

        pres = val = dens = 0.0
        jf = jr = None
        jn = ""

        if use_llm_judge:
            if llm is None:
                llm = LlamaCppChatClient(cfg)
            context, sources = build_context(cfg, retrieved)
            out = rag_answer(cfg, retriever, llm, query, chat_history=None, chunks=retrieved)
            answer = out["answer"]

            pres = citation_presence(answer)
            val = citation_validity(answer, n_sources=len(sources))
            dens = citation_density(answer)

            j = judge_with_llm(llm, query, context, answer)
            jf = j.get("faithfulness")
            jr = j.get("relevance")
            jn = str(j.get("notes", ""))

        per_item.append(
            EvalResult(
                id=qid,
                query=query,
                p_at_k=p,
                r_at_k=r,
                mrr=mrr,
                ndcg=nd,
                hard_negative_count=hn_count,
                hard_negative_rate=hn_rate,
                first_hard_negative_rank=hn_rank,
                has_hard_negatives=has_hard_negatives,
                doc_hit_at_k=doc_hit,
                overlap_hit_at_k=overlap_hit,
                embedding_query_latency_ms=embed_ms,
                dense_search_latency_ms=dense_search_ms,
                dense_retrieval_latency_ms=dense_ms,
                bm25_search_latency_ms=bm25_ms,
                hybrid_fusion_latency_ms=fusion_ms,
                rerank_latency_ms=rerank_ms,
                total_retrieval_latency_ms=retrieval_ms,
                transform_latency_ms=transform_ms,
                total_query_latency_ms=total_query_ms,
                cite_presence=pres,
                cite_validity=val,
                cite_density=dens,
                judge_faithfulness=jf,
                judge_relevance=jr,
                judge_notes=jn,
                retrieved_count=len(retrieved),
                source_code=source_code,
            )
        )

    # aggregate
    def avg(xs: List[float]) -> float:
        xs2 = [x for x in xs if x is not None]
        return sum(xs2) / max(1, len(xs2))

    def percentile(xs: List[float], pct: float) -> Optional[float]:
        xs2 = sorted(float(x) for x in xs if x is not None)
        if not xs2:
            return None
        if len(xs2) == 1:
            return xs2[0]
        idx = (len(xs2) - 1) * pct
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return xs2[int(idx)]
        return xs2[lo] * (hi - idx) + xs2[hi] * (idx - lo)

    hn_items = [x for x in per_item if x.has_hard_negatives]

    metrics = {
        f"precision@{k}": avg([x.p_at_k for x in per_item]),
        f"recall@{k}": avg([x.r_at_k for x in per_item]),
        f"mrr@{k}": avg([x.mrr for x in per_item]),
        f"ndcg@{k}": avg([x.ndcg for x in per_item]),
        f"doc_hit@{k}": avg([x.doc_hit_at_k for x in per_item]),
        f"overlap_hit@{k}": avg([x.overlap_hit_at_k for x in per_item]),
        f"hard_negative_rate@{k}": avg([x.hard_negative_rate for x in hn_items]) if hn_items else None,
        f"avg_hard_negatives@{k}": avg([float(x.hard_negative_count) for x in hn_items]) if hn_items else None,
        f"hard_negative_count@{k}": avg([float(x.hard_negative_count) for x in hn_items]) if hn_items else None,
        "latency_avg_ms": avg([x.total_query_latency_ms for x in per_item]),
        "latency_p50_ms": percentile([x.total_query_latency_ms for x in per_item], 0.50),
        "latency_p95_ms": percentile([x.total_query_latency_ms for x in per_item], 0.95),
        "embedding_query_latency_avg_ms": avg([x.embedding_query_latency_ms for x in per_item]),
        "embedding_query_latency_p50_ms": percentile([x.embedding_query_latency_ms for x in per_item], 0.50),
        "embedding_query_latency_p95_ms": percentile([x.embedding_query_latency_ms for x in per_item], 0.95),
        "dense_search_latency_avg_ms": avg([x.dense_search_latency_ms for x in per_item]),
        "dense_search_latency_p50_ms": percentile([x.dense_search_latency_ms for x in per_item], 0.50),
        "dense_search_latency_p95_ms": percentile([x.dense_search_latency_ms for x in per_item], 0.95),
        "dense_latency_avg_ms": avg([x.dense_retrieval_latency_ms for x in per_item]),
        "bm25_search_latency_avg_ms": avg([x.bm25_search_latency_ms for x in per_item]),
        "bm25_search_latency_p50_ms": percentile([x.bm25_search_latency_ms for x in per_item], 0.50),
        "bm25_search_latency_p95_ms": percentile([x.bm25_search_latency_ms for x in per_item], 0.95),
        "hybrid_fusion_latency_avg_ms": avg([x.hybrid_fusion_latency_ms for x in per_item]),
        "hybrid_fusion_latency_p50_ms": percentile([x.hybrid_fusion_latency_ms for x in per_item], 0.50),
        "hybrid_fusion_latency_p95_ms": percentile([x.hybrid_fusion_latency_ms for x in per_item], 0.95),
        "rerank_latency_avg_ms": avg([x.rerank_latency_ms for x in per_item]),
        "rerank_latency_p50_ms": percentile([x.rerank_latency_ms for x in per_item], 0.50),
        "rerank_latency_p95_ms": percentile([x.rerank_latency_ms for x in per_item], 0.95),
        "transform_latency_avg_ms": avg([x.transform_latency_ms for x in per_item]),
        "total_retrieval_latency_avg_ms": avg([x.total_retrieval_latency_ms for x in per_item]),
        "citation_presence": avg([x.cite_presence for x in per_item]),
        "citation_validity": avg([x.cite_validity for x in per_item]),
        "citation_density_per_1k_chars": avg([x.cite_density for x in per_item]),
    }

    if use_llm_judge:
        f_vals = [x.judge_faithfulness for x in per_item if isinstance(x.judge_faithfulness, int)]
        r_vals = [x.judge_relevance for x in per_item if isinstance(x.judge_relevance, int)]
        metrics["judge_faithfulness_avg_0_2"] = avg([float(v) for v in f_vals]) if f_vals else None
        metrics["judge_relevance_avg_0_2"] = avg([float(v) for v in r_vals]) if r_vals else None

    # per-source breakdown
    from collections import defaultdict
    by_source: Dict[str, List[EvalResult]] = defaultdict(list)
    for x in per_item:
        by_source[x.source_code or "unknown"].append(x)

    per_source = {}
    for sc, items in sorted(by_source.items()):
        per_source[sc] = {
            "n": len(items),
            f"hit_rate@{k}": avg([x.r_at_k for x in items]),
            f"mrr@{k}":      avg([x.mrr   for x in items]),
            f"ndcg@{k}":     avg([x.ndcg  for x in items]),
            f"doc_hit@{k}":  avg([x.doc_hit_at_k for x in items]),
            f"overlap_hit@{k}": avg([x.overlap_hit_at_k for x in items]),
            f"hard_negative_rate@{k}": avg([x.hard_negative_rate for x in items if x.has_hard_negatives]) if any(x.has_hard_negatives for x in items) else None,
            f"avg_hard_negatives@{k}": avg([float(x.hard_negative_count) for x in items if x.has_hard_negatives]) if any(x.has_hard_negatives for x in items) else None,
        }
    metrics["per_source"] = per_source

    if retrieved_payload_samples:
        first_payload = retrieved_payload_samples[0]
        metrics["chunk_method"] = first_payload.get("chunk_method")
        metrics["chunk_size"] = first_payload.get("chunk_size")
        metrics["chunk_overlap"] = first_payload.get("chunk_overlap")
        metrics["avg_chunk_chars"] = avg([
            float(p.get("chunk_char_len") or len(str(p.get("text") or "")))
            for p in retrieved_payload_samples
        ])
        token_vals = [
            float(p.get("chunk_token_count"))
            for p in retrieved_payload_samples
            if p.get("chunk_token_count") is not None
        ]
        metrics["avg_chunk_tokens"] = avg(token_vals) if token_vals else None
        parent_ids = [p.get("parent_id") for p in retrieved_payload_samples if p.get("parent_id")]
        if first_payload.get("chunk_method") == "parent_child" or parent_ids:
            metrics["parent_chunk_size"] = first_payload.get("parent_chunk_size")
            metrics["child_chunk_size"] = first_payload.get("child_chunk_size")
            metrics["parent_chunker_method"] = first_payload.get("parent_chunker_method")
            metrics["child_chunker_method"] = first_payload.get("child_chunker_method")
            parent_lengths = [
                float(p.get("parent_char_len") or len(str(p.get("parent_text") or "")))
                for p in retrieved_payload_samples
                if p.get("parent_text") or p.get("parent_char_len")
            ]
            metrics["avg_parent_chars"] = avg(parent_lengths) if parent_lengths else None
            metrics["unique_parent_count"] = len(set(parent_ids))
            metrics["duplicate_parent_rate"] = 1.0 - (len(set(parent_ids)) / max(1, len(parent_ids))) if parent_ids else 0.0
            metrics["context_chars_total"] = sum(retrieved_context_chars)

    # dump detailed
    detailed = [
        {
            "id": x.id,
            "query": x.query,
            f"precision@{k}": x.p_at_k,
            f"recall@{k}": x.r_at_k,
            f"mrr@{k}": x.mrr,
            f"ndcg@{k}": x.ndcg,
            f"hard_negative_count@{k}": x.hard_negative_count,
            f"hard_negative_rate@{k}": x.hard_negative_rate,
            f"first_hard_negative_rank@{k}": x.first_hard_negative_rank,
            f"doc_hit@{k}": x.doc_hit_at_k,
            f"overlap_hit@{k}": x.overlap_hit_at_k,
            "embedding_query_latency_ms": x.embedding_query_latency_ms,
            "dense_search_latency_ms": x.dense_search_latency_ms,
            "dense_retrieval_latency_ms": x.dense_retrieval_latency_ms,
            "bm25_search_latency_ms": x.bm25_search_latency_ms,
            "hybrid_fusion_latency_ms": x.hybrid_fusion_latency_ms,
            "rerank_latency_ms": x.rerank_latency_ms,
            "total_retrieval_latency_ms": x.total_retrieval_latency_ms,
            "transform_latency_ms": x.transform_latency_ms,
            "total_query_latency_ms": x.total_query_latency_ms,
            "citation_presence": x.cite_presence,
            "citation_validity": x.cite_validity,
            "citation_density_per_1k_chars": x.cite_density,
            "judge_faithfulness": x.judge_faithfulness,
            "judge_relevance": x.judge_relevance,
            "judge_notes": x.judge_notes,
            "retrieved_count": x.retrieved_count,
        }
        for x in per_item
    ]

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_commit = None

    from src.rag.rerankers import auto_detect_reranker_type
    effective_type = cfg.reranker_type
    if effective_type == "auto":
        effective_type = auto_detect_reranker_type(cfg.reranker_model)
    effective_reranker_max_length = None
    if use_reranker:
        effective_reranker_max_length = cfg.reranker_max_length
        if retrieved_payload_samples:
            effective_reranker_max_length = retrieved_payload_samples[0].get(
                "reranker_effective_max_length",
                effective_reranker_max_length,
            )
        reranker_obj = getattr(retriever, "_reranker", None)
        if reranker_obj is not None:
            effective_reranker_max_length = getattr(
                reranker_obj,
                "max_length",
                effective_reranker_max_length,
            )

    report_meta = {
        "dataset_path": dataset_path,
        "dataset_stem": Path(dataset_path).stem,
        "method": method or ("reranker" if use_reranker else "baseline"),
        "retrieval_mode": retrieval_mode,
        "bm25_fetch_k": cfg.bm25_fetch_k if retrieval_mode in ("bm25", "hybrid") else None,
        "hybrid_fetch_k": cfg.hybrid_fetch_k if retrieval_mode == "hybrid" else None,
        "hybrid_rrf_k": cfg.hybrid_rrf_k if retrieval_mode == "hybrid" else None,
        "hybrid_fusion": "rrf" if retrieval_mode == "hybrid" else None,
        "bm25_docs_count": (
            ((getattr(retriever, "_bm25_index", None) or {}).get("n_docs"))
            if retrieval_mode in ("bm25", "hybrid") else None
        ),
        "match_mode": match_mode,
        "overlap_threshold": overlap_threshold,
        "reranker_model": cfg.reranker_model if use_reranker else None,
        "reranker_type": effective_type if use_reranker else None,
        "reranker_fetch_k": cfg.reranker_fetch_k if use_reranker else None,
        "final_k": k,
        "context_order": cfg.context_order,
        "reranker_max_length": cfg.reranker_max_length if use_reranker else None,
        "reranker_effective_max_length": effective_reranker_max_length if use_reranker else None,
        "reranker_batch_size": cfg.reranker_batch_size if use_reranker else None,
        "embed_model": cfg.embed_model_name,
        "embed_backend": cfg.embed_backend,
        "embed_short_name": embedder_short_name(cfg.embed_model_name),
        "embed_dim": getattr(embedder, "dim", None) if embedder is not None else None,
        "embed_max_seq_length": cfg.embed_max_seq_length,
        "embed_normalize": cfg.embed_normalize,
        "embed_query_prefix": cfg.embed_query_prefix,
        "embed_passage_prefix": cfg.embed_passage_prefix,
        "embed_query_instruction": cfg.embed_query_instruction,
        "embed_metadata": getattr(embedder, "metadata", None) if embedder is not None else None,
        "qdrant_collection": cfg.qdrant_collection,
        "chunk_method": metrics.get("chunk_method"),
        "chunk_size": metrics.get("chunk_size"),
        "chunk_overlap": metrics.get("chunk_overlap"),
        "parent_chunk_size": metrics.get("parent_chunk_size"),
        "child_chunk_size": metrics.get("child_chunk_size"),
        "parent_chunker_method": metrics.get("parent_chunker_method"),
        "child_chunker_method": metrics.get("child_chunker_method"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
    }

    return {"meta": report_meta, "metrics": metrics, "detailed": detailed}


EVAL_LOG = "data/metrics/eval_log.jsonl"
EVAL_LOG_SOURCE = "data/metrics/eval_log_per_source.jsonl"
TEST_LLM_BASE_URL = "http://172.18.96.1:1234"
TEST_LLM_MODEL = "openai/gpt-oss-20b"
TEST_LLM_NO_PROXY_HOST = "172.18.96.1"


def configure_no_proxy(host: str) -> None:
    import os

    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [part.strip() for part in current.split(",") if part.strip()]
    if host not in parts:
        parts.append(host)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def cleanup_cuda_cache() -> None:
    try:
        import gc
        gc.collect()
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


def _append_log(row: Dict[str, Any], log_path: str = EVAL_LOG) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_log(log_path: str = EVAL_LOG) -> List[Dict[str, Any]]:
    p = Path(log_path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _print_log_table(log_path: str = EVAL_LOG) -> None:
    rows = _load_log(log_path)
    if not rows:
        return

    # Columns to display
    cols = [
        ("timestamp",       "time",         19),
        ("model",           "model",        28),
        ("dataset",         "dataset",      24),
        ("k",               "k",             3),
        ("n",               "n",             5),
        ("hit_rate",        "hit@k",         6),
        ("mrr",             "mrr@k",         6),
        ("ndcg",            "ndcg@k",        7),
        ("precision",       "prec@k",        7),
        ("hard_neg_rate",   "hn@k",          6),
        ("hard_neg_count",  "hn_cnt",        6),
        ("dense_latency_avg_ms", "dense_ms", 8),
        ("bm25_search_latency_avg_ms", "bm25_ms", 8),
        ("rerank_latency_avg_ms", "rerank_ms", 9),
        ("latency_p95_ms",  "p95_ms",        8),
        ("judge_faith",     "faith",         5),
        ("judge_rel",       "rel",           5),
    ]

    def cell(row: Dict, key: str, width: int) -> str:
        v = row.get(key)
        if v is None:
            s = "-"
        elif isinstance(v, float):
            s = f"{v:.3f}"
        else:
            s = str(v)
        return s[:width].ljust(width)

    header = "  ".join(label.ljust(w) for _, label, w in cols)
    sep    = "  ".join("-" * w for _, _, w in cols)
    print("\n=== eval log ===")
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(cell(row, key, w) for key, _, w in cols))
    print()


def main() -> None:
    import argparse
    import datetime
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to eval_dataset.jsonl")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true", help="Disable LLM-as-judge scoring")
    ap.add_argument("--rewrite", action="store_true", help="Enable query rewriting before retrieval")
    ap.add_argument("--hyde", action="store_true", help="Use HyDE: generate hypothetical document passage for retrieval")
    ap.add_argument("--reranker", action="store_true", help="Use cross-encoder reranker (BAAI/bge-reranker-v2-m3)")
    ap.add_argument("--reranker-model", default=None)
    ap.add_argument("--reranker-type", default=None)
    ap.add_argument("--reranker-fetch-k", type=int, default=None)
    ap.add_argument("--reranker-max-length", type=int, default=None)
    ap.add_argument("--reranker-batch-size", type=int, default=None)
    ap.add_argument("--reranker-device", default=None)
    fp16_group = ap.add_mutually_exclusive_group()
    fp16_group.add_argument("--reranker-use-fp16", dest="reranker_use_fp16", action="store_true", default=None)
    fp16_group.add_argument("--no-reranker-use-fp16", dest="reranker_use_fp16", action="store_false")
    ap.add_argument("--reranker-normalize", action="store_true", default=None)
    ap.add_argument("--reranker-instruction", default=None)
    ap.add_argument("--out", default="eval_report.json", help="Output JSON report")
    ap.add_argument("--model", default="", help="Short description of model/config for the log (e.g. 'Qwen3-8B bge-m3 top6')")
    ap.add_argument("--log", default=EVAL_LOG, help=f"Path to eval log file (default: {EVAL_LOG})")
    ap.add_argument("--print-log", action="store_true", help="Print the historical eval log table after this run.")
    ap.add_argument("--llm-base-url", default=TEST_LLM_BASE_URL, help="Override OpenAI-compatible base URL")
    ap.add_argument("--llm-model", default=TEST_LLM_MODEL)
    ap.add_argument("--qdrant-url", default=None, help="Override Qdrant URL, e.g. http://127.0.0.1:6333")
    ap.add_argument("--qdrant-collection", default=None, help="Override Qdrant collection")
    ap.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default="dense",
                    help="Retrieval backend: dense Qdrant vectors, CPU BM25 over Qdrant payloads, or RRF hybrid.")
    ap.add_argument("--bm25-fetch-k", type=int, default=None)
    ap.add_argument("--hybrid-fetch-k", type=int, default=None)
    ap.add_argument("--hybrid-rrf-k", type=int, default=None)
    ap.add_argument("--context-order", choices=sorted(CONTEXT_ORDER_MODES), default=None,
                    help="Order retrieved chunks in the LLM context. rerank_132 splits reranked output into thirds and uses 1-3-2.")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--embed-backend", default=None)
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--embed-batch-size", type=int, default=None)
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
    ap.add_argument("--match-mode", choices=MATCH_MODES, default="strict")
    ap.add_argument("--overlap-threshold", type=float, default=0.25)
    args = ap.parse_args()

    configure_no_proxy(TEST_LLM_NO_PROXY_HOST)
    os.environ["LLM_BASE_URL"] = args.llm_base_url
    os.environ["LLM"] = args.llm_model
    if args.qdrant_url:
        os.environ["QDRANT_URL"] = args.qdrant_url
    if args.qdrant_collection:
        os.environ["QDRANT_COLLECTION"] = args.qdrant_collection
    if args.embed_model:
        os.environ["EMBED_MODEL"] = args.embed_model
    if args.embed_device:
        os.environ["EMBED_DEVICE"] = args.embed_device

    report = evaluate_dataset(
        args.dataset,
        k=args.k,
        use_llm_judge=not args.no_judge,
        use_rewrite=args.rewrite,
        use_hyde=args.hyde,
        use_reranker=args.reranker,
        reranker_model=args.reranker_model,
        reranker_type=args.reranker_type,
        reranker_fetch_k=args.reranker_fetch_k,
        reranker_max_length=args.reranker_max_length,
        reranker_batch_size=args.reranker_batch_size,
        reranker_device=args.reranker_device,
        reranker_use_fp16=args.reranker_use_fp16,
        reranker_normalize=args.reranker_normalize,
        reranker_instruction=args.reranker_instruction,
        embed_model=args.embed_model,
        embed_backend=args.embed_backend,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
        embed_max_seq_length=args.embed_max_seq_length,
        embed_normalize=args.embed_normalize,
        embed_trust_remote_code=args.embed_trust_remote_code,
        embed_query_prefix=args.query_prefix,
        embed_passage_prefix=args.passage_prefix,
        embed_query_instruction=args.query_instruction,
        qdrant_collection=args.qdrant_collection,
        retrieval_mode=args.retrieval_mode,
        bm25_fetch_k=args.bm25_fetch_k,
        hybrid_fetch_k=args.hybrid_fetch_k,
        hybrid_rrf_k=args.hybrid_rrf_k,
        context_order=args.context_order,
        match_mode=args.match_mode,
        overlap_threshold=args.overlap_threshold,
        method=args.model or None,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    m = report["metrics"]
    k_used = args.k or RAGConfig().top_k
    n = len(report["detailed"])

    log_row: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":     args.model or "-",
        "retrieval_mode": args.retrieval_mode,
        "dataset":   Path(args.dataset).name,
        "k":         k_used,
        "n":         n,
        "hit_rate":  m.get(f"recall@{k_used}"),
        "mrr":       m.get(f"mrr@{k_used}"),
        "ndcg":      m.get(f"ndcg@{k_used}"),
        "precision": m.get(f"precision@{k_used}"),
        "hard_neg_rate": m.get(f"hard_negative_rate@{k_used}"),
        "hard_neg_count": m.get(f"avg_hard_negatives@{k_used}"),
        "doc_hit": m.get(f"doc_hit@{k_used}"),
        "overlap_hit": m.get(f"overlap_hit@{k_used}"),
        "embedding_query_latency_avg_ms": m.get("embedding_query_latency_avg_ms"),
        "dense_search_latency_avg_ms": m.get("dense_search_latency_avg_ms"),
        "dense_latency_avg_ms": m.get("dense_latency_avg_ms"),
        "bm25_search_latency_avg_ms": m.get("bm25_search_latency_avg_ms"),
        "hybrid_fusion_latency_avg_ms": m.get("hybrid_fusion_latency_avg_ms"),
        "rerank_latency_avg_ms": m.get("rerank_latency_avg_ms"),
        "latency_p95_ms": m.get("latency_p95_ms"),
        "judge_faith": m.get("judge_faithfulness_avg_0_2"),
        "judge_rel":   m.get("judge_relevance_avg_0_2"),
    }
    _append_log(log_row, args.log)

    # per-source log: one row per source_code
    if m.get("per_source"):
        source_log = Path(args.log).parent / Path(EVAL_LOG_SOURCE).name
        for sc, sm in m["per_source"].items():
            _append_log({
                "timestamp": log_row["timestamp"],
                "model":     log_row["model"],
                "dataset":   log_row["dataset"],
                "k":         k_used,
                "source_code": sc,
                **sm,
            }, str(source_log))

    # print overall metrics (without per_source noise)
    m_print = {kk: vv for kk, vv in m.items() if kk != "per_source"}
    print(json.dumps(m_print, ensure_ascii=False, indent=2))

    # per-source table
    if m.get("per_source"):
        k_used = args.k or RAGConfig().top_k
        print(f"\n--- per source (hit_rate@{k_used}) ---")
        rows = sorted(m["per_source"].items(), key=lambda x: x[1].get(f"hit_rate@{k_used}", 0))
        for sc, sm in rows:
            bar = "█" * int(sm.get(f"hit_rate@{k_used}", 0) * 20)
            print(f"  {sc:<28} n={sm['n']:<4} hit={sm.get(f'hit_rate@{k_used}', 0):.3f}  {bar}")

    print(f"\nSaved report to: {args.out}")

    if args.print_log:
        _print_log_table(args.log)
    cleanup_cuda_cache()


if __name__ == "__main__":
    main()
