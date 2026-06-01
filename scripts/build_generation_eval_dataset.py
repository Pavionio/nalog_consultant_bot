#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.eval.eval import load_jsonl
from src.rag.core import (
    CONTEXT_ORDER_MODES,
    HYDE_PROMPT,
    REWRITE_PROMPT,
    LlamaCppChatClient,
    RAGConfig,
    Retriever,
    STEmbedder,
    build_context,
    hyde_query,
    rewrite_query,
)


METHODS = {
    "baseline",
    "reranker",
    "rewrite",
    "hyde",
    "rewrite+reranker",
    "hyde+reranker",
    "bm25",
}
PROMPT_VARIANTS = {"strict_citations", "strict_citations_examples", "default", "abstain_if_no_context"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure_no_proxy_for_url(url: str) -> None:
    host = url.removeprefix("http://").removeprefix("https://").split("/", 1)[0].split(":", 1)[0]
    hosts = ["localhost", "127.0.0.1"] if host in {"", "localhost", "127.0.0.1"} else [host]
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [x.strip() for x in current.split(",") if x.strip()]
    for host in hosts:
        if host not in parts:
            parts.append(host)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def normalize_chunk_i(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def target_identity(target: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        str(target.get("source_code")) if target.get("source_code") else None,
        str(target.get("external_id")) if target.get("external_id") else None,
        normalize_chunk_i(target.get("chunk_i")),
    )


def result_identities(result: Dict[str, Any]) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
    payload = result.get("payload") or {}
    identities: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
    source_code = payload.get("source_code")
    external_id = payload.get("external_id")
    chunk_i = payload.get("chunk_i")
    identities.append((
        str(source_code) if source_code else None,
        str(external_id) if external_id else None,
        normalize_chunk_i(chunk_i),
    ))

    matched_child = payload.get("matched_child")
    if isinstance(matched_child, dict):
        matched_chunk_i = matched_child.get("chunk_i")
        if matched_chunk_i is None:
            matched_chunk_i = matched_child.get("child_i")
        identities.append((
            str(matched_child.get("source_code") or source_code) if (matched_child.get("source_code") or source_code) else None,
            str(matched_child.get("external_id") or external_id) if (matched_child.get("external_id") or external_id) else None,
            normalize_chunk_i(matched_chunk_i),
        ))

    child_i = payload.get("child_i")
    if child_i is not None:
        identities.append((
            str(source_code) if source_code else None,
            str(external_id) if external_id else None,
            normalize_chunk_i(chunk_i if chunk_i is not None else child_i),
        ))
    return identities


def match_level(result: Dict[str, Any], targets: Iterable[Dict[str, Any]]) -> str:
    identities = result_identities(result)
    for target in targets:
        ts, te, tc = target_identity(target)
        if not ts or not te:
            continue
        for rs, re, rc in identities:
            if rs == ts and re == te and tc is not None and rc is not None and rc == tc:
                return "chunk"
    for target in targets:
        ts, te, _ = target_identity(target)
        if not ts or not te:
            continue
        for rs, re, _ in identities:
            if rs == ts and re == te:
                return "document"
    return "none"


def label_retrieved(result: Dict[str, Any], relevant: List[Dict[str, Any]], hard_negatives: List[Dict[str, Any]]) -> Dict[str, Any]:
    rel_level = match_level(result, relevant)
    if rel_level != "none":
        return {
            "is_gold_relevant": True,
            "relevance_label": "gold_relevant",
            "relevance_match_level": rel_level,
        }
    hard_level = match_level(result, hard_negatives)
    if hard_level != "none":
        return {
            "is_gold_relevant": False,
            "relevance_label": "hard_negative",
            "relevance_match_level": hard_level,
        }
    return {
        "is_gold_relevant": False,
        "relevance_label": "unknown",
        "relevance_match_level": "none",
    }


def payload_text(result: Dict[str, Any]) -> str:
    payload = result.get("payload") or {}
    return (
        result.get("text")
        or payload.get("text")
        or payload.get("chunk")
        or payload.get("parent_text")
        or payload.get("child_text")
        or ""
    )


def serialize_retrieved(retrieved: List[Dict[str, Any]], record: Dict[str, Any]) -> List[Dict[str, Any]]:
    relevant = record.get("relevant") or []
    if not relevant and record.get("source_chunk"):
        relevant = [record["source_chunk"]]
    hard_negatives = record.get("hard_negatives") or []
    out: List[Dict[str, Any]] = []
    for rank, item in enumerate(retrieved, start=1):
        payload = item.get("payload") or {}
        labels = label_retrieved(item, relevant, hard_negatives)
        out.append({
            "rank": rank,
            "text": payload_text(item),
            "source_code": payload.get("source_code"),
            "external_id": payload.get("external_id"),
            "chunk_i": payload.get("chunk_i"),
            "score": item.get("score"),
            "score_dense": item.get("score_dense"),
            "score_rerank": item.get("score_rerank") or item.get("rerank_score"),
            "score_bm25": item.get("score_bm25"),
            "title": payload.get("title"),
            "canonical_url": payload.get("canonical_url") or payload.get("url") or payload.get("source_url") or payload.get("doc_url"),
            "doc_id": payload.get("doc_id"),
            "chunk_method": payload.get("chunk_method"),
            "parent_id": payload.get("parent_id"),
            "parent_i": payload.get("parent_i"),
            "child_i": payload.get("child_i"),
            "matched_child": payload.get("matched_child"),
            **labels,
        })
    return out


def retrieval_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold_ranks = [int(x["rank"]) for x in items if x.get("relevance_label") == "gold_relevant"]
    hard_count = sum(1 for x in items if x.get("relevance_label") == "hard_negative")
    unknown_count = sum(1 for x in items if x.get("relevance_label") == "unknown")
    return {
        "retrieved_count": len(items),
        "gold_relevant_retrieved_count": len(gold_ranks),
        "hard_negative_retrieved_count": hard_count,
        "unknown_retrieved_count": unknown_count,
        "has_gold_relevant_context": bool(gold_ranks),
        "gold_relevant_ranks": gold_ranks,
    }


def build_generation_messages(query: str, context: str, variant: str) -> List[Dict[str, str]]:
    if variant == "strict_citations":
        system = """Ты налоговый консультант.
Используй только предоставленный контекст.
Не используй внешние знания.
Не отвечай по памяти и не добавляй факты из собственных знаний.
Если в контексте нет ответа, скажи: "В предоставленных документах нет достаточной информации для ответа."
Если в контексте нет фактов, подтверждающих налоговый или правовой вывод, обязательно откажись отвечать этой фразой.
Ответ должен быть структурирован:
1. Краткий вывод
2. Обоснование по документам
3. Источники
Не выдумывай реквизиты, ссылки, сроки, суммы штрафов и номера статей."""
    elif variant == "strict_citations_examples":
        system = """Ты налоговый консультант. Отвечай только по предоставленному контексту.

Жесткие правила:
- Не используй внешние знания, память модели, общие налоговые знания или догадки.
- Любой налоговый или правовой вывод должен прямо следовать из контекста.
- Если вопрос пользователя не относится к налогам, сборам, страховым взносам, отчетности, проверкам, льготам, налоговым режимам или связанным правовым последствиям, не отвечай по существу и скажи только: "Я отвечаю только на налоговую тематику."
- Если вопрос относится к налогам или праву, но предоставленный контекст не связан с этим вопросом, не пытайся подобрать похожую норму и скажи только: "К сожалению, я не нашел нужную информацию в доступных материалах."
- Если контекст не содержит достаточного основания для ответа, ответь только фразой: "К сожалению, я не нашел нужную информацию в доступных материалах."
- Если в контексте есть только похожая тема, но нет ответа именно на вопрос пользователя, откажись этой же фразой.
- Не заполняй пробелы предположениями. Не пиши "обычно", "как правило", "вероятно", если это не сказано в контексте.
- Не выдумывай реквизиты, номера статей, даты, сроки, суммы, ставки, условия применения льгот и названия документов.
- Не ссылайся на источник, если этот источник не был использован для конкретного вывода.
- Если источники противоречат друг другу или ответа недостаточно, скажи: "К сожалению, я не нашел нужную информацию в доступных материалах."

Формат ответа:
1. Краткий вывод
- 1-3 предложения.
- Только подтвержденный контекстом ответ.

2. Обоснование по документам
- Перечисли ключевые условия и ограничения из контекста.
- Укажи, какие факты из документов подтверждают вывод.
- Если часть вопроса не покрыта контекстом, прямо напиши: "Эта часть вопроса в предоставленных документах не раскрыта."

3. Источники
- Укажи только реально использованные источники из контекста.
- Для каждого источника кратко напиши, какой вывод он подтверждает.

Примеры поведения:
Вопрос: Можно ли применить льготу?
Контекст содержит прямое условие льготы и категорию налогоплательщика.
Ответ: дай вывод, перечисли условия, укажи источник.

Вопрос: Какой штраф будет?
Контекст говорит только о сроке подачи, но не содержит размера штрафа.
Ответ: "К сожалению, я не нашел нужную информацию в доступных материалах."

Вопрос: Нужно ли платить налог ИП на УСН?
Контекст содержит норму только про НДФЛ физлица, без ИП и УСН.
Ответ: "К сожалению, я не нашел нужную информацию в доступных материалах."

Вопрос: Как выбрать ноутбук для бухгалтера?
Контекст содержит письма ФНС и Минфина по налогам.
Ответ: "Я отвечаю только на налоговую тематику."

Вопрос: Какой налоговый режим выгоднее для кафе?
Контекст содержит документы только про сроки подачи уведомлений по НДФЛ.
Ответ: "К сожалению, я не нашел нужную информацию в доступных материалах."

Перед ответом проверь:
- Относится ли вопрос к налоговой или связанной правовой теме?
- Связан ли контекст именно с вопросом пользователя?
- Есть ли в контексте прямой ответ на вопрос?
- Подтвержден ли каждый вывод конкретным источником?
- Нет ли в ответе фактов, которых нет в контексте?
Если хотя бы один ключевой вывод не подтвержден, откажись отвечать."""
    elif variant == "default":
        system = """Ответь по контексту кратко и точно.
Не используй внешние знания.
Не отвечай по памяти и не добавляй факты из собственных знаний.
Если в контексте нет фактов для ответа, скажи: "В предоставленных документах нет достаточной информации для ответа." """
    elif variant == "abstain_if_no_context":
        system = """Если контекст не содержит ответа, обязательно откажись отвечать.
Не делай предположений.
Используй только предоставленный контекст.
Не используй внешние знания и не отвечай по памяти.
Если в контексте нет фактов, подтверждающих налоговый или правовой вывод, скажи: "В предоставленных документах нет достаточной информации для ответа." """
    else:
        raise ValueError(f"Unsupported prompt variant: {variant}")
    user = f"Вопрос:\n{query}\n\nКонтекст:\n{context if context else '(контекст пуст)'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def method_flags(method: str) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unsupported method={method!r}")
    return {
        "use_rewrite": "rewrite" in method,
        "use_hyde": "hyde" in method,
        "use_reranker": "reranker" in method,
        "retrieval_mode": "bm25" if method == "bm25" else "dense",
    }


def build_config(args: argparse.Namespace) -> RAGConfig:
    flags = method_flags(args.method)
    cfg = RAGConfig()
    overrides: Dict[str, Any] = {
        "top_k": args.top_k,
        "use_rewrite": flags["use_rewrite"],
        "use_hyde": flags["use_hyde"],
        "use_reranker": flags["use_reranker"],
        "retrieval_mode": flags["retrieval_mode"],
        "context_order": args.context_order,
    }
    if args.qdrant_url:
        overrides["qdrant_url"] = args.qdrant_url
    if args.qdrant_collection:
        overrides["qdrant_collection"] = args.qdrant_collection
    if args.embed_model:
        overrides["embed_model_name"] = args.embed_model
    if args.reranker_model:
        overrides["reranker_model"] = args.reranker_model
    if args.reranker_fetch_k is not None:
        overrides["reranker_fetch_k"] = args.reranker_fetch_k
        overrides["reranker_top_k"] = args.reranker_fetch_k
    return dataclasses.replace(cfg, **overrides)


def build_embedder(cfg: RAGConfig):
    if cfg.retrieval_mode == "bm25":
        return None
    return STEmbedder(
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


def search_query_for_method(method: str, query: str, llm: LlamaCppChatClient) -> str:
    if "hyde" in method:
        return hyde_query(llm, query)
    if "rewrite" in method:
        return rewrite_query(llm, query)
    return query


def build_record(
    item: Dict[str, Any],
    *,
    dataset_name: str,
    method: str,
    cfg: RAGConfig,
    retriever: Retriever,
    transform_llm: LlamaCppChatClient,
    generator_llm: LlamaCppChatClient,
    generator_base_url: str,
    prompt_variant: str,
    generator_temperature: float,
    generator_max_tokens: int,
) -> Dict[str, Any]:
    query = str(item["query"])
    search_query = search_query_for_method(method, query, transform_llm)
    retrieved = retriever.search(search_query)
    retrieval_timings = getattr(retriever, "last_timings", {}) or {}
    return build_record_from_retrieval(
        item,
        dataset_name=dataset_name,
        method=method,
        cfg=cfg,
        search_query=search_query,
        retrieved=retrieved,
        retrieval_timings=retrieval_timings,
        generator_llm=generator_llm,
        generator_base_url=generator_base_url,
        prompt_variant=prompt_variant,
        generator_temperature=generator_temperature,
        generator_max_tokens=generator_max_tokens,
    )


def build_record_from_retrieval(
    item: Dict[str, Any],
    *,
    dataset_name: str,
    method: str,
    cfg: RAGConfig,
    search_query: str,
    retrieved: List[Dict[str, Any]],
    retrieval_timings: Dict[str, Any],
    generator_llm: LlamaCppChatClient,
    generator_base_url: str,
    prompt_variant: str,
    generator_temperature: float,
    generator_max_tokens: int,
) -> Dict[str, Any]:
    query = str(item["query"])
    retrieved_for_generation = list(retrieved)[: int(cfg.top_k)]
    context, _sources = build_context(cfg, retrieved_for_generation)

    messages = build_generation_messages(query, context, prompt_variant)
    started = time.perf_counter()
    answer = generator_llm.chat(messages, max_tokens=generator_max_tokens, temperature=generator_temperature)
    latency_ms = int((time.perf_counter() - started) * 1000)

    retrieved_rows = serialize_retrieved(retrieved_for_generation, item)
    return {
        "id": str(item.get("id") or ""),
        "dataset": dataset_name,
        "method": method,
        "query": query,
        "search_query": search_query,
        "answer": answer,
        "retrieved": retrieved_rows,
        "retrieval_gold_summary": retrieval_summary(retrieved_rows),
        "generation_metadata": {
            "top_k": cfg.top_k,
            "qdrant_collection": cfg.qdrant_collection,
            "embed_model": cfg.embed_model_name,
            "reranker_model": cfg.reranker_model if cfg.use_reranker else None,
            "reranker_fetch_k": cfg.reranker_fetch_k if cfg.use_reranker else None,
            "context_order": cfg.context_order,
            "prompt_variant": prompt_variant,
            "generator_model": generator_llm.cfg.llm_model,
            "generator_base_url": generator_base_url,
            "temperature": generator_temperature,
            "max_tokens": generator_max_tokens,
            "generated_at": utc_now(),
            "latency_ms": latency_ms,
            "retrieval_timings": retrieval_timings,
        },
        "human": {
            "precision": -1,
            "completeness": -1,
            "format": -1,
            "comment": "",
        },
        "judge": None,
    }


def precompute_retrieval_rows(
    selected: List[Dict[str, Any]],
    *,
    existing_ids: set[str],
    method: str,
    retriever: Retriever,
    transform_llm: LlamaCppChatClient,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in tqdm(selected, desc="retrieval-precompute", unit="q"):
        qid = str(item.get("id") or "")
        if qid in existing_ids:
            continue
        query = str(item["query"])
        search_query = search_query_for_method(method, query, transform_llm)
        retrieved = retriever.search(search_query)
        rows.append(
            {
                "item": item,
                "search_query": search_query,
                "retrieved": retrieved,
                "retrieval_timings": dict(getattr(retriever, "last_timings", {}) or {}),
            }
        )
    return rows


def retrieval_cache_record(row: Dict[str, Any]) -> Dict[str, Any]:
    item = row["item"]
    return {
        "id": str(item.get("id") or ""),
        "item": item,
        "search_query": row["search_query"],
        "retrieved": row["retrieved"],
        "retrieval_timings": row.get("retrieval_timings") or {},
    }


def load_retrieval_cache(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = row.get("item") or {}
        out.append(
            {
                "item": item,
                "search_query": row.get("search_query") or str(item.get("query") or ""),
                "retrieved": row.get("retrieved") or [],
                "retrieval_timings": row.get("retrieval_timings") or {},
            }
        )
    return out


def unload_retrieval_models(retriever: Optional[Retriever], embedder: Any) -> None:
    def drop_model_refs(obj: Any, seen: Optional[set[int]] = None) -> None:
        if obj is None:
            return
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        # First descend into common wrapper attributes so inner CUDA modules are
        # dereferenced before the wrapper itself is nulled.
        for attr in ("model", "inner", "_fallback", "cross_encoder"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                drop_model_refs(child, seen)

        for attr in (
            "model",
            "inner",
            "tokenizer",
            "cross_encoder",
            "_fallback",
            "torch",
            "client",
            "session",
        ):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, None)
                except Exception:
                    pass

    if retriever is not None and hasattr(retriever, "_reranker"):
        drop_model_refs(getattr(retriever, "_reranker", None))
        retriever._reranker = None
    if retriever is not None and hasattr(retriever, "embedder"):
        drop_model_refs(getattr(retriever, "embedder", None))
        retriever.embedder = None
    drop_model_refs(embedder)
    try:
        from src.rag import rerankers

        cached_rerankers = list(getattr(rerankers, "_RERANKER_MODEL_CACHE", {}).values())
        for reranker in cached_rerankers:
            drop_model_refs(reranker)
        with rerankers._RERANKER_MODEL_CACHE_LOCK:
            rerankers._RERANKER_MODEL_CACHE.clear()
        if cached_rerankers:
            print(f"Cleared global reranker cache: {len(cached_rerankers)} object(s).")
        del cached_rerankers
    except Exception as exc:
        print(f"Could not clear global reranker cache: {exc}")
    del retriever
    del embedder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated() / (1024**3)
            before_reserved = torch.cuda.memory_reserved() / (1024**3)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after_allocated = torch.cuda.memory_allocated() / (1024**3)
            after_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(
                "CUDA cache cleared after unloading retrieval models: "
                f"allocated {before_allocated:.2f}GB -> {after_allocated:.2f}GB, "
                f"reserved {before_reserved:.2f}GB -> {after_reserved:.2f}GB."
            )
    except Exception as exc:
        print(f"Could not clear CUDA cache after unloading retrieval models: {exc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build generation quality eval JSONL with retrieved context and generated answers.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--qdrant-collection", default=None)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--reranker-model", default=None)
    ap.add_argument("--reranker-fetch-k", type=int, default=None)
    ap.add_argument("--context-order", choices=sorted(CONTEXT_ORDER_MODES), default="rerank_123",
                    help="Order retrieved chunks in the LLM context. rerank_132 uses thirds order 1-3-2.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--generator-model", default=None)
    ap.add_argument("--generator-base-url", default=None)
    ap.add_argument("--generator-temperature", type=float, default=0.0)
    ap.add_argument("--generator-max-tokens", type=int, default=1500)
    ap.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="strict_citations")
    ap.add_argument(
        "--pause-after-reranker",
        action="store_true",
        help="Pause after embedder/retriever/reranker are loaded; press Enter to start LLM generation.",
    )
    ap.add_argument(
        "--precompute-retrieval",
        action="store_true",
        help="Run all retrieval/rerank first, then generate answers from cached retrieved chunks.",
    )
    ap.add_argument(
        "--unload-retrieval-before-generation",
        action="store_true",
        help="After --precompute-retrieval, drop embedder/reranker/retriever and clear CUDA cache before LLM generation.",
    )
    ap.add_argument(
        "--pause-before-generation",
        action="store_true",
        help="Pause after retrieval precompute/unload and before LLM generation; press Enter to continue.",
    )
    ap.add_argument(
        "--retrieval-cache-out",
        default=None,
        help="Write retrieval/rerank results to this JSONL cache.",
    )
    ap.add_argument(
        "--retrieval-cache-in",
        default=None,
        help="Read retrieval/rerank results from this JSONL cache and skip loading embedder/reranker.",
    )
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only compute/write retrieval cache and exit. Requires --retrieval-cache-out.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if args.unload_retrieval_before_generation and not args.precompute_retrieval:
        raise SystemExit("--unload-retrieval-before-generation requires --precompute-retrieval")
    if args.retrieval_only and not args.retrieval_cache_out:
        raise SystemExit("--retrieval-only requires --retrieval-cache-out")
    if args.retrieval_cache_in and args.precompute_retrieval:
        raise SystemExit("--retrieval-cache-in cannot be combined with --precompute-retrieval")
    if args.retrieval_cache_in and args.retrieval_only:
        raise SystemExit("--retrieval-cache-in cannot be combined with --retrieval-only")
    if args.retrieval_cache_in and not Path(args.retrieval_cache_in).exists():
        raise SystemExit(f"Retrieval cache not found: {args.retrieval_cache_in}")

    cfg = build_config(args)
    generator_base_url = args.generator_base_url or cfg.llm_base_url
    generator_model = args.generator_model or cfg.llm_model
    configure_no_proxy_for_url(generator_base_url)
    configure_no_proxy_for_url(cfg.llm_base_url)

    transform_llm = LlamaCppChatClient(cfg)
    generator_llm = None
    if not args.retrieval_only:
        generator_cfg = dataclasses.replace(
            cfg,
            llm_base_url=generator_base_url,
            llm_model=generator_model,
            temperature=args.generator_temperature,
            max_tokens=args.generator_max_tokens,
        )
        generator_llm = LlamaCppChatClient(generator_cfg)

    items = load_jsonl(dataset_path)
    end = None if args.limit is None else args.offset + args.limit
    selected = items[args.offset:end]

    existing_ids = set()
    if out_path.exists() and not args.force:
        existing_ids = {str(x.get("id") or "") for x in read_jsonl(out_path)}
    if args.force and out_path.exists():
        out_path.unlink()

    embedder = None
    retriever = None
    if not args.retrieval_cache_in:
        embedder = build_embedder(cfg)
        retriever = Retriever(cfg, embedder)
        if cfg.use_reranker:
            print("Loading reranker...")
            retriever._get_reranker()
            print("Reranker loaded.")

    if args.pause_after_reranker:
        input("Retriever/reranker ready. Press Enter to start LLM generation...")

    print(f"Dataset: {dataset_path.name}")
    print(f"Output: {out_path}")
    print(f"Method: {args.method}")
    print(f"Context order: {cfg.context_order}")
    print(f"Records selected: {len(selected)}")
    if existing_ids:
        print(f"Existing records in output: {len(existing_ids)}; they will be skipped.")

    precomputed_rows: Optional[List[Dict[str, Any]]] = None
    if args.retrieval_cache_in:
        precomputed_rows = load_retrieval_cache(Path(args.retrieval_cache_in))
        print(f"Loaded retrieval cache: {args.retrieval_cache_in} ({len(precomputed_rows)} records).")
    elif args.precompute_retrieval or args.retrieval_cache_out:
        if retriever is None:
            raise RuntimeError("Retriever is not available for retrieval precompute.")
        precomputed_rows = precompute_retrieval_rows(
            selected,
            existing_ids=existing_ids,
            method=args.method,
            retriever=retriever,
            transform_llm=transform_llm,
        )
        print(f"Retrieval precomputed: {len(precomputed_rows)} records.")
        if args.retrieval_cache_out:
            cache_path = Path(args.retrieval_cache_out)
            write_jsonl(cache_path, (retrieval_cache_record(row) for row in precomputed_rows))
            print(f"Retrieval cache written: {cache_path}")
        if args.retrieval_only:
            print("Retrieval-only mode complete.")
            return
        if args.unload_retrieval_before_generation:
            print("Unloading embedder/retriever/reranker before LLM generation...")
            unload_retrieval_models(retriever, embedder)
            retriever = None
            embedder = None
            print("Retrieval models unloaded.")
        if args.pause_before_generation:
            input("Retrieval stage complete. Press Enter to start LLM generation...")

    generation_items: Iterable[Any] = precomputed_rows if precomputed_rows is not None else selected

    for item_or_precomputed in tqdm(generation_items, desc="generation-eval", unit="q"):
        if precomputed_rows is not None:
            precomputed = item_or_precomputed
            item = precomputed["item"]
        else:
            item = item_or_precomputed
        qid = str(item.get("id") or "")
        if qid in existing_ids:
            continue
        if precomputed_rows is not None:
            if generator_llm is None:
                raise RuntimeError("Generator LLM client is not available in generation mode.")
            row = build_record_from_retrieval(
                item,
                dataset_name=dataset_path.name,
                method=args.method,
                cfg=cfg,
                search_query=precomputed["search_query"],
                retrieved=precomputed["retrieved"],
                retrieval_timings=precomputed["retrieval_timings"],
                generator_llm=generator_llm,
                generator_base_url=generator_base_url,
                prompt_variant=args.prompt_variant,
                generator_temperature=args.generator_temperature,
                generator_max_tokens=args.generator_max_tokens,
            )
        else:
            if retriever is None:
                raise RuntimeError("Retriever is not available; use --precompute-retrieval before unloading retrieval models.")
            if generator_llm is None:
                raise RuntimeError("Generator LLM client is not available in generation mode.")
            row = build_record(
                item,
                dataset_name=dataset_path.name,
                method=args.method,
                cfg=cfg,
                retriever=retriever,
                transform_llm=transform_llm,
                generator_llm=generator_llm,
                generator_base_url=generator_base_url,
                prompt_variant=args.prompt_variant,
                generator_temperature=args.generator_temperature,
                generator_max_tokens=args.generator_max_tokens,
            )
        append_jsonl(out_path, row)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
