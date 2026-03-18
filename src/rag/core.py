from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence

from dotenv import load_dotenv
load_dotenv()

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass(frozen=True)
class RAGConfig:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8080")
    llm_model: str = os.getenv("LLM", "Qwen2.5-7B-Instruct")  # llama.cpp может игнорировать

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # 0.0 = не фильтровать

    # Prompting
    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "18000"))
    max_chunk_chars: int = int(os.getenv("RAG_MAX_CHUNK_CHARS", "2500"))

    # LLM generation
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "900"))
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "120"))

    # Embeddings
    embed_model_name: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    embed_device: str = os.getenv("EMBED_DEVICE", "cuda")  # "cuda" / "cpu"

    # Query rewriting / HyDE
    use_rewrite: bool = os.getenv("RAG_USE_REWRITE", "0") not in ("0", "false", "no")
    use_hyde: bool = os.getenv("RAG_USE_HYDE", "0") not in ("0", "false", "no")

    # Reranker
    use_reranker: bool = os.getenv("RAG_USE_RERANKER", "0") not in ("0", "false", "no")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_top_k: int = int(os.getenv("RAG_RERANKER_FETCH_K", "12"))  # сколько брать из Qdrant перед rerank
    
    
def load_reranker(model_name: str, device: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device, token=os.getenv("HF_TOKEN"))


def load_embedder(model_name: str) -> SentenceTransformer:
    # device можно не задавать — SentenceTransformer сам подхватит cuda, если есть
    return SentenceTransformer(model_name, device=os.getenv("EMBED_DEVICE", "cuda"), token=os.getenv("HF_TOKEN"))
    
@dataclass
class STEmbedder:
    model_name: str

    def __post_init__(self) -> None:
        self.model = load_embedder(self.model_name)

    @property
    def dim(self) -> int:
        # безопасно и быстро
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 64) -> List[List[float]]:
        """
        Возвращает list[vector] где vector = list[float], удобно для Qdrant.
        normalize_embeddings=True обычно улучшает cosine.
        """
        clean = [str(t) for t in texts if t is not None]
        embs = self.model.encode(
            clean,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if isinstance(embs, np.ndarray):
            return embs.astype(np.float32).tolist()
        return np.asarray(embs, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> List[float]:
        text = str(text or "").strip()
        if not text:
            raise ValueError("embed_query: пустая строка запроса")
        return self.embed_texts([text], batch_size=1)[0]
    
    
class Retriever:
    def __init__(self, cfg: RAGConfig, embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.qdrant = QdrantClient(url=cfg.qdrant_url)
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = load_reranker(self.cfg.reranker_model, self.cfg.embed_device)
        return self._reranker

    def _qdrant_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        qvec = self.embedder.embed_query(query)
        resp = self.qdrant.query_points(
            collection_name=self.cfg.qdrant_collection,
            query=qvec,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for p in resp.points:
            if p.score < self.cfg.score_threshold:
                continue
            payload = p.payload or {}
            out.append(
                {
                    "id": p.id,
                    "score": float(p.score) if p.score is not None else None,
                    "text": payload.get("text") or payload.get("chunk") or "",
                    "payload": payload,
                }
            )
        return out

    def search(self, query: str) -> List[Dict[str, Any]]:
        if self.cfg.use_reranker:
            return self._search_rerank(query)
        return self._qdrant_search(query, limit=self.cfg.top_k)

    def _search_rerank(self, query: str) -> List[Dict[str, Any]]:
        # 1. fetch more candidates from Qdrant
        candidates = self._qdrant_search(query, limit=self.cfg.reranker_top_k)
        if not candidates:
            return []

        # 2. rerank with cross-encoder
        reranker = self._get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs, batch_size=len(pairs), show_progress_bar=False)

        # 3. sort by reranker score, keep top_k
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = []
        for rerank_score, chunk in ranked[:self.cfg.top_k]:
            chunk = dict(chunk)
            chunk["rerank_score"] = float(rerank_score)
            result.append(chunk)
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

    for ch in chunks:
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
            }
        )

    context = sep.join(ctx_parts)
    return context, sources


class LlamaCppChatClient:
    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.url = cfg.llm_base_url.rstrip("/") + "/v1/chat/completions"

    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
        if r.status_code != 200:
            # llama.cpp обычно кладёт причину сюда
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
    
REWRITE_PROMPT = """Ты — эксперт по налоговому праву РФ. Твоя задача — извлечь суть налогового вопроса и переформулировать его для поиска по базе документов ФНС.

Шаг 1. Определи СУТЬ вопроса — что именно хочет узнать пользователь про налоги/отчётность/документы. Игнорируй лишний контекст: тип бизнеса, регион, историю ситуации, детали про сотрудников и т.д.

Шаг 2. Перефразируй суть с профессиональными терминами ФНС: добавь синонимы, раскрой аббревиатуры, добавь возможные названия документов/форм.

Примеры:
- "у нас ооо в подмосковье торгуем запчастями, нужно ли сдавать отчёт по счёту в латвийском банке?" → "отчёт о движении денежных средств по счёту в иностранном банке КНД 1112521 обязанность резидента валютный контроль"
- "я ип стригу людей, если найму мастера что будет с патентом" → "патентная система налогообложения найм работников превышение численности утрата права ПСН"
- "пришло требование от налоговой не понимаю как считать штраф" → "требование об уплате налога штраф расчёт пени недоимка"

Верни ТОЛЬКО переформулированный поисковый запрос, без пояснений.

Вопрос: {query}"""


HYDE_PROMPT = """Ты — эксперт по налоговому праву РФ. Напиши короткий фрагмент официального документа ФНС, который содержит ответ на вопрос пользователя.

Требования:
- пиши языком официальных писем/приказов ФНС (сухой, юридический стиль)
- используй профессиональные термины, названия форм, статей НК РФ
- не отвечай на вопрос напрямую — имитируй выдержку из документа
- 3-5 предложений

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


SYSTEM_PROMPT = """Ты — помощник по налоговым документам ФНС РФ. Работаешь ТОЛЬКО по предоставленному контексту (выдержкам документов).

ЖЁСТКИЕ ПРАВИЛА
1) НЕЛЬЗЯ добавлять факты, нормы, даты, номера писем/приказов, если их нет в контексте.
2) Если контекста недостаточно для точного ответа — прямо скажи “Недостаточно данных в контексте” и перечисли, какие сведения/фрагменты нужны.
3) Если источники противоречат друг другу — не выбирай “по ощущениям”. Опиши конфликт и укажи оба источника.
4) Не пересказывай весь документ. Используй только релевантные фрагменты.
5) Всегда привязывай утверждения к источникам: после ключевой фразы ставь [n] (номер источника из контекста).
ПРАВИЛО РЕЛЕВАНТНОСТИ
6) Отвечай ТОЛЬКО на вопросы, на которые можно ответить на основе предоставленного контекста. Любые общие знания вне контекста запрещены.
7) Если вопрос НЕ релевантен (не про ФНС/налоги/документы, либо контекст не помогает), откажись:
   - скажи “Вопрос нерелевантен моему назначению / нет данных в контексте”
   - предложи 1–2 уточняющих направления, как переформулировать вопрос по документам ФНС
   - НЕ пытайся отвечать “в общем”

ЗАДАЧА
- Понять вопрос пользователя.
- Найти в контексте релевантные фрагменты.
- Сформулировать ответ кратко и юридически аккуратно.

ФОРМАТ ОТВЕТА (всегда)
1) Краткий ответ (2–5 предложений, без воды)
2) Обоснование:
   - пунктами, с привязкой к источникам [n]
   - если есть условия/исключения — перечисли их отдельно
3) Источники: [1], [2], ...

ЕСЛИ НЕДОСТАТОЧНО ДАННЫХ
Скажи:
- “Недостаточно данных в контексте.”
- “Чтобы ответить точно, нужны: …”
- “Что уже есть в контексте и почему этого мало: …”
"""

def rag_answer(
    cfg: RAGConfig,
    retriever: Retriever,
    llm: LlamaCppChatClient,
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
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

    return {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(chunks),
    }