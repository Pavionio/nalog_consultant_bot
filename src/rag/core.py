from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence

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
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "900"))
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "120"))

    # Embeddings
    embed_model_name: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    embed_device: str = os.getenv("EMBED_DEVICE", "cuda")  # "cuda" / "cpu"
    
    
def load_embedder(model_name: str) -> SentenceTransformer:
    # device можно не задавать — SentenceTransformer сам подхватит cuda, если есть
    return SentenceTransformer(model_name, device=RAGConfig.embed_device)
    
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
        embs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if isinstance(embs, np.ndarray):
            return embs.astype(np.float32).tolist()
        return np.asarray(embs, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], batch_size=1)[0]
    
    
@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    payload: Dict[str, Any]


class Retriever:
    def __init__(self, cfg: RAGConfig, embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.qdrant = QdrantClient(url=cfg.qdrant_url)

    def search(self, query: str):
        qvec = self.embedder.embed_query(query)
        resp = self.qdrant.query_points(
            collection_name=self.cfg.qdrant_collection,
            query=qvec,                  # list[float]
            limit=self.cfg.top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = resp.points

        # приведи к своему формату
        out = []
        for p in points:
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
    
    
def _format_sources(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    sources = []
    for i, ch in enumerate(chunks, start=1):
        sources.append(
            {
                "n": i,
                "score": round(ch.score, 4),
                "source_url": ch.payload.get("source_url") or ch.payload.get("url") or ch.payload.get("doc_url"),
                "doc_id": ch.payload.get("doc_id") or ch.payload.get("external_id"),
                "chunk_id": ch.payload.get("chunk_id"),
            }
        )
    return sources


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

    def get_score(ch) -> float | None:
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

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        r = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
        if r.status_code != 200:
            # llama.cpp обычно кладёт причину сюда
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
    
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
) -> Dict[str, Any]:
    chunks = retriever.search(user_query)
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

    answer = llm.chat(msgs)

    return {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(chunks),
    }