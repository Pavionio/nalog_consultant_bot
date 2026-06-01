from __future__ import annotations

from typing import Any, Dict, List

from src.rag.core import RAGConfig, STEmbedder, Retriever, LlamaCppChatClient, rag_answer


SESSION_REWRITE_PROMPT = """Ты помогаешь Telegram-боту по налоговым документам ФНС РФ.

Нужно переформулировать последний вопрос пользователя в самостоятельный вопрос для RAG-поиска, используя историю текущей сессии.

Правила:
- сохрани исходный смысл последнего вопроса;
- добавь недостающие ссылки на контекст из истории: режим налогообложения, статус ИП/ООО/физлица, объект вопроса, период, документ, если это явно есть в истории;
- не отвечай на вопрос;
- не добавляй факты, которых нет в истории;
- если последний вопрос уже самодостаточный, верни его без существенных изменений.

История сессии:
{history}

Последний вопрос:
{question}

Верни ТОЛЬКО переформулированный самостоятельный вопрос."""


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(история пустая)"

    lines: List[str] = []
    for item in history:
        role = item.get("role", "")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        label = "Пользователь" if role == "user" else "Бот" if role == "assistant" else role or "Сообщение"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(история пустая)"


class RagService:
    """Thin adapter around the public RAG API used by Telegram handlers."""

    def __init__(self) -> None:
        self.cfg = RAGConfig()
        self.embedder = STEmbedder(self.cfg.embed_model_name)
        self.retriever = Retriever(self.cfg, self.embedder)
        self.llm = LlamaCppChatClient(self.cfg)
        self._warmup()

    def _warmup(self) -> None:
        """Загрузить эмбеддер и реранкер в видеопамять сразу при старте.

        Иначе реранкер грузится лениво при первом вопросе пользователя, что даёт
        задержку первого ответа и всплеск CPU на загрузке модели.
        """
        try:
            print("[RagService] warmup: загрузка эмбеддера и реранкера в VRAM...", flush=True)
            self.retriever.search("разогрев модели: налоги и отчётность")
            print("[RagService] warmup завершён.", flush=True)
        except Exception as exc:  # прогрев не должен ронять бот
            print(f"[RagService] warmup пропущен: {exc}", flush=True)

    def _rewrite_question_for_session(self, question: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return question

        try:
            rewritten = self.llm.chat(
                [
                    {
                        "role": "user",
                        "content": SESSION_REWRITE_PROMPT.format(
                            history=_format_history(history),
                            question=question,
                        ),
                    }
                ],
                max_tokens=400,
                temperature=0,
            ).strip()
        except Exception:
            return question

        if len(rewritten) < 10:
            return question
        return rewritten

    def answer(
        self,
        user_id: int,
        question: str,
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        _ = user_id
        rewritten_question = self._rewrite_question_for_session(question, history)
        result = rag_answer(
            self.cfg,
            self.retriever,
            self.llm,
            rewritten_question,
            chat_history=history,
            html_links=True,
        )
        result["original_query"] = question
        result["session_rewritten_query"] = rewritten_question if rewritten_question != question else None
        return result
