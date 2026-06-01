from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv


DATA_DIR = Path("data")
USERS_PATH = DATA_DIR / "bot_users.json"
LOG_PATH = DATA_DIR / "bot_logs.jsonl"
ACCEPT_CALLBACK = "accept_rules"
TELEGRAM_MESSAGE_LIMIT = 4096
SEND_CHUNK_SIZE = 3800

WELCOME_TEXT = (
    "Здравствуйте! Я налоговый консультант. Помогу разобраться с вопросами по налогам и отчётности.\n\n"
    "Перед началом нужно принять правила использования."
)

RULES_TEXT = (
    "Правила использования:\n"
    "1. Бот дает справочную информацию и не является юридически значимой консультацией.\n"
    "2. Ответы строятся по базе документов и RAG-поиску, поэтому могут быть неполными.\n"
    "3. Все сообщения пользователя, вопросы и ответы бота логируются для улучшения качества.\n"
    "4. Не отправляйте персональные данные, коммерческие тайны и другую чувствительную информацию."
)

ACCEPT_PROMPT = "Нажмите кнопку ниже или отправьте /accept, если согласны."

HELP_TEXT = (
    "Я — налоговый консультант. Отвечаю на вопросы по налогам, сборам, страховым взносам, "
    "отчётности, проверкам и льготам на основе официальных налоговых документов и привожу "
    "ссылки на источники.\n\n"
    "Как пользоваться: после принятия правил просто отправьте вопрос обычным сообщением.\n"
    "Пример: «Какой срок уплаты налога по УСН за первый квартал?»\n\n"
    "Важно: бот отвечает на каждый вопрос отдельно — диалоговый режим (multiturn) не "
    "поддерживается: один вопрос — один ответ, предыдущие сообщения не учитываются.\n\n"
    "Команды:\n"
    "/rules — показать правила\n"
    "/accept — принять правила (нужно один раз)\n"
    "/help — это сообщение"
)

NEED_ACCEPT_TEXT = "Сначала примите правила использования: /rules"
RAG_ERROR_TEXT = "Не удалось подготовить ответ. Попробуйте позже."
UNKNOWN_COMMAND_TEXT = "Неизвестная команда. Наберите /help или отправьте налоговый вопрос."

router = Router()
file_lock = asyncio.Lock()
rag_service: Optional[Any] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def user_payload(message_or_callback: Message | CallbackQuery) -> Dict[str, Any]:
    user = message_or_callback.from_user
    chat = getattr(message_or_callback, "chat", None)
    if chat is None and isinstance(message_or_callback, CallbackQuery) and message_or_callback.message:
        chat = message_or_callback.message.chat
    return {
        "user_id": user.id if user else None,
        "username": user.username if user else None,
        "chat_id": chat.id if chat else None,
    }


async def append_log(
    event_type: str,
    message_or_callback: Message | CallbackQuery,
    *,
    user_text: Optional[str] = None,
    bot_answer: Optional[str] = None,
    rewritten_query: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    retrieved_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    record = {
        "timestamp": now_iso(),
        **user_payload(message_or_callback),
        "event_type": event_type,
        "user_message_text": user_text,
        "session_rewritten_query": rewritten_query,
        "bot_answer_text": bot_answer,
        "sources": sources,
        "retrieved_count": retrieved_count,
        "error": error,
    }
    async with file_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def load_users() -> Dict[str, Any]:
    async with file_lock:
        if not USERS_PATH.exists():
            return {}
        try:
            with USERS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


async def save_user_acceptance(message_or_callback: Message | CallbackQuery) -> None:
    payload = user_payload(message_or_callback)
    user_id = payload["user_id"]
    if user_id is None:
        return

    async with file_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        users: Dict[str, Any] = {}
        if USERS_PATH.exists():
            try:
                with USERS_PATH.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    users = loaded
            except json.JSONDecodeError:
                users = {}

        users[str(user_id)] = {
            "user_id": user_id,
            "username": payload["username"],
            "accepted_at": now_iso(),
        }
        with USERS_PATH.open("w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)


async def has_accepted(user_id: int) -> bool:
    users = await load_users()
    return str(user_id) in users


def rules_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Принять правила", callback_data=ACCEPT_CALLBACK)]
        ]
    )


async def send_long_text(message: Message, text: str, parse_mode: Optional[str] = None) -> None:
    text = text or ""
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        await message.answer(text, parse_mode=parse_mode)
        return

    # Делим по переводам строк, поэтому HTML-ссылки (одна строка) не разрываются.
    start = 0
    while start < len(text):
        chunk = text[start : start + SEND_CHUNK_SIZE]
        split_at = chunk.rfind("\n")
        if split_at > 1000:
            chunk = chunk[:split_at]
        await message.answer(chunk, parse_mode=parse_mode)
        start += len(chunk)


async def send_rules(message: Message, *, with_welcome: bool) -> None:
    accepted = bool(message.from_user) and await has_accepted(message.from_user.id)
    if with_welcome:
        await message.answer(WELCOME_TEXT)
    if accepted:
        await message.answer(RULES_TEXT)
    else:
        await message.answer(RULES_TEXT + "\n\n" + ACCEPT_PROMPT, reply_markup=rules_keyboard())
    await append_log("rules_shown", message)


async def accept_rules(message_or_callback: Message | CallbackQuery) -> None:
    await save_user_acceptance(message_or_callback)
    await append_log("rules_accepted", message_or_callback)


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await append_log("start", message, user_text=message.text)
    await send_rules(message, with_welcome=True)


@router.message(Command("rules"))
async def cmd_rules(message: Message) -> None:
    await append_log("message", message, user_text=message.text)
    await send_rules(message, with_welcome=False)


@router.message(Command("accept"))
async def cmd_accept(message: Message) -> None:
    await append_log("message", message, user_text=message.text)
    await accept_rules(message)
    await message.answer("Правила приняты. Теперь можно отправить налоговый вопрос.")


@router.callback_query(F.data == ACCEPT_CALLBACK)
async def callback_accept_rules(callback: CallbackQuery) -> None:
    await accept_rules(callback)
    await callback.answer("Правила приняты")
    if callback.message:
        await callback.message.answer("Правила приняты. Теперь можно отправить налоговый вопрос.")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await append_log("message", message, user_text=message.text)
    await message.answer(HELP_TEXT)


@router.message(F.text)
async def handle_question(message: Message) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        return

    await append_log("message", message, user_text=text)

    # Неизвестные команды (бот без истории: /reset, /new_session и т.п. удалены).
    if text.startswith("/"):
        await message.answer(UNKNOWN_COMMAND_TEXT)
        return

    if not await has_accepted(user_id):
        await message.answer(NEED_ACCEPT_TEXT)
        return

    service = rag_service
    if service is None:
        await append_log("error", message, user_text=text, error="RAG service is not initialized")
        await message.answer(RAG_ERROR_TEXT)
        return

    # Один вопрос — один ответ: история диалога не ведётся (multiturn не поддерживается).
    try:
        await message.bot.send_chat_action(message.chat.id, "typing")
        result = await asyncio.to_thread(service.answer, user_id, text, [])
    except Exception as exc:
        await append_log("error", message, user_text=text, error=str(exc))
        await message.answer(RAG_ERROR_TEXT)
        return

    answer = str(result.get("answer") or "")
    await send_long_text(message, answer, parse_mode="HTML")
    await append_log(
        "answer",
        message,
        user_text=text,
        bot_answer=answer,
        sources=result.get("sources") or [],
        retrieved_count=result.get("retrieved_count"),
    )


async def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set. Add it to .env or export it before starting the bot.")

    from src.bot.rag_service import RagService

    global rag_service
    rag_service = RagService()

    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
