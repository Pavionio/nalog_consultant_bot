#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.rag.core import LlamaCppChatClient, RAGConfig


VALIDATORS = ("precision", "completeness", "format")
SCORE_RE = re.compile(r"\{.*\}", re.S)
SCORE_VALUE_RE = re.compile(r'"score"\s*:\s*([1-5])')
BOOL_FIELD_RE = re.compile(r'"(?P<field>missed_gold_relevant_information|covered_main_answer)"\s*:\s*(?P<value>true|false)', re.I)
COMMON_CONTEXT_RULES = """Общие правила по найденному контексту:
- Чанки с меткой gold_relevant — это эталонные правильные чанки для вопроса.
- Чанки с меткой hard_negative — это похожие, но нерелевантные чанки. Их нельзя считать надёжным основанием ответа.
- Чанки с меткой unknown могут быть полезны или бесполезны, но они не размечены как эталонные.
- Если есть gold_relevant chunks, судить ответ нужно прежде всего по ним.
- Не поощрять ответ за использование hard_negative chunks как доказательства.
- Если gold_relevant chunks не найдены, осторожный отказ от ответа может быть правильным.
- Если has_gold_relevant_context=false, а ответ уверенно даёт налоговый или правовой совет без поддержки контекста, precision должна быть низкой.
- Если has_gold_relevant_context=true, а ответ отказывается отвечать и игнорирует релевантный контекст, completeness должна быть низкой.
- Возвращать только строгий JSON.
- Не возвращать markdown.
- Не возвращать пояснения вне JSON."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def clean_text(value: Any, limit: int = 3500) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def format_context_for_judge(record: Dict[str, Any]) -> str:
    retrieved = record.get("retrieved") or []
    if not retrieved:
        return "Найденный контекст отсутствует."
    parts: List[str] = []
    labels_available = any("relevance_label" in x or "is_gold_relevant" in x for x in retrieved)
    if not labels_available:
        parts.append("Метки эталонной релевантности недоступны для этого JSONL.")
    for idx, chunk in enumerate(retrieved, start=1):
        label = chunk.get("relevance_label")
        if label is None:
            if chunk.get("is_gold_relevant") is True:
                label = "gold_relevant"
            else:
                label = "unknown"
        level = chunk.get("relevance_match_level") or "none"
        source = (
            f"source_code={chunk.get('source_code')}, "
            f"external_id={chunk.get('external_id')}, "
            f"chunk_i={chunk.get('chunk_i')}"
        )
        parts.append(
            f"[Чанк {idx}]\n"
            f"Эталонная релевантность: {label}\n"
            f"Уровень совпадения: {level}\n"
            f"Источник: {source}\n"
            f"Текст:\n{clean_text(chunk.get('text'))}"
        )
    return "\n\n".join(parts)


def _examples_precision() -> str:
    return r'''
Примеры для precision:

Пример P1 — полностью точный ответ:
Вопрос:
"нужно ли самозанятому платить НДФЛ с доходов, облагаемых НПД?"

Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Уровень совпадения: chunk
Текст:
"Доходы, облагаемые налогом на профессиональный доход, не подлежат обложению НДФЛ в части доходов, признаваемых объектом налогообложения НПД."

Ответ:
"Нет. Если доход признаётся объектом налогообложения НПД, он не облагается НДФЛ."

Ожидаемый JSON:
{
  "score": 5,
  "metric": "precision",
  "should_abstain": false,
  "did_abstain": false,
  "used_gold_relevant_context": true,
  "used_hard_negative_as_support": false,
  "explanation": "Ответ полностью подтверждён gold_relevant фрагментом и не содержит лишних неподтверждённых утверждений."
}

Пример P2 — ответ использовал hard_negative:
Вопрос:
"нужно ли самозанятому платить НДФЛ с доходов, облагаемых НПД?"

Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Доходы, облагаемые налогом на профессиональный доход, не подлежат обложению НДФЛ в части доходов, признаваемых объектом налогообложения НПД."

[Чанк 2]
Эталонная релевантность: hard_negative
Текст:
"Физические лица обязаны уплачивать НДФЛ с доходов от продажи имущества, если не применяются освобождения."

Ответ:
"Да, самозанятый должен платить НДФЛ, потому что физические лица обязаны платить НДФЛ с доходов."

Ожидаемый JSON:
{
  "score": 1,
  "metric": "precision",
  "should_abstain": false,
  "did_abstain": false,
  "used_gold_relevant_context": false,
  "used_hard_negative_as_support": true,
  "explanation": "Ответ противоречит gold_relevant фрагменту и использует hard_negative как основание."
}

Пример P3 — неподтверждённая конкретика:
Вопрос:
"какой штраф если не подать уведомление?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Непредставление уведомления может повлечь ответственность в соответствии с Налоговым кодексом Российской Федерации. Конкретный размер штрафа зависит от состава правонарушения."
Ответ:
"Штраф составит 200 рублей."
Ожидаемый JSON:
{
  "score": 2,
  "metric": "precision",
  "should_abstain": false,
  "did_abstain": false,
  "used_gold_relevant_context": true,
  "used_hard_negative_as_support": false,
  "explanation": "Контекст не подтверждает конкретную сумму штрафа 200 рублей. Ответ добавляет неподтверждённую конкретику."
}

Пример P4 — правильный отказ при отсутствии gold context:
Вопрос:
"можно ли получить налоговый вычет на покупку криптовалюты?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: unknown
Текст:
"Имущественный налоговый вычет применяется при приобретении жилых домов, квартир, комнат или долей в них."
Ответ:
"В предоставленных документах нет достаточной информации о вычете на покупку криптовалюты."
Ожидаемый JSON:
{
  "score": 5,
  "metric": "precision",
  "should_abstain": true,
  "did_abstain": true,
  "used_gold_relevant_context": false,
  "used_hard_negative_as_support": false,
  "explanation": "Gold relevant context отсутствует, и ответ корректно отказался делать неподтверждённый вывод."
}

Пример P5 — уверенный ответ без оснований:
Вопрос:
"можно ли получить налоговый вычет на покупку криптовалюты?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: unknown
Текст:
"Имущественный налоговый вычет применяется при приобретении жилых домов, квартир, комнат или долей в них."
Ответ:
"Да, такой вычет можно получить по аналогии с имущественным вычетом."
Ожидаемый JSON:
{
  "score": 1,
  "metric": "precision",
  "should_abstain": true,
  "did_abstain": false,
  "used_gold_relevant_context": false,
  "used_hard_negative_as_support": false,
  "explanation": "Ответ делает неподтверждённый налоговый вывод, которого нет в контексте."
}

Пример P6 — в целом точный ответ:
Вопрос:
"нужно ли подавать уведомление по НДФЛ при выплате дохода?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Налоговый агент обязан представить уведомление об исчисленных суммах НДФЛ в установленный срок."
Ответ:
"Если вы являетесь налоговым агентом, уведомление по НДФЛ нужно представить в установленный срок."
Ожидаемый JSON:
{
  "score": 5,
  "metric": "precision",
  "should_abstain": false,
  "did_abstain": false,
  "used_gold_relevant_context": true,
  "used_hard_negative_as_support": false,
  "explanation": "Ответ аккуратно ограничен условием налогового агента и подтверждается gold_relevant контекстом."
}

Пример P7 — красивый формат, но галлюцинация:
Вопрос:
"какой срок подачи уведомления?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Уведомление представляется в установленный срок. Конкретный срок зависит от периода выплаты дохода."
Ответ:
"Краткий вывод: уведомление нужно подать строго до 25 января. Обоснование: такой срок установлен для всех случаев."
Ожидаемый JSON:
{
  "score": 2,
  "metric": "precision",
  "should_abstain": false,
  "did_abstain": false,
  "used_gold_relevant_context": true,
  "used_hard_negative_as_support": false,
  "explanation": "Ответ добавляет неподтверждённую дату и неверно обобщает срок на все случаи."
}
'''


def build_precision_judge_prompt(record: Dict[str, Any], include_examples: bool = True) -> str:
    examples = _examples_precision() if include_examples else ""
    return f"""Ты — строгий валидатор точности ответа налогового RAG-бота.

Задача:
Оцени только precision / фактическую точность ответа по шкале 1–5.
Не оценивай полноту и стиль, кроме случаев, когда стиль мешает понять факты.
Проверяй, подтверждаются ли утверждения ответа найденным контекстом.
Основной источник истины — чанки с эталонной релевантностью gold_relevant.
Чанки hard_negative нельзя использовать как основание правильного ответа.

Важно:
Precision / точность здесь не означает retrieval Precision@K.
Это точность генерации: насколько фактические утверждения ответа подтверждены контекстом и нет ли галлюцинаций.

Общие правила:
- Используй только предоставленный найденный контекст.
- Не используй внешние знания.
- Не засчитывай факты, которые модель могла взять из собственных знаний, если они не подтверждены найденным контекстом.
- Если в найденном контексте нет фактов, достаточных для ответа на вопрос, правильное поведение модели — отказ от ответа.
- Если в найденном контексте нет фактов для ответа, а модель всё равно даёт налоговый или правовой вывод, ставь precision 1 или 2.
- Если модель даёт верный с точки зрения внешнего мира ответ, но этот ответ не подтверждён найденным контекстом, снижай precision: здесь оценивается только grounded answer.
- Если ответ содержит факт, которого нет в gold_relevant или хотя бы в найденном контексте, снижай оценку.
- Если ответ противоречит gold_relevant chunks, ставь 1 или 2.
- Если ответ опирается на hard_negative как на основание, ставь низкую оценку.
- Если gold_relevant context отсутствует, а ответ корректно отказывается отвечать, precision может быть 5.
- Если gold_relevant context отсутствует, а ответ уверенно даёт налоговый совет, precision должна быть 1 или 2.
- Если ответ содержит неподтверждённые сроки, суммы штрафов, номера статей, реквизиты писем или даты, снижай precision.

{COMMON_CONTEXT_RULES}

Шкала Precision:
5 — все фактические утверждения ответа полностью подтверждаются gold_relevant chunks или, если gold_relevant chunks отсутствуют, ответ корректно отказывается отвечать; нет выдуманных норм, сроков, сумм, реквизитов.
4 — ответ в целом точный, есть незначительные неточности, осторожные обобщения или формулировки без существенного искажения смысла.
3 — ответ частично точный, но часть утверждений слабо подтверждена gold_relevant chunks или опирается на unknown chunks.
2 — много неточностей, ответ частично противоречит gold_relevant chunks или использует hard_negative chunks как основание для вывода.
1 — ответ неверный, галлюцинирует, противоречит gold_relevant chunks или даёт неподтверждённый налоговый совет.

{examples}

Алгоритм выставления Precision:
- Шаг 1: Определи, есть ли gold_relevant chunks.
- Шаг 2: Выдели фактические утверждения ответа: сроки, суммы, обязанности, исключения, категории налогоплательщиков, реквизиты.
- Шаг 3: Проверь каждое утверждение по gold_relevant chunks.
- Шаг 4: Если утверждение подтверждено только hard_negative chunk, считай это ошибкой.
- Шаг 5: Если ответа в gold_relevant нет, проверь, отказался ли бот отвечать.
- Шаг 5.1: Если контекст не содержит достаточных фактов, а бот использовал собственные знания или дал вывод по памяти, считай это неподтверждённым ответом.
- Шаг 6: Выбери score 1–5 по шкале.
- Шаг 7: Верни только JSON.

Реальный пример:
Вопрос:
{record.get("query", "")}

Найденный контекст:
{format_context_for_judge(record)}

Ответ:
{record.get("answer", "")}

Верни только JSON:
{{
  "score": <integer 1..5>,
  "metric": "precision",
  "should_abstain": <true|false>,
  "did_abstain": <true|false>,
  "used_gold_relevant_context": <true|false>,
  "used_hard_negative_as_support": <true|false>,
  "explanation": "<короткое объяснение на русском>"
}}"""


def _examples_completeness() -> str:
    return r'''
Примеры для completeness:

Пример C1 — полный ответ, score 5:
Вопрос:
"какой срок подачи уведомления по НДФЛ при выплате доходов?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Для доходов, выплаченных с 1-го по 22-е число месяца, уведомление представляется не позднее 25-го числа этого месяца. Для доходов, выплаченных с 23-го числа до конца месяца, уведомление представляется не позднее 3-го числа следующего месяца."
Ответ:
"Срок зависит от даты выплаты дохода: если доход выплачен с 1-го по 22-е число месяца, уведомление подают не позднее 25-го числа этого месяца; если выплата была с 23-го числа до конца месяца — не позднее 3-го числа следующего месяца."
Ожидаемый JSON:
{
  "score": 5,
  "metric": "completeness",
  "covered_facts": ["срок 25-е число", "срок 3-е число"],
  "missed_items": [],
  "missed_gold_relevant_information": false,
  "covered_main_answer": true,
  "ceiling_applied": null,
  "explanation": "Покрыты оба срока из gold_relevant."
}

Пример C2 — частичный ответ, score 3:
Вопрос:
"какой срок подачи уведомления по НДФЛ при выплате доходов?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Для доходов, выплаченных с 1-го по 22-е число месяца, уведомление представляется не позднее 25-го числа этого месяца. Для доходов, выплаченных с 23-го числа до конца месяца, уведомление представляется не позднее 3-го числа следующего месяца."
Ответ:
"Уведомление нужно подать не позднее 25-го числа."
Ожидаемый JSON:
{
  "score": 3,
  "metric": "completeness",
  "covered_facts": ["срок 25-е число"],
  "missed_items": ["срок 3-е число для выплат с 23-го числа до конца месяца"],
  "missed_gold_relevant_information": true,
  "covered_main_answer": true,
  "ceiling_applied": "max_3_missing_specific_when_asked",
  "explanation": "Есть основной ответ, но пропущен второй срок."
}

Пример C3 — общий ответ без условий, score 3:
Вопрос:
"может ли ИП применять патент при найме работников?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Индивидуальный предприниматель вправе применять патентную систему налогообложения при соблюдении ограничений, включая ограничение по средней численности работников."
Ответ:
"Да, ИП может применять патент."
Ожидаемый JSON:
{
  "score": 3,
  "metric": "completeness",
  "covered_facts": ["ИП может применять патент"],
  "missed_items": ["ограничение по средней численности работников"],
  "missed_gold_relevant_information": true,
  "covered_main_answer": true,
  "ceiling_applied": "max_3_missing_important_condition",
  "explanation": "Основной вывод есть, но пропущено важное условие."
}

Пример C4 — ответ игнорирует gold context, score 2:
Вопрос:
"нужно ли самозанятому платить НДФЛ с доходов на НПД?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: gold_relevant
Текст:
"Доходы, облагаемые налогом на профессиональный доход, не подлежат обложению НДФЛ."
Ответ:
"Нужно смотреть налоговый режим и вид дохода."
Ожидаемый JSON:
{
  "score": 2,
  "metric": "completeness",
  "covered_facts": [],
  "missed_items": ["доходы на НПД не облагаются НДФЛ"],
  "missed_gold_relevant_information": true,
  "covered_main_answer": false,
  "ceiling_applied": "max_2_gold_relevant_ignored",
  "explanation": "Ответ не использует прямой вывод из gold_relevant."
}

Пример C5 — правильный отказ при отсутствии gold context, score 5:
Вопрос:
"можно ли получить вычет на покупку криптовалюты?"
Найденный контекст:
[Чанк 1]
Эталонная релевантность: unknown
Текст:
"Имущественный налоговый вычет применяется при приобретении жилых помещений."
Ответ:
"В предоставленных документах нет достаточной информации для ответа."
Ожидаемый JSON:
{
  "score": 5,
  "metric": "completeness",
  "covered_facts": ["корректный отказ при отсутствии gold_relevant"],
  "missed_items": [],
  "missed_gold_relevant_information": false,
  "covered_main_answer": true,
  "ceiling_applied": null,
  "explanation": "Gold_relevant нет, отказ уместен."
}
'''


def build_completeness_judge_prompt(record: Dict[str, Any], include_examples: bool = True) -> str:
    examples = _examples_completeness() if include_examples else ""
    return f"""Ты — строгий валидатор полноты ответа налогового RAG-бота.

Задача:
Оцени только completeness / полноту ответа по шкале 1–5.
Не оценивай фактическую точность, кроме случаев, когда ошибка мешает полноте.
Не оценивай стиль, кроме случаев, когда из-за стиля невозможно понять, покрыты ли аспекты.
Основной источник того, что должно быть покрыто, — чанки с эталонной релевантностью gold_relevant.

Общие правила:
- Мысленно составь короткий checklist обязательных фактов из gold_relevant chunks.
- В JSON не пиши длинный checklist. Запиши только 1-3 коротких covered_facts и 1-3 коротких missed_items.
- Не ставь 5 за общий или приблизительный ответ. Score 5 возможен только если покрыт весь materially relevant checklist.
- Если answer использует unknown chunks, но пропускает gold_relevant, снижай score.
- Если answer подменяет gold_relevant информацией из hard_negative chunks, score должен быть низким.
- Если gold_relevant context отсутствует и answer корректно отказывается отвечать, completeness может быть 5.
- Если gold_relevant context есть, а answer отказывается отвечать или игнорирует его, completeness низкая.

Ceiling rules:
- Если gold_relevant chunks есть, но answer почти не использует их, score не выше 2.
- Если answer даёт только общий вывод, но пропускает важные условия, исключения, сроки, ставки или лимиты из gold_relevant, score не выше 3.
- Если вопрос спрашивает "когда", "сколько", "куда", "как подать", а gold_relevant содержит эту конкретику, но answer её не даёт, score не выше 3.
- Если answer отвечает только на одну часть составного вопроса, score не выше 3.
- Если answer подменяет gold_relevant информацией из hard_negative или unknown и из-за этого пропускает gold_relevant, score не выше 2.
- Если answer корректно отказывается при наличии достаточного gold_relevant, score не выше 2.
- Если gold_relevant отсутствует, но answer даёт общий неподтверждённый ответ вместо отказа, score обычно 1 или 2.

{COMMON_CONTEXT_RULES}

Шкала Completeness:
5 — answer покрывает весь checklist из gold_relevant: основной вывод, все важные условия, исключения, сроки/ставки/лимиты/порядок действий, если они есть и нужны для вопроса.
4 — покрыт основной вывод и почти весь checklist; пропущена только второстепенная деталь, без которой практический ответ остаётся полноценным.
3 — частичный ответ: основной вывод есть, но пропущен хотя бы один существенный пункт checklist, либо не раскрыта одна часть составного вопроса.
2 — сильно неполный ответ: покрыта малая часть checklist, answer слишком общий, либо значимая часть gold_relevant заменена hard_negative/unknown.
1 — answer почти не отвечает на вопрос, игнорирует gold_relevant, отвечает на другой вопрос или ошибочно отказывается при наличии достаточного gold_relevant.

{examples}

Алгоритм выставления Completeness:
- Шаг 1: Найди gold_relevant chunks.
- Шаг 2: Мысленно выдели обязательные пункты из gold_relevant, релевантные вопросу.
- Шаг 3: Выпиши covered_facts: 1-3 коротких факта из gold_relevant, которые answer покрыл.
- Шаг 4: Выпиши missed_items: 1-3 важных пункта gold_relevant, которые answer пропустил.
- Шаг 5: Определи, есть ли основной ответ на вопрос.
- Шаг 6: Проверь, не заменил ли answer gold_relevant информацию на hard_negative или unknown.
- Шаг 7: Примени ceiling rules и заполни ceiling_applied, если правило ограничило score.
- Шаг 8: Если gold_relevant отсутствует, оцени, корректно ли answer отказывается отвечать.
- Шаг 9: Выбери score 1–5 по шкале.
- Шаг 10: Верни только JSON.

Реальный пример:
Вопрос:
{record.get("query", "")}

Найденный контекст:
{format_context_for_judge(record)}

Ответ:
{record.get("answer", "")}

Верни только JSON:
{{
  "score": <integer 1..5>,
  "metric": "completeness",
  "covered_facts": ["<короткий покрытый факт>", "..."],
  "missed_items": ["<короткий пропущенный пункт>", "..."],
  "missed_gold_relevant_information": <true|false>,
  "covered_main_answer": <true|false>,
  "ceiling_applied": "<название ceiling rule или null>",
  "explanation": "<короткое объяснение на русском>"
}}"""


def _examples_format() -> str:
    return r'''
Примеры для format:

Пример F1 — отличный формат:
Вопрос:
"какой срок подачи уведомления по НДФЛ?"
Ответ:
"Краткий вывод: срок подачи уведомления зависит от даты выплаты дохода.\n\nОбоснование: если доход выплачен с 1-го по 22-е число месяца, уведомление подают не позднее 25-го числа этого месяца. Если доход выплачен с 23-го числа до конца месяца — не позднее 3-го числа следующего месяца.\n\nИсточник: фрагмент 1."
Ожидаемый JSON:
{
  "score": 5,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": true,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ структурирован, читаем, профессионален и содержит источник."
}

Пример F2 — читаемо, но слабая структура:
Вопрос:
"какой срок подачи уведомления по НДФЛ?"
Ответ:
"Срок зависит от выплаты если с 1 по 22 то до 25 числа если с 23 до конца месяца то до 3 числа следующего месяца."
Ожидаемый JSON:
{
  "score": 3,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ понятен, но плохо структурирован, написан одним предложением и без источников."
}

Пример F3 — грубый ответ:
Вопрос:
"нужно ли платить налог?"
Ответ:
"Ну вы бы хоть документы читали. Да, платить надо, это очевидно."
Ожидаемый JSON:
{
  "score": 2,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": true,
  "explanation": "Ответ читаемый, но тон грубый и непрофессиональный для налогового консультанта."
}

Пример F4 — сломанная кодировка:
Вопрос:
"какой срок подачи уведомления?"
Ответ:
"Ð£Ð²ÐµÐ´Ð¾Ð¼Ð»ÐµÐ½Ð¸Ðµ Ð½ÑƒÐ¶Ð½Ð¾ Ð¿Ð¾Ð´Ð°Ñ‚ÑŒ Ð´Ð¾ 25 Ñ‡Ð¸ÑÐ»Ð°."
Ожидаемый JSON:
{
  "score": 1,
  "metric": "format",
  "is_readable": false,
  "has_required_structure": false,
  "has_broken_encoding": true,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ практически нечитаем из-за сломанной кодировки."
}

Пример F5 — сырой HTML:
Вопрос:
"какой срок подачи уведомления?"
Ответ:
"<div><p>Краткий вывод:</p><br>Уведомление нужно подать до 25 числа.&nbsp;</div>"
Ожидаемый JSON:
{
  "score": 2,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": true,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ частично читаем, но содержит сырой HTML и HTML entities, что нарушает формат."
}

Пример F6 — нежелательный LaTeX:
Вопрос:
"как рассчитать налог?"
Ответ:
"Сумма налога считается так: \\( tax = income \\times rate \\). Если доход 100000, то \\frac{100000 * 6}{100}."
Ожидаемый JSON:
{
  "score": 3,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": true,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ читаем, но LaTeX-разметка неуместна для налоговой консультации и ухудшает восприятие."
}

Пример F7 — сломанный markdown:
Вопрос:
"какой срок подачи уведомления?"
Ответ:
"### **КРАТКИЙ ВЫВОД*** | срок | до 25 ||| \n\n```Уведомление надо подать```"
Ожидаемый JSON:
{
  "score": 2,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": true,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ содержит хаотичный markdown, сломанную таблицу и code block без необходимости."
}

Пример F8 — слишком много воды:
Вопрос:
"нужно ли подавать уведомление?"
Ответ:
"В современном налоговом администрировании очень важно своевременно и ответственно подходить к своим обязанностям. Налоговая система устроена так, что каждый налогоплательщик должен понимать свои права и обязанности. В целом уведомление лучше подать."
Ожидаемый JSON:
{
  "score": 3,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ читаемый и вежливый, но содержит много воды и плохо структурирован."
}

Пример F9 — нет источников при strict_citations:
Вопрос:
"какой срок подачи уведомления?"
Ответ:
"Краткий вывод: уведомление нужно подать до 25 числа.\n\nОбоснование: такой срок установлен для соответствующего периода выплаты дохода."
Ожидаемый JSON:
{
  "score": 4,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": true,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": false,
  "explanation": "Ответ структурирован и читаем, но не содержит явного блока источников."
}

Пример F10 — слишком фамильярный ответ:
Вопрос:
"нужно ли платить налог?"
Ответ:
"Да, дружище, налог придётся платить, никуда не денешься :)"
Ожидаемый JSON:
{
  "score": 2,
  "metric": "format",
  "is_readable": true,
  "has_required_structure": false,
  "has_broken_encoding": false,
  "has_raw_html": false,
  "has_unwanted_markdown_or_latex": false,
  "is_rude_or_unprofessional": true,
  "explanation": "Ответ читаемый, но фамильярный и непрофессиональный для налоговой консультации."
}
'''


def build_format_judge_prompt(record: Dict[str, Any], include_examples: bool = True) -> str:
    examples = _examples_format() if include_examples else ""
    prompt_variant = (record.get("generation_metadata") or {}).get("prompt_variant")
    return f"""Ты — строгий валидатор формата, читаемости и профессиональности ответа налогового RAG-бота.

Задача:
Оцени только format / формат по шкале 1–5.
Не оценивай фактическую точность и полноту, кроме случаев, когда формат делает ответ непонятным.
Проверяй:
- читаемость;
- структуру;
- профессиональный тон;
- отсутствие грубости;
- отсутствие сломанной кодировки;
- отсутствие сырого HTML;
- отсутствие нежелательной Markdown/LaTeX разметки;
- отсутствие мусорных символов;
- соблюдение требуемого формата, если prompt_variant требовал структуру;
- наличие источников/ссылок, если они были доступны и требовались.

Общие правила:
- Хороший формат не делает фактически неверный ответ правильным; но в этом валидаторе оценивай именно форму.
- Если ответ грубый, токсичный, насмешливый или непрофессиональный — format не выше 2.
- Если у ответа съехала кодировка, есть кракозябры, нечитаемые символы — format 1 или 2.
- Если ответ содержит сырой HTML вроде <div>, <br>, <p>, &nbsp; — format обычно не выше 3, а если мешает чтению — 1 или 2.
- Если ответ содержит нежелательную Markdown/LaTeX разметку, например необработанные ###, ***, \\[...\\], \\( ... \\), \\frac{{}}, таблицы с поломанными pipes — снижай score.
- Markdown допустим только если он аккуратный и помогает структуре. Сломанный или избыточный markdown ухудшает score.
- LaTeX обычно не нужен в налоговом ответе. Если LaTeX делает ответ менее читаемым, снижай score.
- Если ответ состоит из одного длинного полотна без структуры при сложном вопросе — снижай score.
- Если ответ не содержит источники при prompt_variant strict_citations, score не выше 4, а если источники были явно нужны — не выше 3.
- Если ответ слишком разговорный, грубый или фамильярный для налоговой консультации — снижай score.
- Если ответ читабелен, структурирован, вежлив, без мусора и с понятным выводом — высокий score.

{COMMON_CONTEXT_RULES}

Шкала Format:
5 — ответ хорошо структурирован, читаемый, профессиональный, без лишней воды; соблюдает требуемый формат; нет HTML/битой кодировки/мусорной Markdown или LaTeX разметки; источники указаны, если доступны и требуются.
4 — ответ понятный и профессиональный, но есть небольшие проблемы: неидеальная структура, немного лишнего текста, неполные источники или minor formatting issues.
3 — ответ в целом читаемый, но заметно страдает структура: много воды, слишком длинный абзац, неаккуратный markdown, частично лишняя разметка, слабое оформление источников.
2 — ответ трудно читать: плохая структура, грубоватый или непрофессиональный тон, сырой HTML/Markdown/LaTeX заметно мешает, есть мусорные символы или частично сломанная кодировка.
1 — ответ нечитаемый или почти нечитаемый: сильно съехала кодировка, много HTML/технического мусора, хаотичная разметка, грубый/оскорбительный стиль, формат полностью нарушен.

{examples}

Алгоритм выставления Format:
- Шаг 1: Проверь, можно ли легко прочитать ответ.
- Шаг 2: Проверь, нет ли сломанной кодировки или мусорных символов.
- Шаг 3: Проверь, нет ли сырого HTML, HTML entities, технического мусора.
- Шаг 4: Проверь, нет ли нежелательной или сломанной Markdown/LaTeX разметки.
- Шаг 5: Проверь тон: вежливый ли он, профессиональный ли, нет ли грубости.
- Шаг 6: Проверь структуру: есть ли краткий вывод, обоснование, источники, если они требуются.
- Шаг 7: Проверь читаемость: нет ли длинного полотна, воды, хаоса.
- Шаг 8: Выбери score 1–5 по шкале.
- Шаг 9: Верни только JSON.

Реальный пример:
Вопрос:
{record.get("query", "")}

Вариант промпта генерации:
{prompt_variant}

Найденный контекст:
{format_context_for_judge(record)}

Ответ:
{record.get("answer", "")}

Верни только JSON:
{{
  "score": <integer 1..5>,
  "metric": "format",
  "is_readable": <true|false>,
  "has_required_structure": <true|false>,
  "has_broken_encoding": <true|false>,
  "has_raw_html": <true|false>,
  "has_unwanted_markdown_or_latex": <true|false>,
  "is_rude_or_unprofessional": <true|false>,
  "explanation": "<короткое объяснение на русском>"
}}"""


def build_prompt(metric: str, record: Dict[str, Any], include_examples: bool) -> str:
    if metric == "precision":
        return build_precision_judge_prompt(record, include_examples)
    if metric == "completeness":
        return build_completeness_judge_prompt(record, include_examples)
    if metric == "format":
        return build_format_judge_prompt(record, include_examples)
    raise ValueError(f"Unsupported validator: {metric}")


def parse_validator_response(metric: str, raw: str, latency_ms: int) -> Dict[str, Any]:
    match = SCORE_RE.search(raw or "")
    if not match:
        salvaged = _salvage_validator_response(metric, raw, latency_ms)
        if salvaged is not None:
            return salvaged
        return {"score": None, "metric": metric, "status": "parse_failed", "latency_ms": latency_ms, "raw_response": raw}
    try:
        data = json.loads(match.group(0))
    except Exception:
        salvaged = _salvage_validator_response(metric, raw, latency_ms)
        if salvaged is not None:
            return salvaged
        return {"score": None, "metric": metric, "status": "parse_failed", "latency_ms": latency_ms, "raw_response": raw}
    score = data.get("score")
    if not isinstance(score, int) or score < 1 or score > 5:
        data["score"] = None
        data["status"] = "invalid_score"
        data["latency_ms"] = latency_ms
        data["raw_response"] = raw
        return data
    if metric == "completeness":
        checklist = data.get("gold_checklist")
        if not isinstance(checklist, list):
            checklist = []
        normalized_checklist = []
        for item in checklist:
            if isinstance(item, dict):
                text = str(item.get("item") or "").strip()
                covered = item.get("covered")
                if text:
                    normalized_checklist.append({"item": text, "covered": bool(covered)})
            elif item is not None:
                text = str(item).strip()
                if text:
                    normalized_checklist.append({"item": text, "covered": False})
        data["gold_checklist"] = normalized_checklist

        covered_facts = data.get("covered_facts")
        if not isinstance(covered_facts, list):
            covered_facts = []
        data["covered_facts"] = [str(x).strip() for x in covered_facts if str(x).strip()][:3]

        missed_items = data.get("missed_items")
        if not isinstance(missed_items, list):
            missed_items = []
        data["missed_items"] = [str(x).strip() for x in missed_items if str(x).strip()][:3]
        data["ceiling_applied"] = data.get("ceiling_applied") or None
        if "missed_gold_relevant_information" not in data:
            data["missed_gold_relevant_information"] = bool(data["missed_items"])
        if "covered_main_answer" not in data:
            data["covered_main_answer"] = score >= 3
    data.pop("metric", None)
    data["status"] = "ok"
    data["latency_ms"] = latency_ms
    return data


def _salvage_validator_response(metric: str, raw: str, latency_ms: int) -> Optional[Dict[str, Any]]:
    text = raw or ""
    score_match = SCORE_VALUE_RE.search(text)
    if not score_match:
        return None
    score = int(score_match.group(1))
    data: Dict[str, Any] = {
        "score": score,
        "status": "ok_salvaged",
        "latency_ms": latency_ms,
        "raw_response": raw,
    }
    if metric == "completeness":
        data.update(
            {
                "covered_facts": [],
                "missed_items": [],
                "gold_checklist": [],
                "ceiling_applied": None,
                "missed_gold_relevant_information": score <= 3,
                "covered_main_answer": score >= 3,
            }
        )
        for match in BOOL_FIELD_RE.finditer(text):
            data[match.group("field")] = match.group("value").lower() == "true"
    return data


def run_validator(llm: LlamaCppChatClient, record: Dict[str, Any], metric: str, include_examples: bool, max_tokens: int, temperature: float) -> Dict[str, Any]:
    prompt = build_prompt(metric, record, include_examples)
    started = time.perf_counter()
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {"score": None, "status": "llm_error", "latency_ms": latency_ms, "raw_response": str(exc)}
    latency_ms = int((time.perf_counter() - started) * 1000)
    return parse_validator_response(metric, raw, latency_ms)


def valid_score(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and 1 <= value <= 5 else None


def human_score(record: Dict[str, Any], metric: str) -> Optional[int]:
    value = (record.get("human") or {}).get(metric, -1)
    return value if isinstance(value, int) and 1 <= value <= 5 else None


def judge_score(record: Dict[str, Any], metric: str) -> Optional[int]:
    value = ((record.get("judge") or {}).get(metric) or {}).get("score")
    return valid_score(value)


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rate(values: Iterable[Any]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if bool(v)) / len(vals)


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else None


def kappa_scores(human: List[int], judge: List[int]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    try:
        from sklearn.metrics import cohen_kappa_score
    except Exception:
        return None, None, None, "Install scikit-learn to compute Cohen's kappa"
    if not human:
        return None, None, None, None
    labels = [1, 2, 3, 4, 5]

    def finite_or_none(value: Any) -> Optional[float]:
        value = float(value)
        return value if math.isfinite(value) else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            finite_or_none(cohen_kappa_score(human, judge, labels=labels)),
            finite_or_none(cohen_kappa_score(human, judge, labels=labels, weights="linear")),
            finite_or_none(cohen_kappa_score(human, judge, labels=labels, weights="quadratic")),
            None,
        )


def agreement_for_metric(records: List[Dict[str, Any]], metric: str) -> Tuple[Dict[str, Any], Optional[str]]:
    pairs = [(human_score(r, metric), judge_score(r, metric)) for r in records]
    pairs = [(h, j) for h, j in pairs if h is not None and j is not None]
    if not pairs:
        return {
            "n": 0,
            "exact_agreement": None,
            "within_1_agreement": None,
            "mae": None,
            "cohen_kappa": None,
            "linear_weighted_kappa": None,
            "quadratic_weighted_kappa": None,
        }, None
    hs = [int(h) for h, _ in pairs]
    js = [int(j) for _, j in pairs]
    diffs = [abs(h - j) for h, j in zip(hs, js)]
    kappa, linear, quadratic, warning = kappa_scores(hs, js)
    return {
        "n": len(pairs),
        "exact_agreement": sum(1 for d in diffs if d == 0) / len(diffs),
        "within_1_agreement": sum(1 for d in diffs if d <= 1) / len(diffs),
        "mae": sum(diffs) / len(diffs),
        "cohen_kappa": kappa,
        "linear_weighted_kappa": linear,
        "quadratic_weighted_kappa": quadratic,
    }, warning


def compute_summary(records: List[Dict[str, Any]], input_path: str) -> Dict[str, Any]:
    warnings = set()
    judged_records = sum(1 for r in records if r.get("judge"))
    judge_averages = {
        "precision": avg(judge_score(r, "precision") for r in records),
        "completeness": avg(judge_score(r, "completeness") for r in records),
        "format": avg(judge_score(r, "format") for r in records),
    }
    judge_averages["avg_score"] = avg(judge_averages.values())

    human_averages = {
        "precision": avg(human_score(r, "precision") for r in records),
        "completeness": avg(human_score(r, "completeness") for r in records),
        "format": avg(human_score(r, "format") for r in records),
    }
    human_averages["avg_score"] = avg(human_averages.values())

    agreement: Dict[str, Any] = {}
    for metric in VALIDATORS:
        metric_agreement, warning = agreement_for_metric(records, metric)
        if warning:
            warnings.add(warning)
        agreement[metric] = metric_agreement
    agreement["macro"] = {
        "exact_agreement": safe_mean(agreement[m]["exact_agreement"] for m in VALIDATORS),
        "within_1_agreement": safe_mean(agreement[m]["within_1_agreement"] for m in VALIDATORS),
        "mae": safe_mean(agreement[m]["mae"] for m in VALIDATORS),
        "cohen_kappa": safe_mean(agreement[m]["cohen_kappa"] for m in VALIDATORS),
        "linear_weighted_kappa": safe_mean(agreement[m]["linear_weighted_kappa"] for m in VALIDATORS),
        "quadratic_weighted_kappa": safe_mean(agreement[m]["quadratic_weighted_kappa"] for m in VALIDATORS),
    }

    precision_judges = [((r.get("judge") or {}).get("precision") or {}) for r in records]
    completeness_judges = [((r.get("judge") or {}).get("completeness") or {}) for r in records]
    format_judges = [((r.get("judge") or {}).get("format") or {}) for r in records]
    summaries = [r.get("retrieval_gold_summary") or {} for r in records]

    summary = {
        "input": input_path,
        "total_records": len(records),
        "judged_records": judged_records,
        "human_labeled_precision_count": sum(1 for r in records if human_score(r, "precision") is not None),
        "human_labeled_completeness_count": sum(1 for r in records if human_score(r, "completeness") is not None),
        "human_labeled_format_count": sum(1 for r in records if human_score(r, "format") is not None),
        "fully_human_labeled_count": sum(1 for r in records if all(human_score(r, m) is not None for m in VALIDATORS)),
        "judge_averages": judge_averages,
        "human_averages": human_averages,
        "agreement": agreement,
        "abstention": {
            "should_abstain_rate": rate(x.get("should_abstain") for x in precision_judges),
            "did_abstain_rate": rate(x.get("did_abstain") for x in precision_judges),
            "abstention_alignment_rate": rate(
                (x.get("should_abstain") == x.get("did_abstain"))
                for x in precision_judges
                if x.get("should_abstain") is not None and x.get("did_abstain") is not None
            ),
        },
        "retrieval_context_quality": {
            "records_with_gold_relevant_context_rate": rate(s.get("has_gold_relevant_context") for s in summaries),
            "avg_gold_relevant_retrieved_count": avg(s.get("gold_relevant_retrieved_count") for s in summaries),
            "avg_hard_negative_retrieved_count": avg(s.get("hard_negative_retrieved_count") for s in summaries),
            "avg_unknown_retrieved_count": avg(s.get("unknown_retrieved_count") for s in summaries),
        },
        "judge_behavior": {
            "used_gold_relevant_context_rate": rate(x.get("used_gold_relevant_context") for x in precision_judges),
            "used_hard_negative_as_support_rate": rate(x.get("used_hard_negative_as_support") for x in precision_judges),
            "missed_gold_relevant_information_rate": rate(x.get("missed_gold_relevant_information") for x in completeness_judges),
        },
        "format_defects": {
            "unreadable_rate": rate((not x.get("is_readable")) if x.get("is_readable") is not None else None for x in format_judges),
            "broken_encoding_rate": rate(x.get("has_broken_encoding") for x in format_judges),
            "raw_html_rate": rate(x.get("has_raw_html") for x in format_judges),
            "unwanted_markdown_or_latex_rate": rate(x.get("has_unwanted_markdown_or_latex") for x in format_judges),
            "rude_or_unprofessional_rate": rate(x.get("is_rude_or_unprofessional") for x in format_judges),
            "missing_required_structure_rate": rate((not x.get("has_required_structure")) if x.get("has_required_structure") is not None else None for x in format_judges),
        },
    }
    if warnings:
        summary["warnings"] = sorted(warnings)
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(summary: Dict[str, Any]) -> None:
    human_any = any(summary.get(k, 0) for k in (
        "human_labeled_precision_count",
        "human_labeled_completeness_count",
        "human_labeled_format_count",
    ))
    if not human_any:
        print("Ручные оценки не найдены; Каппа Коэна и метрики согласованности пропущены.")
    print()
    print("| Metric | Precision | Completeness | Format | Macro |")
    print("|---|---:|---:|---:|---:|")
    human_avg = summary["human_averages"]
    judge_avg = summary["judge_averages"]
    agreement = summary["agreement"]
    print(f"| Human avg | {fmt(human_avg.get('precision'))} | {fmt(human_avg.get('completeness'))} | {fmt(human_avg.get('format'))} | {fmt(human_avg.get('avg_score'))} |")
    print(f"| Judge avg | {fmt(judge_avg.get('precision'))} | {fmt(judge_avg.get('completeness'))} | {fmt(judge_avg.get('format'))} | {fmt(judge_avg.get('avg_score'))} |")
    rows = [
        ("Exact agreement", "exact_agreement"),
        ("±1 agreement", "within_1_agreement"),
        ("MAE", "mae"),
        ("Cohen's κ", "cohen_kappa"),
        ("Linear weighted κ", "linear_weighted_kappa"),
        ("Quadratic weighted κ", "quadratic_weighted_kappa"),
    ]
    for label, key in rows:
        print(f"| {label} | {fmt(agreement['precision'].get(key))} | {fmt(agreement['completeness'].get(key))} | {fmt(agreement['format'].get(key))} | {fmt(agreement['macro'].get(key))} |")
    print()
    print("| Format defect | Rate |")
    print("|---|---:|")
    defects = summary["format_defects"]
    print(f"| Unreadable | {fmt(defects.get('unreadable_rate'))} |")
    print(f"| Broken encoding | {fmt(defects.get('broken_encoding_rate'))} |")
    print(f"| Raw HTML | {fmt(defects.get('raw_html_rate'))} |")
    print(f"| Unwanted Markdown/LaTeX | {fmt(defects.get('unwanted_markdown_or_latex_rate'))} |")
    print(f"| Rude/unprofessional | {fmt(defects.get('rude_or_unprofessional_rate'))} |")
    print(f"| Missing required structure | {fmt(defects.get('missing_required_structure_rate'))} |")
    for warning in summary.get("warnings") or []:
        print(f"\nwarning: {warning}")


def judge_record(
    record: Dict[str, Any],
    *,
    llm: LlamaCppChatClient,
    validators: List[str],
    include_examples: bool,
    max_tokens: int,
    temperature: float,
    judge_model: str,
    judge_base_url: str,
    force: bool,
    skip_existing_judge: bool,
) -> Dict[str, Any]:
    existing = record.get("judge") if isinstance(record.get("judge"), dict) else {}
    if existing and skip_existing_judge and not force:
        return record

    judge = dict(existing or {}) if not force else {}
    started = time.perf_counter()
    for metric in validators:
        if metric in judge and not force:
            continue
        judge[metric] = run_validator(llm, record, metric, include_examples, max_tokens, temperature)
    judge["judge_model"] = judge_model
    judge["judge_base_url"] = judge_base_url
    judge["judged_at"] = utc_now()
    judge["latency_ms"] = int((time.perf_counter() - started) * 1000)
    updated = dict(record)
    updated["judge"] = judge
    return updated


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate generation JSONL with separate LLM validators.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-max-tokens", type=int, default=800)
    ap.add_argument("--judge-reasoning-effort", choices=["off", "low", "medium", "high"], default="medium")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing-judge", action="store_true")
    ap.add_argument("--judge-examples", choices=["on", "off"], default="on")
    ap.add_argument("--validators", nargs="+", choices=VALIDATORS, default=list(VALIDATORS))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    cfg = RAGConfig()
    judge_base_url = args.judge_base_url or cfg.llm_base_url
    judge_model = args.judge_model or cfg.llm_model
    configure_no_proxy_for_url(judge_base_url)
    judge_cfg = dataclasses.replace(
        cfg,
        llm_base_url=judge_base_url,
        llm_model=judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        reasoning_effort=args.judge_reasoning_effort,
    )
    llm = LlamaCppChatClient(judge_cfg)

    records = load_jsonl(input_path)
    end = None if args.limit is None else args.offset + args.limit
    selected_indexes = set(range(args.offset, min(len(records), end if end is not None else len(records))))
    include_examples = args.judge_examples == "on"
    out_records: List[Dict[str, Any]] = []
    for idx, record in enumerate(tqdm(records, desc="generation-judges", unit="row")):
        if idx in selected_indexes:
            out_records.append(judge_record(
                record,
                llm=llm,
                validators=list(args.validators),
                include_examples=include_examples,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                force=args.force,
                skip_existing_judge=args.skip_existing_judge,
            ))
        else:
            out_records.append(record)

    out_path = Path(args.out)
    write_jsonl(out_path, out_records)
    summary = compute_summary(out_records, str(input_path))
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\nJSONL: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
