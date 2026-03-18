"""
Generate eval_dataset.jsonl for RAG retrieval quality evaluation.

Usage:
    python scripts/gen_eval_dataset.py --out eval_dataset.jsonl --per-source 8
    python scripts/gen_eval_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from src.rag.core import RAGConfig, LlamaCppChatClient

GENERATE_PROMPT = """\
Ты — составитель тестовых вопросов для RAG-системы ФНС РФ.

Задача: по фрагменту документа составь ОДИН конкретный вопрос на русском.
Требования к вопросу:
- ответ содержится именно в этом фрагменте
- вопрос специфичен (не "что такое НДС?", а "какой срок подачи уведомления по НДФЛ при выплате дивидендов?")
- звучит как вопрос реального пользователя

Фрагмент:
{text}

Верни ТОЛЬКО текст вопроса, без пояснений."""

GENERATE_HARD_PROMPT = """\
Ты — составитель сложных тестовых вопросов для RAG-системы ФНС РФ.

Задача: по фрагменту документа составь ОДИН вопрос, имитирующий реального пользователя — не юриста.

Требования:
- ответ содержится в этом фрагменте
- вопрос сформулирован косвенно, разговорным языком (не юридическими терминами)
- допусти 1-2 опечатки или грамматические ошибки (но не в каждом вопросе — примерно в каждом третьем)
- можно использовать сокращения, просторечие, незаконченные фразы
- НЕ используй слова из самого документа дословно — перефразируй

Примеры стиля:
- "сколько платить за патент если я ИП в москве"
- "мне надо сдавать какой-то отчёт по счёту в иностранном банке?"
- "какой штраф если не подать увидомление"  (опечатка намеренная)
- "у меня ООО, нужно ли нам платить за землю под офисом"

Фрагмент:
{text}

Верни ТОЛЬКО текст вопроса, без пояснений."""

GENERATE_SUPERHARD_PROMPT = """\
Ты — составитель очень сложных тестовых вопросов для RAG-системы ФНС РФ.

Задача: по фрагменту документа составь ОДИН вопрос от реального пользователя с низкой грамотностью.

Требования:
- ответ содержится в этом фрагменте
- вопрос сформулирован криво, с лишней информацией, отвлекающим контекстом
- добавь нерелевантные детали про бизнес/ситуацию пользователя (например: "у меня ООО на УСН, мы торгуем запчастями для грузовиков, склад в подмосковье")
- используй просторечие, опечатки, грамматические ошибки, пропуски слов
- вопрос может быть незаконченным или размытым
- тема должна соответствовать фрагменту, но сформулирована настолько косвенно, что без контекста непонятно о чём

Примеры стиля (ВАЖНО: каждый раз придумывай новый контекст — разный тип бизнеса, регион, ситуацию):
- "у нас ооо торгуем стройматриалами в спб работаем с юриками, бухгалтер говорит надо какойто отчот сдавать в налоговую по нашему счоту в латвийском банке, это вобще обязательно и когда?"
- "я ип на патенте, салон красоты, 3 мастера, сам тоже стригу иногда, скажите если я найму ещо одного мастера что будет с патентом мне его надо переделывать или нет"
- "добрый день у нас была проверка налоговая в прошлом месяце, сейчас пришло какоето требование, мы должны заплатить штраф, но там написано про какойто коэфицент я не понимаю как считать сумму"
- "мама пенсионерка продала дачу в тверской области которую получила в наследство от бабушки три года назад, теперь говорят надо платить налог, это правда и сколько"
- "работаю фрилансером делаю сайты, деньги получаю на карту от физиков, зарегистрировался как самозанятый месяц назад, теперь запутался какой налог и куда платить"
- "у нас кафе в казани, работаем на енвд нет подождите его же отменили, не знаю на чём мы теперь, бухгалтер уволился, скажите что нам делать с отчётностью за прошлый квартал"

Фрагмент:
{text}

Верни ТОЛЬКО текст вопроса, без пояснений."""

# Artifacts that indicate the LLM leaked prompt structure instead of a real question
_BAD_PHRASES = ("фрагмент", "приведён", "приведен", "в данном тексте", "из текста")


def _is_valid_question(q: str) -> bool:
    q = q.strip()
    if len(q) < 20:
        return False
    q_lower = q.lower()
    for phrase in _BAD_PHRASES:
        if phrase in q_lower:
            return False
    return True


def scroll_all_chunks(client: QdrantClient, collection: str, batch: int = 100) -> List[Dict[str, Any]]:
    """Scroll through entire Qdrant collection and return all points as dicts."""
    chunks: List[Dict[str, Any]] = []
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=batch,
            offset=offset,
        )
        for p in result:
            payload = p.payload or {}
            text = payload.get("text") or payload.get("chunk") or ""
            if not text.strip():
                continue
            chunks.append({
                "id": p.id,
                "text": text,
                "external_id": payload.get("external_id") or payload.get("doc_id") or "",
                "source_code": payload.get("source_code") or "",
                "chunk_i": payload.get("chunk_i"),
            })
        if next_offset is None:
            break
        offset = next_offset

    return chunks


def stratified_sample(
    chunks: List[Dict[str, Any]],
    per_source: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample up to `per_source` chunks per source_code.
    To get one question per document, deduplicate by (source_code, external_id) first —
    prefer the first chunk (chunk_i=0) when available, otherwise any.
    """
    rng = random.Random(seed)

    # Group by source_code, deduplicate by external_id (keep chunk_i=0 or first seen)
    by_source: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for ch in chunks:
        sc = ch["source_code"]
        eid = ch["external_id"]
        if not sc or not eid:
            continue
        existing = by_source[sc].get(eid)
        if existing is None:
            by_source[sc][eid] = ch
        else:
            # prefer chunk_i == 0
            if (ch.get("chunk_i") or 999) < (existing.get("chunk_i") or 999):
                by_source[sc][eid] = ch

    selected: List[Dict[str, Any]] = []
    for sc, docs in by_source.items():
        doc_list = list(docs.values())
        n = min(per_source, len(doc_list))
        selected.extend(rng.sample(doc_list, n))

    return selected


def generate_question(llm: LlamaCppChatClient, text: str, hard: bool = False, superhard: bool = False) -> Optional[str]:
    """Call LLM to produce a single question from the chunk text."""
    truncated = text[:2000]
    if superhard:
        prompt = GENERATE_SUPERHARD_PROMPT
    elif hard:
        prompt = GENERATE_HARD_PROMPT
    else:
        prompt = GENERATE_PROMPT
    msgs = [
        {"role": "user", "content": prompt.format(text=truncated)},
    ]
    try:
        raw = llm.chat(msgs, temperature=0.8).strip()
        # strip surrounding quotes if the model added them
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].strip()
        return raw if raw else None
    except Exception as e:
        print(f"  [warn] LLM error: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate eval_dataset.jsonl from Qdrant chunks via LLM")
    ap.add_argument("--out", default="eval_dataset.jsonl", help="Output file path")
    ap.add_argument("--per-source", type=int, default=8, help="Questions per source_code (default: 8)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--dry-run", action="store_true", help="Sample chunks only, skip LLM generation")
    ap.add_argument("--hard", action="store_true", help="Generate hard questions: typos, indirect language, colloquial style")
    ap.add_argument("--superhard", action="store_true", help="Generate superhard questions: noisy context, business details, heavy typos")
    ap.add_argument("--collection", default=None, help="Qdrant collection name (overrides env/default)")
    args = ap.parse_args()

    cfg = RAGConfig()
    collection = args.collection or cfg.qdrant_collection

    print(f"Connecting to Qdrant at {cfg.qdrant_url}, collection={collection}")
    client = QdrantClient(url=cfg.qdrant_url)

    print("Scrolling all chunks...")
    chunks = scroll_all_chunks(client, collection)
    print(f"  Total chunks with text: {len(chunks)}")

    sources_seen = {ch["source_code"] for ch in chunks if ch["source_code"]}
    print(f"  Sources: {sorted(sources_seen)}")

    sample = stratified_sample(chunks, per_source=args.per_source, seed=args.seed)
    print(f"  Sampled {len(sample)} chunks ({args.per_source} per source_code)")

    if args.dry_run:
        print("\n--- DRY RUN: sampled chunks (no LLM calls) ---")
        for i, ch in enumerate(sample, 1):
            preview = ch["text"][:120].replace("\n", " ")
            print(f"  [{i:03d}] source={ch['source_code']} ext_id={ch['external_id']} chunk_i={ch.get('chunk_i')}")
            print(f"        {preview}...")
        print(f"\nWould write {len(sample)} items to {args.out}")
        return

    llm = LlamaCppChatClient(cfg)
    out_path = Path(args.out)
    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, ch in enumerate(sample, 1):
            print(f"  [{i}/{len(sample)}] source={ch['source_code']} ext_id={ch['external_id']}", end=" ... ", flush=True)
            q = generate_question(llm, ch["text"], hard=args.hard, superhard=args.superhard)

            if q is None or not _is_valid_question(q):
                reason = "too short or contains artifacts" if q else "LLM returned empty"
                print(f"skip ({reason})")
                skipped += 1
                continue

            record = {
                "id": f"q{written + 1:04d}",
                "query": q,
                "relevant": [
                    {
                        "external_id": ch["external_id"],
                        "source_code": ch["source_code"],
                    }
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"ok → {q[:80]}")

    print(f"\nDone. Written: {written}, Skipped: {skipped}")
    print(f"Dataset saved to: {out_path.resolve()}")

    if written < 50:
        print(f"[warn] Only {written} items — consider increasing --per-source or checking Qdrant data", file=sys.stderr)


if __name__ == "__main__":
    main()
