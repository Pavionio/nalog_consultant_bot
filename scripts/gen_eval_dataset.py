"""
Generate eval JSONL for RAG retrieval quality evaluation.

Usage:
    python scripts/gen_eval_dataset.py --out eval_dataset.jsonl --per-source 8
    python scripts/gen_eval_dataset.py --out eval_hard_dataset.jsonl --difficulty hard --per-source 18
    python scripts/gen_eval_dataset.py --out eval_superhard_dataset.jsonl --difficulty superhard --candidate-k 20
    python scripts/gen_eval_dataset.py --dry-run

Output records keep backward-compatible fields (`id`, `query`, `relevant`) and add:
`difficulty`, `query_type`, `source_chunk`, `hard_negatives`, `metadata`.
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from src.rag.core import RAGConfig, LlamaCppChatClient, STEmbedder

TEST_LLM_BASE_URL = "http://172.18.96.1:1234"
TEST_LLM_MODEL = "openai/gpt-oss-20b"
TEST_LLM_NO_PROXY_HOST = "172.18.96.1"
GENERATION_TEMPERATURE = 0.8

DIVERSITY_PROFILES = [
    "физлицо",
    "ИП на патенте",
    "самозанятый",
    "ООО на ОСНО",
    "НКО",
    "пенсионер",
    "арендодатель",
    "маркетплейс-продавец",
    "фрилансер с иностранными доходами",
    "кафе/розница",
]

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

Профиль пользователя: {profile}

Задача: по фрагменту документа составь ОДИН вопрос, имитирующий реального пользователя — не юриста.

Требования:
- ответ содержится в этом фрагменте
- вопрос должен быть совместим с профилем пользователя, если это не противоречит фрагменту
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

Профиль пользователя: {profile}

Задача: по фрагменту документа составь ОДИН вопрос от реального пользователя с низкой грамотностью.

Требования:
- ответ содержится в этом фрагменте
- вопрос должен быть совместим с профилем пользователя, если это не противоречит фрагменту
- вопрос сформулирован криво, с лишней информацией, отвлекающим контекстом
- добавь нерелевантные детали про бизнес/ситуацию пользователя
- используй просторечие, опечатки, грамматические ошибки, пропуски слов
- вопрос может быть незаконченным или размытым
- тема должна соответствовать фрагменту, но сформулирована настолько косвенно, что без контекста непонятно о чём

Примеры стиля (ВАЖНО: каждый раз придумывай новый контекст — разный тип бизнеса, регион, ситуацию):
- "у нас ооо торгуем стройматриалами в спб работаем с юриками, бухгалтер говорит надо какойто отчот сдавать в налоговую по нашему счоту в латвийском банке, это вобще обязательно и когда?"
- "я ип на патенте, салон красоты, 3 мастера, сам тоже стригу иногда, скажите если я найму ещо одного мастера что будет с патентом мне его надо переделывать или нет"
- "мама пенсионерка продала дачу в тверской области которую получила в наследство от бабушки три года назад, теперь говорят надо платить налог, это правда и сколько"
- "работаю фрилансером делаю сайты, деньги получаю на карту от физиков, зарегистрировался как самозанятый месяц назад, теперь запутался какой налог и куда платить"

Фрагмент:
{text}

Верни ТОЛЬКО текст вопроса, без пояснений."""

VALIDATE_CANDIDATES_PROMPT = """\
Ты — строгий валидатор eval dataset для RAG retrieval по налоговым документам РФ.

Тебе дан вопрос, исходный фрагмент, по которому вопрос был создан, и top-k кандидатов из векторного поиска.

Нужно разметить каждый candidate одним label:
- "relevant": кандидат содержит ответ на вопрос или явно достаточную правовую норму/разъяснение
- "hard_negative": кандидат тематически похож, но не содержит ответа или отвечает на другой нюанс
- "irrelevant": кандидат не по теме вопроса

Правила:
- исходный source_chunk должен быть relevant
- похожий документ нельзя помечать relevant, если в нём нет ответа
- не выдумывай факты вне текста кандидата
- верни STRICT JSON без markdown и пояснений

JSON schema:
{{
  "labels": [
    {{
      "point_id": "string",
      "label": "relevant|hard_negative|irrelevant",
      "reason": "short reason in Russian"
    }}
  ]
}}

Вопрос:
{query}

Исходный фрагмент:
{source_text}

Кандидаты:
{candidates_json}
"""

# Artifacts that indicate the LLM leaked prompt structure instead of a real question
_BAD_PHRASES = ("фрагмент", "приведён", "приведен", "в данном тексте", "из текста")
_NON_WORD_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)


def _is_valid_question(q: str) -> bool:
    q = q.strip()
    if len(q) < 20:
        return False
    q_lower = q.lower()
    return not any(phrase in q_lower for phrase in _BAD_PHRASES)


def normalize_query(query: str) -> str:
    normalized = query.lower().replace("ё", "е")
    normalized = _NON_WORD_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def is_duplicate_query(
    query: str,
    normalized_seen: List[Tuple[str, str]],
    threshold: float,
) -> Tuple[bool, Optional[Tuple[str, float]]]:
    normalized = normalize_query(query)
    best: Optional[Tuple[str, float]] = None
    for seen_raw, seen_norm in normalized_seen:
        score = _similarity(normalized, seen_norm)
        if best is None or score > best[1]:
            best = (seen_raw, score)
    if best and best[1] >= threshold:
        return True, best
    return False, best


def _candidate_key(item: Dict[str, Any]) -> Tuple[str, str, Any]:
    return (str(item.get("source_code") or ""), str(item.get("external_id") or ""), item.get("chunk_i"))


def _compact_chunk(ch: Dict[str, Any], *, include_text: bool = True) -> Dict[str, Any]:
    out = {
        "source_code": ch.get("source_code") or "",
        "external_id": ch.get("external_id") or "",
        "chunk_i": ch.get("chunk_i"),
        "point_id": str(ch.get("point_id") or ch.get("id") or ""),
    }
    if ch.get("score") is not None:
        out["score"] = float(ch["score"])
    if include_text:
        out["text"] = ch.get("text") or ""
    return out


def _relevant_item(ch: Dict[str, Any], *, primary: bool = False) -> Dict[str, Any]:
    item = {
        "external_id": ch.get("external_id") or "",
        "source_code": ch.get("source_code") or "",
        "chunk_i": ch.get("chunk_i"),
    }
    if ch.get("point_id") or ch.get("id"):
        item["point_id"] = str(ch.get("point_id") or ch.get("id"))
    if ch.get("score") is not None:
        item["score"] = float(ch["score"])
    if ch.get("text"):
        item["text"] = ch["text"]
    if primary:
        item["primary"] = True
    return item


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
                "id": str(p.id),
                "point_id": str(p.id),
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
    To get one question per document, deduplicate by (source_code, external_id) first -
    prefer the first chunk (chunk_i=0) when available, otherwise any.
    """
    rng = random.Random(seed)

    by_source: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for ch in chunks:
        sc = ch["source_code"]
        eid = ch["external_id"]
        if not sc or not eid:
            continue
        existing = by_source[sc].get(eid)
        if existing is None:
            by_source[sc][eid] = ch
        elif (ch.get("chunk_i") or 999) < (existing.get("chunk_i") or 999):
            by_source[sc][eid] = ch

    selected: List[Dict[str, Any]] = []
    for _sc, docs in by_source.items():
        doc_list = list(docs.values())
        n = min(per_source, len(doc_list))
        selected.extend(rng.sample(doc_list, n))

    rng.shuffle(selected)
    return selected


def generate_question(
    llm: LlamaCppChatClient,
    text: str,
    difficulty: str,
    profile: Optional[str] = None,
) -> Optional[str]:
    """Call LLM to produce a single question from the chunk text."""
    truncated = text[:2200]
    if difficulty == "superhard":
        prompt = GENERATE_SUPERHARD_PROMPT
    elif difficulty == "hard":
        prompt = GENERATE_HARD_PROMPT
    else:
        prompt = GENERATE_PROMPT

    content = prompt.format(text=truncated, profile=profile or "обычный пользователь")
    msgs = [{"role": "user", "content": content}]
    try:
        raw = llm.chat(msgs, temperature=GENERATION_TEMPERATURE).strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].strip()
        return raw if raw else None
    except Exception as e:
        print(f"  [warn] LLM generation error: {e}", file=sys.stderr)
        return None


def query_candidates(
    client: QdrantClient,
    collection: str,
    embedder: STEmbedder,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    qvec = embedder.embed_query(query)
    resp = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    candidates: List[Dict[str, Any]] = []
    for point in resp.points:
        payload = point.payload or {}
        text = payload.get("text") or payload.get("chunk") or ""
        candidates.append({
            "point_id": str(point.id),
            "source_code": payload.get("source_code") or "",
            "external_id": payload.get("external_id") or payload.get("doc_id") or "",
            "chunk_i": payload.get("chunk_i"),
            "text": text,
            "score": float(point.score) if point.score is not None else None,
        })
    return candidates


def ensure_source_candidate(source_chunk: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source = _compact_chunk(source_chunk, include_text=True)
    existing_keys = {_candidate_key(c) for c in candidates}
    if _candidate_key(source) not in existing_keys:
        return [source] + candidates
    return candidates


def _extract_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def validate_candidates(
    llm: LlamaCppChatClient,
    query: str,
    source_chunk: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Return labels by point_id, or (None, error) if validation failed."""
    prompt_candidates = []
    for c in candidates:
        prompt_candidates.append({
            "point_id": c["point_id"],
            "source_code": c.get("source_code"),
            "external_id": c.get("external_id"),
            "chunk_i": c.get("chunk_i"),
            "score": c.get("score"),
            "text": (c.get("text") or "")[:1200],
        })

    msgs = [{
        "role": "user",
        "content": VALIDATE_CANDIDATES_PROMPT.format(
            query=query,
            source_text=(source_chunk.get("text") or "")[:1800],
            candidates_json=json.dumps(prompt_candidates, ensure_ascii=False, indent=2),
        ),
    }]
    try:
        raw = llm.chat(msgs, temperature=0, max_tokens=1400)
        parsed = _extract_json_object(raw)
        labels: Dict[str, str] = {}
        for item in parsed.get("labels", []):
            point_id = str(item.get("point_id") or "")
            label = str(item.get("label") or "")
            if point_id and label in {"relevant", "hard_negative", "irrelevant"}:
                labels[point_id] = label
        if not labels:
            raise ValueError("validation JSON has no usable labels")
        return labels, None
    except Exception as e:
        return None, str(e)


def fallback_labels(source_chunk: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    source_key = _candidate_key(source_chunk)
    scores = [c["score"] for c in candidates if c.get("score") is not None]
    high_score = max(0.70, statistics.mean(scores)) if scores else 0.70

    for c in candidates:
        point_id = str(c.get("point_id") or "")
        if not point_id:
            continue
        if _candidate_key(c) == source_key:
            labels[point_id] = "relevant"
        elif c.get("source_code") == source_chunk.get("source_code") or (c.get("score") or 0.0) >= high_score:
            labels[point_id] = "hard_negative"
        else:
            labels[point_id] = "irrelevant"
    return labels


def build_record(
    record_id: str,
    query: str,
    source_chunk: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    labels: Dict[str, str],
    difficulty: str,
    query_type: str,
    profile: Optional[str],
    max_relevant: int,
    max_hard_negatives: int,
    validation_error: Optional[str],
) -> Dict[str, Any]:
    source_key = _candidate_key(source_chunk)
    relevant: List[Dict[str, Any]] = [_relevant_item(source_chunk, primary=True)]
    hard_negatives: List[Dict[str, Any]] = []
    relevant_keys = {source_key}
    hard_negative_keys = set()

    for c in candidates:
        key = _candidate_key(c)
        label = labels.get(c["point_id"], "irrelevant")
        if label == "relevant" and key not in relevant_keys and len(relevant) < max_relevant:
            relevant.append(_relevant_item(c))
            relevant_keys.add(key)
        elif label == "hard_negative" and key not in hard_negative_keys and len(hard_negatives) < max_hard_negatives:
            hard_negatives.append(_compact_chunk(c, include_text=True))
            hard_negative_keys.add(key)

    return {
        "id": record_id,
        "query": query,
        "relevant": relevant,
        "difficulty": difficulty,
        "query_type": query_type,
        "source_chunk": _compact_chunk(source_chunk, include_text=True),
        "hard_negatives": hard_negatives,
        "metadata": {
            "profile": profile,
            "candidate_count": len(candidates),
            "validation": "fallback" if validation_error else "llm",
            "validation_error": validation_error,
            "candidate_k": len(candidates),
            "candidates": [_compact_chunk(c, include_text=True) for c in candidates],
        },
    }


def parse_difficulty(args: argparse.Namespace) -> str:
    if args.superhard:
        return "superhard"
    if args.hard:
        return "hard"
    return args.difficulty


def query_type_for(difficulty: str) -> str:
    if difficulty == "superhard":
        return "noisy_profile"
    if difficulty == "hard":
        return "colloquial"
    return "direct"


def configure_no_proxy(host: str) -> None:
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [part.strip() for part in current.split(",") if part.strip()]
    if host not in parts:
        parts.append(host)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def print_stats(
    records: List[Dict[str, Any]],
    skipped: int,
    duplicates: List[Dict[str, Any]],
) -> None:
    written = len(records)
    avg_rel = sum(len(r["relevant"]) for r in records) / written if written else 0.0
    avg_hn = sum(len(r["hard_negatives"]) for r in records) / written if written else 0.0
    by_source = Counter((r["source_chunk"].get("source_code") or "unknown") for r in records)
    by_difficulty = Counter(r.get("difficulty") or "unknown" for r in records)

    print("\n=== Stats ===")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"duplicates skipped: {len(duplicates)}")
    print(f"avg relevant per query: {avg_rel:.2f}")
    print(f"avg hard negatives per query: {avg_hn:.2f}")

    print("distribution by source_code:")
    for source_code, count in by_source.most_common():
        print(f"  {source_code}: {count}")

    print("distribution by difficulty:")
    for difficulty, count in by_difficulty.most_common():
        print(f"  {difficulty}: {count}")

    if duplicates:
        print("top duplicate-like questions:")
        for item in sorted(duplicates, key=lambda x: x["score"], reverse=True)[:10]:
            print(f"  {item['score']:.3f} :: {item['query']} ~~ {item['matched']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate eval JSONL from Qdrant chunks via LLM")
    ap.add_argument("--out", default="eval_dataset.jsonl", help="Output file path")
    ap.add_argument("--per-source", type=int, default=8, help="Questions per source_code (default: 8)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--dry-run", action="store_true", help="Sample chunks only, skip LLM generation")
    ap.add_argument("--difficulty", choices=["easy", "hard", "superhard"], default="easy")
    ap.add_argument("--hard", action="store_true", help="Compatibility alias for --difficulty hard")
    ap.add_argument("--superhard", action="store_true", help="Compatibility alias for --difficulty superhard")
    ap.add_argument("--candidate-k", type=int, default=12, help="Top-N Qdrant candidates to validate (default: 12)")
    ap.add_argument("--max-relevant-per-query", type=int, default=5)
    ap.add_argument("--max-hard-negatives", type=int, default=5)
    ap.add_argument("--dedup-threshold", type=float, default=0.88)
    ap.add_argument("--validate-candidates", dest="validate_candidates", action="store_true", default=True)
    ap.add_argument("--no-validate-candidates", dest="validate_candidates", action="store_false")
    ap.add_argument("--collection", default=None, help="Qdrant collection name (overrides env/default)")
    ap.add_argument("--llm-base-url", default=TEST_LLM_BASE_URL)
    ap.add_argument("--llm-model", default=TEST_LLM_MODEL)
    args = ap.parse_args()

    difficulty = parse_difficulty(args)
    query_type = query_type_for(difficulty)

    configure_no_proxy(TEST_LLM_NO_PROXY_HOST)
    cfg = dataclasses.replace(RAGConfig(), llm_base_url=args.llm_base_url, llm_model=args.llm_model)
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
        print(f"\nWould write up to {len(sample)} items to {args.out}")
        return

    rng = random.Random(args.seed)
    llm = LlamaCppChatClient(cfg)
    print(f"Loading embedder {cfg.embed_model_name!r}...")
    embedder = STEmbedder(cfg.embed_model_name)

    out_path = Path(args.out)
    records: List[Dict[str, Any]] = []
    normalized_seen: List[Tuple[str, str]] = []
    duplicates: List[Dict[str, Any]] = []
    skipped = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, ch in enumerate(sample, 1):
            profile = rng.choice(DIVERSITY_PROFILES) if difficulty in {"hard", "superhard"} else None
            print(f"  [{i}/{len(sample)}] source={ch['source_code']} ext_id={ch['external_id']}", end=" ... ", flush=True)

            q = generate_question(llm, ch["text"], difficulty=difficulty, profile=profile)
            if q is None or not _is_valid_question(q):
                reason = "too short or contains artifacts" if q else "LLM returned empty"
                print(f"skip ({reason})")
                skipped += 1
                continue

            is_dup, match = is_duplicate_query(q, normalized_seen, args.dedup_threshold)
            if is_dup:
                duplicates.append({"query": q, "matched": match[0], "score": match[1]})
                print(f"skip (duplicate {match[1]:.3f})")
                skipped += 1
                continue

            try:
                candidates = query_candidates(client, collection, embedder, q, args.candidate_k)
                candidates = ensure_source_candidate(ch, candidates)
            except Exception as e:
                print(f"skip (Qdrant/embed error: {e})")
                skipped += 1
                continue

            labels: Optional[Dict[str, str]] = None
            validation_error: Optional[str] = None
            if args.validate_candidates:
                labels, validation_error = validate_candidates(llm, q, ch, candidates)
            else:
                validation_error = "disabled"
            if labels is None:
                labels = fallback_labels(ch, candidates)
                if validation_error and validation_error != "disabled":
                    print(f"[validation fallback: {validation_error[:80]}] ", end="")

            record = build_record(
                record_id=f"q{len(records) + 1:04d}",
                query=q,
                source_chunk=ch,
                candidates=candidates,
                labels=labels,
                difficulty=difficulty,
                query_type=query_type,
                profile=profile,
                max_relevant=args.max_relevant_per_query,
                max_hard_negatives=args.max_hard_negatives,
                validation_error=validation_error,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            normalized_seen.append((q, normalize_query(q)))
            print(f"ok rel={len(record['relevant'])} hard_neg={len(record['hard_negatives'])} -> {q[:80]}")

    print_stats(records, skipped=skipped, duplicates=duplicates)
    print(f"Dataset saved to: {out_path.resolve()}")

    if len(records) < 50:
        print(f"[warn] Only {len(records)} items - consider increasing --per-source or checking Qdrant data", file=sys.stderr)


if __name__ == "__main__":
    main()
