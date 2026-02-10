#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup

import psycopg
from psycopg.rows import dict_row

from chonkie import Pipeline

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import get_handler
from fetch.format_parsers import parse_nalog_calendar_xml, calendar_days_to_rag_text

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from email import policy
from email.parser import BytesParser

from dotenv import load_dotenv
load_dotenv()

UA = "nalog-consultant-bot/0.1"
DEFAULT_TIMEOUT = 60.0


# ----------------------------
# DB
# ----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_doc (
  id              uuid PRIMARY KEY,
  source_code     text NOT NULL,
  external_id     text NOT NULL,
  url             text NOT NULL,
  kind            text NOT NULL,

  title           text,
  content_sha256  text,

  raw_path        text,
  text_path       text,
  chunks_path     text,

  created_at      timestamptz NOT NULL,
  last_seen_at    timestamptz NOT NULL,

  UNIQUE (source_code, external_id)
);

CREATE INDEX IF NOT EXISTS idx_rag_doc_source ON rag_doc(source_code);
"""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def init_db(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def get_doc_id(conn: psycopg.Connection, source_code: str, external_id: str) -> Optional[uuid.UUID]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id FROM rag_doc WHERE source_code=%s AND external_id=%s",
            (source_code, external_id),
        )
        row = cur.fetchone()
    return row["id"] if row else None


def insert_doc(
    conn: psycopg.Connection,
    discovered: DiscoveredDoc,
    title: Optional[str],
    content_sha256: str,
    raw_path: str,
    text_path: str,
    chunks_path: str,
) -> uuid.UUID:
    doc_id = uuid.uuid4()
    now = utcnow()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_doc (
              id, source_code, external_id, url, kind,
              title, content_sha256,
              raw_path, text_path, chunks_path,
              created_at, last_seen_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (source_code, external_id) DO NOTHING
            """,
            (
                doc_id,
                discovered.source_code,
                discovered.external_id,
                discovered.url,
                discovered.kind,
                title,
                content_sha256,
                raw_path,
                text_path,
                chunks_path,
                now,
                now,
            ),
        )

    conn.commit()
    existing = get_doc_id(conn, discovered.source_code, discovered.external_id)
    return existing or doc_id


# ----------------------------
# HTTP / HTML utils
# ----------------------------

def extract_html_from_mime(blob: bytes) -> Tuple[str, str]:
    """
    Returns (html_str, charset_used).
    Works for multipart/related MHTML where HTML is quoted-printable, cp1251, etc.
    """
    msg = BytesParser(policy=policy.default).parsebytes(blob)

    # If not multipart: might already be HTML
    if not msg.is_multipart():
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        html = payload.decode(charset, errors="replace")
        return html, charset

    # Find first text/html part
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = (part.get_content_type() or "").lower()
        if ctype == "text/html" or (ctype.startswith("text/") and "html" in ctype):
            payload = part.get_payload(decode=True) or b""  # <-- decodes quoted-printable to bytes
            charset = part.get_content_charset() or "utf-8"
            html = payload.decode(charset, errors="replace")
            return html, charset

    raise RuntimeError("No text/html part found inside MIME/MHTML container")

def extract_title_str(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        return t[:800] if t else None
    if soup.title:
        t = soup.title.get_text(" ", strip=True)
        return t[:800] if t else None
    return None

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    body = soup.body or soup
    text = body.get_text("\n")   # <-- важно: не strip=True
    text = text.replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)  # сжать пустоты
    return text

def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": UA,
            "Accept-Language": "ru,en;q=0.8",
        }
    )
    return s


def fetch_html(
    s: requests.Session,
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    sleep: float = 0.8,
    retries: int = 2,
) -> bytes:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            r = s.get(url, timeout=timeout)
            time.sleep(sleep)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    assert last_err is not None
    raise last_err


def soup_from_html(html: bytes) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def extract_title(html: bytes) -> Optional[str]:
    soup = soup_from_html(html)
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        return t[:800] if t else None
    if soup.title:
        t = soup.title.get_text(" ", strip=True)
        return t[:800] if t else None
    return None



_NBSP_RE = re.compile(r"\xa0|&nbsp;")
_WS_RE = re.compile(r"[ \t]+\n|\n[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def extract_main_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")

    # 1) Берём самый узкий контейнер, где реально живёт контент страницы
    main = soup.select_one("#divSecondPageColumns") or soup.select_one(".page-content__center") or soup.body
    if main is None:
        return ""

    # 2) Внутри main удаляем заведомый мусор
    # Скрипты/стили/ноускрипт
    for t in main.select("script, style, noscript"):
        t.decompose()

    # Формы/попапы/футерные анкеты внутри контента
    # В твоём HTML форма начинается с блока id="ctl00_ctl03_ctl02_pnlMain"
    for t in main.select(
        "#ctl00_ctl03_ctl02_pnlMain, "
        "#dUserForm, "
        ".mfp-hide, "
        ".popup, "
        "a.js-popup, "
        "#mkgu-widget, "
        ".DoYouFoundWrapper, "
        "#CtrlEnterPopup"
    ):
        t.decompose()

    # Иногда есть “плашки/ссылки” справа (“Оставить отзыв”, “О сервисе”)
    for t in main.select(".div_move_to_right, .link-block"):
        t.decompose()

    # 3) Конвертируем разметку в текст аккуратно
    #    - <br> => перевод строки
    #    - <p>, <div> => текст с переносами
    text = main.get_text("\n", strip=True)

    # 4) Нормализация пробелов/переносов
    text = _NBSP_RE.sub(" ", text)
    text = _WS_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text).strip()

    return text



def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----------------------------
# Chunking (chonkie)
# ----------------------------

def build_chunker() -> Pipeline:
    return (
        Pipeline()
        .chunk_with(
            "recursive",
            tokenizer="word",
            chunk_size=1100,
            recipe="markdown",
            min_characters_per_chunk=1100,
        )
        .refine_with("overlap", context_size=160)
    )


def chunk_text(pipe: Pipeline, text: str) -> List[dict]:
    doc = pipe.run(texts=text)
    return [{"i": i, "text": ch.text} for i, ch in enumerate(doc.chunks)]


# ----------------------------
# Embeddings (local)
# ----------------------------

def load_embedder(model_name: str) -> SentenceTransformer:
    # CPU тоже ок; если есть CUDA — sentence-transformers сам подхватит
    model = SentenceTransformer(model_name)
    return model


def embed_passages(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()



# ----------------------------
# Qdrant
# ----------------------------

def init_qdrant(client: QdrantClient, collection: str, vector_size: int) -> None:
    # создаём коллекцию только если её нет
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def stable_point_id(source_code: str, external_id: str, chunk_i: int) -> str:
    # стабильный UUID: при повторном прогоне будет тот же id -> upsert перезапишет point
    key = f"{source_code}:{external_id}:{chunk_i}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def upsert_chunks_qdrant(
    client: QdrantClient,
    collection: str,
    doc_id: uuid.UUID,
    doc: DiscoveredDoc,
    title: Optional[str],
    kind: str,
    chunks: List[dict],
    vectors: List[List[float]],
) -> None:
    points: List[PointStruct] = []
    for ch, vec in zip(chunks, vectors):
        pid = stable_point_id(doc.source_code, doc.external_id, int(ch["i"]))
        points.append(
            PointStruct(
                id=pid,
                vector=vec,
                payload={
                    "doc_id": str(doc_id),
                    "source_code": doc.source_code,
                    "external_id": doc.external_id,
                    "url": doc.url,
                    "title": title,
                    "kind": kind,
                    "chunk_i": int(ch["i"]),
                    "text": ch["text"],
                },
            )
        )

    if points:
        client.upsert(collection_name=collection, points=points)


# ----------------------------
# Config
# ----------------------------

def load_sources(path: Path) -> List[Source]:
    cfg = yaml.safe_load(path.read_text("utf-8"))
    out: List[Source] = []
    for x in cfg.get("sources", []):
        out.append(
            Source(
                code=x["code"],
                base_url=x["base_url"],
                kind=x.get("kind", "unknown"),
                active=bool(x.get("active", False)),
                handler=x.get("handler") or "",
            )
        )
    return out


# ----------------------------
# Filesystem
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Ingest
# ----------------------------

def ingest_one(
    s: requests.Session,
    conn: psycopg.Connection,
    pipe: Pipeline,
    doc: DiscoveredDoc,
    data_dir: Path,
    sleep: float,
    qdrant: Optional[QdrantClient],
    qdrant_collection: Optional[str],
    embedder: Optional[SentenceTransformer],
) -> str:
    # MVP: если документ уже есть в БД — пропускаем целиком (включая Qdrant)
    if get_doc_id(conn, doc.source_code, doc.external_id):
        return "SKIP_EXISTS"

    raw_dir = data_dir / "raw" / doc.source_code / doc.external_id
    txt_dir = data_dir / "text" / doc.source_code / doc.external_id
    chunk_dir = data_dir / "chunks" / doc.source_code

    ensure_dir(raw_dir)
    ensure_dir(txt_dir)
    ensure_dir(chunk_dir)

    blob = fetch_html(s, doc.url, sleep=sleep)
    head = blob[:2048]
    is_xml = doc.url.lower().endswith(".xml")
    is_mime = re.search(br"^\s*MIME-Version\s*:", head, re.I) is not None \
          or re.search(br"Content-Type\s*:\s*multipart/", head, re.I) is not None
    is_pravo = "pravo.gov.ru" in doc.url or "savertf" in doc.url.lower()

    title: Optional[str] = None
    if is_xml:
        raw_xml_path = raw_dir / "file.xml"
        raw_xml_path.write_bytes(blob)

        cal_title, days = parse_nalog_calendar_xml(blob)
        title = cal_title or f"Налоговый календарь ({doc.external_id})"

        text = calendar_days_to_rag_text(cal_title, days)

        text_path = txt_dir / "doc.txt"
        text_path.write_text(text, "utf-8")

        content_hash = sha256_text(text or "")
        chunks = chunk_text(pipe, text) if text else []
        chunks_path = chunk_dir / f"{doc.external_id}.jsonl"
        write_jsonl(chunks_path, chunks)

        doc_id = insert_doc(
            conn=conn,
            discovered=doc,
            title=title,
            content_sha256=content_hash,
            raw_path=str(raw_xml_path.relative_to(data_dir)),
            text_path=str(text_path.relative_to(data_dir)),
            chunks_path=str(chunks_path.relative_to(data_dir)),
        )

        if qdrant and qdrant_collection and embedder and chunks:
            texts = [c["text"] for c in chunks]
            vectors = embed_passages(embedder, texts, batch_size=32)
            upsert_chunks_qdrant(
                client=qdrant,
                collection=qdrant_collection,
                doc_id=doc_id,
                doc=doc,
                title=title,
                kind=doc.kind,
                chunks=chunks,
                vectors=vectors,
            )

        return "INGESTED_XML"
    
    if is_mime:
        raw_mime_path = raw_dir / "file.mhtml"
        raw_mime_path.write_bytes(blob)

        html_str, charset = extract_html_from_mime(blob)

        # debug-friendly: save extracted html
        raw_html_path = raw_dir / "page.html"
        raw_html_path.write_text(html_str, "utf-8")

        # optional: title from html (можешь позже улучшить)
        title = extract_title_str(html_str)

        text = html_to_text(html_str)

        raw_rel = raw_mime_path.relative_to(data_dir)
    else:
        # обычный HTML
        raw_html_path = raw_dir / "page.html"
        raw_html_path.write_bytes(blob)

        title = extract_title(blob)
        text = extract_main_text(blob)

        raw_rel = raw_html_path.relative_to(data_dir)

    # текст пишем одинаково
    text_path = txt_dir / "doc.txt"
    text_path.write_text(text, "utf-8")

    content_hash = sha256_text(text or "")

    chunks = chunk_text(pipe, text) if text else []
    chunks_path = chunk_dir / f"{doc.external_id}.jsonl"
    write_jsonl(chunks_path, chunks)

    doc_id = insert_doc(
        conn=conn,
        discovered=doc,
        title=title,
        content_sha256=content_hash,
        raw_path=str(raw_rel),
        text_path=str(text_path.relative_to(data_dir)),
        chunks_path=str(chunks_path.relative_to(data_dir)),
    )

    # Qdrant (опционально)
    if qdrant and qdrant_collection and embedder and chunks:
        texts = [c["text"] for c in chunks]
        vectors = embed_passages(embedder, texts, batch_size=32)
        upsert_chunks_qdrant(
            client=qdrant,
            collection=qdrant_collection,
            doc_id=doc_id,
            doc=doc,
            title=title,
            kind=doc.kind,
            chunks=chunks,
            vectors=vectors,
        )

    return "INGESTED"


def run_once(
    config: Path,
    data_dir: Path,
    db_url: str,
    only: Optional[List[str]],
    max_pages: Optional[int],
    max_items: Optional[int],
    sleep: float,
    use_qdrant: bool,
    qdrant_url: str,
    qdrant_collection: str,
    embed_model: str,
) -> None:
    sources = load_sources(config)
    sources = [x for x in sources if x.active]
    if only:
        only_set = set(only)
        sources = [x for x in sources if x.code in only_set]

    ensure_dir(data_dir)

    sess = requests_session()
    chunker = build_chunker()

    qdrant: Optional[QdrantClient] = None
    embedder: Optional[SentenceTransformer] = None

    if use_qdrant:
        embedder = load_embedder(embed_model)
        dim = embedder.get_sentence_embedding_dimension()
        qdrant = QdrantClient(url=qdrant_url)
        init_qdrant(qdrant, qdrant_collection, dim)

    with psycopg.connect(db_url) as conn:
        init_db(conn)

        discovered: List[DiscoveredDoc] = []
        for src in sources:
            handler_fn = get_handler(src.handler)
            discovered.extend(handler_fn(sess, src, max_pages=max_pages, sleep=sleep))

        uniq: Dict[Tuple[str, str], DiscoveredDoc] = {}
        for d in discovered:
            uniq[(d.source_code, d.external_id)] = d

        docs = list(uniq.values())
        if max_items is not None:
            docs = docs[:max_items]

        total = len(docs)
        for i, d in enumerate(docs, 1):
            print(f"[{i}/{total}] {d.source_code} {d.external_id} {d.url}")
            try:
                status = ingest_one(
                    sess, conn, chunker, d, data_dir, sleep,
                    qdrant=qdrant, qdrant_collection=qdrant_collection, embedder=embedder,
                )
                print(f"  {status}")
            except Exception as e:
                print(f"  ERROR: {e}")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--config", default="config/sources.yaml")
        p.add_argument("--data-dir", default="data")
        p.add_argument("--db-url", default=os.getenv("DATABASE_URL", ""))
        p.add_argument("--only", nargs="*", default=None)
        p.add_argument("--max-pages", type=int, default=None)
        p.add_argument("--max-items", type=int, default=None)
        p.add_argument("--sleep", type=float, default=0.9)

        p.add_argument("--use-qdrant", action="store_true", default=True)
        p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
        p.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "rag_chunks"))
        p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base"))

    p_run = sub.add_parser("run")
    add_common(p_run)

    p_loop = sub.add_parser("loop")
    add_common(p_loop)
    p_loop.add_argument("--interval-seconds", type=int, default=24 * 3600)

    args = ap.parse_args()
    if not args.db_url:
        raise SystemExit("DB url is empty. Provide --db-url or set DATABASE_URL.")

    if args.cmd == "run":
        run_once(
            config=Path(args.config),
            data_dir=Path(args.data_dir),
            db_url=args.db_url,
            only=args.only,
            max_pages=args.max_pages,
            max_items=args.max_items,
            sleep=args.sleep,
            use_qdrant=args.use_qdrant,
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.qdrant_collection,
            embed_model=args.embed_model,
        )
    else:
        while True:
            run_once(
                config=Path(args.config),
                data_dir=Path(args.data_dir),
                db_url=args.db_url,
                only=args.only,
                max_pages=args.max_pages,
                max_items=args.max_items,
                sleep=args.sleep,
                use_qdrant=args.use_qdrant,
                qdrant_url=args.qdrant_url,
                qdrant_collection=args.qdrant_collection,
                embed_model=args.embed_model,
            )
            print(f"Sleep {args.interval_seconds}s...")
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
