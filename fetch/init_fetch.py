#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import functools
import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import yaml
from tqdm import tqdm
from bs4 import BeautifulSoup

import psycopg
from psycopg.rows import dict_row

from chonkie import Pipeline

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import get_handler
from fetch.format_parsers import parse_nalog_calendar_xml, calendar_days_to_rag_text

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, FilterSelector,
)

from email import policy
from email.parser import BytesParser

from dotenv import load_dotenv
load_dotenv()

UA = "nalog-consultant-bot/0.1"
DEFAULT_TIMEOUT = 60.0


# ----------------------------
# DB (async)
# ----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_doc (
  id                bigserial PRIMARY KEY,

  source_code       text NOT NULL,
  external_id       text,
  canonical_url     text NOT NULL,

  kind              text NOT NULL,
  title             text,
  published_at      date,
  doc_date          date,
  doc_number        text,

  status            text NOT NULL DEFAULT 'active',
  last_seen_at      timestamptz NOT NULL DEFAULT now(),

  next_check_at     timestamptz,
  last_fetch_at     timestamptz,

  http_etag         text,
  http_last_mod     text,

  content_sha256    text,

  error_count       int NOT NULL DEFAULT 0,
  last_error        text,

  qdrant_collection text NOT NULL DEFAULT 'rag_chunks',
  qdrant_doc_key    text,
  qdrant_revision   int NOT NULL DEFAULT 0,
  indexed_at        timestamptz,

  raw_path          text,
  text_path         text,
  chunks_path       text
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_doc_source_external
  ON rag_doc(source_code, external_id) WHERE external_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_doc_canonical_url ON rag_doc(canonical_url);
CREATE INDEX IF NOT EXISTS idx_rag_doc_next_check ON rag_doc(next_check_at);
CREATE INDEX IF NOT EXISTS idx_rag_doc_source ON rag_doc(source_code);
CREATE INDEX IF NOT EXISTS idx_rag_doc_seen ON rag_doc(last_seen_at);
"""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


async def init_db(conn: psycopg.AsyncConnection) -> None:
    async with conn.cursor() as cur:
        await cur.execute(SCHEMA_SQL)
    await conn.commit()


async def get_doc(conn: psycopg.AsyncConnection, source_code: str, external_id: str) -> Optional[dict]:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT * FROM rag_doc WHERE source_code=%s AND external_id=%s",
            (source_code, external_id),
        )
        return await cur.fetchone()


async def insert_doc(
    conn: psycopg.AsyncConnection,
    discovered: DiscoveredDoc,
    title: Optional[str],
    content_sha256: str,
    raw_path: str,
    text_path: str,
    chunks_path: str,
    crawl_freq_days: int,
    http_etag: Optional[str] = None,
    http_last_mod: Optional[str] = None,
    qdrant_collection: str = "rag_chunks",
) -> int:
    now = utcnow()
    next_check = now + dt.timedelta(days=crawl_freq_days)
    doc_key = f"{discovered.source_code}:{discovered.external_id}"
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO rag_doc (
              source_code, external_id, canonical_url, kind,
              title, status, last_seen_at, next_check_at, last_fetch_at,
              http_etag, http_last_mod, content_sha256,
              qdrant_collection, qdrant_doc_key,
              raw_path, text_path, chunks_path
            ) VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s, %s,%s, %s,%s,%s)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                discovered.source_code, discovered.external_id, discovered.url, discovered.kind,
                title, "active", now, next_check, now,
                http_etag, http_last_mod, content_sha256,
                qdrant_collection, doc_key,
                raw_path, text_path, chunks_path,
            ),
        )
        row = await cur.fetchone()
    await conn.commit()
    if row:
        return row[0]
    existing = await get_doc(conn, discovered.source_code, discovered.external_id)
    return existing["id"]


async def update_doc_content(
    conn: psycopg.AsyncConnection,
    doc_id: int,
    content_sha256: str,
    raw_path: str,
    text_path: str,
    chunks_path: str,
    crawl_freq_days: int,
    http_etag: Optional[str] = None,
    http_last_mod: Optional[str] = None,
) -> None:
    now = utcnow()
    next_check = now + dt.timedelta(days=crawl_freq_days)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            UPDATE rag_doc SET
              content_sha256=%(sha)s, raw_path=%(rp)s, text_path=%(tp)s, chunks_path=%(cp)s,
              last_fetch_at=%(now)s, next_check_at=%(nc)s,
              http_etag=%(etag)s, http_last_mod=%(lm)s,
              qdrant_revision=qdrant_revision+1, indexed_at=NULL,
              error_count=0, last_error=NULL, last_seen_at=%(now)s
            WHERE id=%(id)s
            """,
            dict(sha=content_sha256, rp=raw_path, tp=text_path, cp=chunks_path,
                 now=now, nc=next_check, etag=http_etag, lm=http_last_mod, id=doc_id),
        )
    await conn.commit()


async def touch_doc(
    conn: psycopg.AsyncConnection,
    doc_id: int,
    crawl_freq_days: int,
    http_etag: Optional[str] = None,
    http_last_mod: Optional[str] = None,
) -> None:
    now = utcnow()
    next_check = now + dt.timedelta(days=crawl_freq_days)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            UPDATE rag_doc SET
              last_fetch_at=%(now)s, next_check_at=%(nc)s,
              http_etag=COALESCE(%(etag)s, http_etag),
              http_last_mod=COALESCE(%(lm)s, http_last_mod),
              error_count=0, last_error=NULL, last_seen_at=%(now)s
            WHERE id=%(id)s
            """,
            dict(now=now, nc=next_check, etag=http_etag, lm=http_last_mod, id=doc_id),
        )
    await conn.commit()


async def mark_doc_error(
    conn: psycopg.AsyncConnection,
    doc_id: int,
    error_msg: str,
    crawl_freq_days: int,
) -> None:
    now = utcnow()
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT error_count FROM rag_doc WHERE id=%s", (doc_id,))
        row = await cur.fetchone()
    error_count = (row["error_count"] if row else 0) + 1
    next_check = now + dt.timedelta(days=min(crawl_freq_days, 2 ** error_count))
    async with conn.cursor() as cur:
        await cur.execute(
            "UPDATE rag_doc SET error_count=%s, last_error=%s, next_check_at=%s WHERE id=%s",
            (error_count, error_msg[:500], next_check, doc_id),
        )
    await conn.commit()


async def mark_doc_indexed(conn: psycopg.AsyncConnection, doc_id: int) -> None:
    async with conn.cursor() as cur:
        await cur.execute("UPDATE rag_doc SET indexed_at=%s WHERE id=%s", (utcnow(), doc_id))
    await conn.commit()


# ----------------------------
# HTTP (async, httpx)
# ----------------------------

async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    sleep: float = 0.8,
    etag: Optional[str] = None,
    last_mod: Optional[str] = None,
) -> Tuple[int, Optional[bytes], Optional[str], Optional[str]]:
    """Returns (status, content_or_None, etag, last_mod). 304 → content is None."""
    from fetch.handlers import _domain_sem
    headers: dict = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_mod:
        headers["If-Modified-Since"] = last_mod

    async with _domain_sem(url):
        resp = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 304:
            await asyncio.sleep(sleep)
            return 304, None, etag, last_mod
        resp.raise_for_status()
        content = resp.content
        new_etag = resp.headers.get("ETag")
        new_last_mod = resp.headers.get("Last-Modified")

    await asyncio.sleep(sleep)
    return resp.status_code, content, new_etag, new_last_mod


# ----------------------------
# HTML parsing (sync — runs in executor)
# ----------------------------

def extract_html_from_mime(blob: bytes) -> Tuple[str, str]:
    msg = BytesParser(policy=policy.default).parsebytes(blob)
    if not msg.is_multipart():
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace"), charset
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = (part.get_content_type() or "").lower()
        if "html" in ctype:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace"), charset
    raise RuntimeError("No text/html part found inside MIME/MHTML container")


def extract_title(html: bytes | str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    el = soup.find("h1") or soup.title
    if el:
        t = el.get_text(" ", strip=True)
        return t[:800] if t else None
    return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = (soup.body or soup).get_text("\n")
    text = text.replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_NBSP = re.compile(r"\xa0|&nbsp;")
_WS = re.compile(r"[ \t]+\n|\n[ \t]+")
_MNL = re.compile(r"\n{3,}")
_MSP = re.compile(r"[ \t]{2,}")


def extract_main_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    main = (soup.select_one("#divSecondPageColumns")
            or soup.select_one(".page-content__center")
            or soup.body)
    if not main:
        return ""
    for t in main.select(
        "script,style,noscript,"
        "#ctl00_ctl03_ctl02_pnlMain,#dUserForm,.mfp-hide,.popup,"
        "a.js-popup,#mkgu-widget,.DoYouFoundWrapper,#CtrlEnterPopup,"
        ".div_move_to_right,.link-block"
    ):
        t.decompose()
    text = main.get_text("\n", strip=True)
    text = _NBSP.sub(" ", text)
    text = _WS.sub("\n", text)
    text = _MSP.sub(" ", text)
    return _MNL.sub("\n\n", text).strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ----------------------------
# Chunking / Embeddings (sync)
# ----------------------------

def build_chunker() -> Pipeline:
    return (
        Pipeline()
        .chunk_with("recursive", tokenizer="word", chunk_size=1100,
                    recipe="markdown", min_characters_per_chunk=1100)
        .refine_with("overlap", context_size=160)
    )


def chunk_text(pipe: Pipeline, text: str) -> List[dict]:
    return [{"i": i, "text": ch.text} for i, ch in enumerate(pipe.run(texts=text).chunks)]


def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cuda", token=os.getenv("HF_TOKEN"))


def embed_passages(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    return model.encode(texts, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).tolist()


# ----------------------------
# Qdrant (async)
# ----------------------------

async def init_qdrant(client: AsyncQdrantClient, collection: str, vector_size: int) -> None:
    resp = await client.get_collections()
    if collection not in {c.name for c in resp.collections}:
        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def stable_point_id(source_code: str, external_id: str, chunk_i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_code}:{external_id}:{chunk_i}"))


async def delete_doc_chunks(client: AsyncQdrantClient, collection: str,
                            source_code: str, external_id: str) -> None:
    await client.delete(
        collection_name=collection,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="source_code", match=MatchValue(value=source_code)),
            FieldCondition(key="external_id", match=MatchValue(value=external_id)),
        ])),
    )


async def upsert_chunks(
    client: AsyncQdrantClient, collection: str,
    doc_id: int, doc: DiscoveredDoc, title: Optional[str],
    chunks: List[dict], vectors: List[List[float]],
) -> None:
    points = [
        PointStruct(
            id=stable_point_id(doc.source_code, doc.external_id, ch["i"]),
            vector=vec,
            payload={"doc_id": doc_id, "source_code": doc.source_code,
                     "external_id": doc.external_id, "url": doc.url,
                     "title": title, "kind": doc.kind,
                     "chunk_i": ch["i"], "text": ch["text"]},
        )
        for ch, vec in zip(chunks, vectors)
    ]
    if points:
        await client.upsert(collection_name=collection, points=points)


# ----------------------------
# Config / Filesystem
# ----------------------------

def load_sources(path: Path) -> List[Source]:
    cfg = yaml.safe_load(path.read_text("utf-8"))
    return [
        Source(
            code=x["code"], base_url=x["base_url"],
            kind=x.get("kind", "unknown"), active=bool(x.get("active", False)),
            handler=x.get("handler") or "",
            crawl_freq_days=int(x.get("crawl_freq_days", 7)),
        )
        for x in cfg.get("sources", [])
    ]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Blob processing (sync — runs in executor)
# ----------------------------

def _process_blob(
    blob: bytes,
    doc: DiscoveredDoc,
    data_dir: Path,
    pipe: Pipeline,
) -> Tuple[Optional[str], str, str, str, str, List[dict]]:
    path_id = doc.external_id[:80]  # filesystem limit: 255 bytes per component
    raw_dir = data_dir / "raw" / doc.source_code / path_id
    txt_dir = data_dir / "text" / doc.source_code / path_id
    chunk_dir = data_dir / "chunks" / doc.source_code
    for d in (raw_dir, txt_dir, chunk_dir):
        ensure_dir(d)

    head = blob[:2048]
    is_xml = doc.url.lower().endswith(".xml")
    is_mime = bool(
        re.search(br"^\s*MIME-Version\s*:", head, re.I)
        or re.search(br"Content-Type\s*:\s*multipart/", head, re.I)
    )

    title: Optional[str] = None
    if is_xml:
        raw_file = raw_dir / "file.xml"
        raw_file.write_bytes(blob)
        cal_title, days = parse_nalog_calendar_xml(blob)
        title = cal_title or f"Налоговый календарь ({doc.external_id})"
        text = calendar_days_to_rag_text(cal_title, days)
        raw_rel = raw_file
    elif is_mime:
        raw_file = raw_dir / "file.mhtml"
        raw_file.write_bytes(blob)
        html_str, _ = extract_html_from_mime(blob)
        (raw_dir / "page.html").write_text(html_str, "utf-8")
        title = extract_title(html_str)
        text = html_to_text(html_str)
        raw_rel = raw_file
    else:
        raw_file = raw_dir / "page.html"
        raw_file.write_bytes(blob)
        title = extract_title(blob)
        text = extract_main_text(blob)
        raw_rel = raw_file

    text_path = txt_dir / "doc.txt"
    text_path.write_text(text, "utf-8")
    content_hash = sha256_text(text or "")
    chunks = chunk_text(pipe, text) if text else []
    chunks_path = chunk_dir / f"{path_id}.jsonl"
    write_jsonl(chunks_path, chunks)

    return (
        title, content_hash,
        str(raw_rel.relative_to(data_dir)),
        str(text_path.relative_to(data_dir)),
        str(chunks_path.relative_to(data_dir)),
        chunks,
    )


# ----------------------------
# Ingest one document (async)
# ----------------------------

async def ingest_one(
    client: httpx.AsyncClient,
    conn: psycopg.AsyncConnection,
    pipe: Pipeline,
    doc: DiscoveredDoc,
    data_dir: Path,
    sleep: float,
    qdrant: Optional[AsyncQdrantClient],
    qdrant_collection: Optional[str],
    embedder: Optional[SentenceTransformer],
    crawl_freq_days: int,
    executor: ThreadPoolExecutor,
) -> str:
    loop = asyncio.get_event_loop()
    existing = await get_doc(conn, doc.source_code, doc.external_id)

    status_code, blob, new_etag, new_last_mod = await fetch_url(
        client, doc.url, sleep=sleep,
        etag=existing["http_etag"] if existing else None,
        last_mod=existing["http_last_mod"] if existing else None,
    )

    if status_code == 304:
        if existing:
            await touch_doc(conn, existing["id"], crawl_freq_days, new_etag, new_last_mod)
        return "SKIP_NOT_MODIFIED"

    assert blob is not None
    title, content_hash, raw_path, text_path, chunks_path, chunks = await loop.run_in_executor(
        executor, functools.partial(_process_blob, blob, doc, data_dir, pipe)
    )

    if existing:
        if content_hash == existing["content_sha256"]:
            await touch_doc(conn, existing["id"], crawl_freq_days, new_etag, new_last_mod)
            return "SKIP_UNCHANGED"
        await update_doc_content(
            conn, existing["id"], content_hash, raw_path, text_path, chunks_path,
            crawl_freq_days, new_etag, new_last_mod,
        )
        doc_id = existing["id"]
        if qdrant and qdrant_collection and embedder and chunks:
            await delete_doc_chunks(qdrant, qdrant_collection, doc.source_code, doc.external_id)
            vectors = await loop.run_in_executor(executor, functools.partial(embed_passages, embedder, [c["text"] for c in chunks]))
            await upsert_chunks(qdrant, qdrant_collection, doc_id, doc, title, chunks, vectors)
            await mark_doc_indexed(conn, doc_id)
        return "UPDATED"

    doc_id = await insert_doc(
        conn, doc, title, content_hash, raw_path, text_path, chunks_path,
        crawl_freq_days, new_etag, new_last_mod,
        qdrant_collection=qdrant_collection or "rag_chunks",
    )
    if qdrant and qdrant_collection and embedder and chunks:
        vectors = await loop.run_in_executor(executor, functools.partial(embed_passages, embedder, [c["text"] for c in chunks]))
        await upsert_chunks(qdrant, qdrant_collection, doc_id, doc, title, chunks, vectors)
        await mark_doc_indexed(conn, doc_id)
    return "INGESTED"


# ----------------------------
# Run modes
# ----------------------------

def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": UA, "Accept-Language": "ru,en;q=0.8"},
        follow_redirects=True,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
    )


async def run_once(
    config: Path, data_dir: Path, db_url: str,
    only: Optional[List[str]], max_pages: Optional[int], max_items: Optional[int],
    sleep: float, use_qdrant: bool, qdrant_url: str, qdrant_collection: str,
    embed_model: str, max_concurrent: int,
) -> None:
    sources = [s for s in load_sources(config) if s.active]
    if only:
        sources = [s for s in sources if s.code in set(only)]
    sources_by_code = {s.code: s for s in sources}

    ensure_dir(data_dir)
    print("Building chunker...")
    chunker = build_chunker()
    embedder: Optional[SentenceTransformer] = None
    qdrant: Optional[AsyncQdrantClient] = None

    if use_qdrant:
        print(f"Loading embedding model {embed_model!r} (first run downloads ~500MB)...")
        embedder = load_embedder(embed_model)
        print(f"  Model loaded, dim={embedder.get_sentence_embedding_dimension()}")
        qdrant = AsyncQdrantClient(url=qdrant_url)
        print(f"Connecting to Qdrant at {qdrant_url}...")
        await init_qdrant(qdrant, qdrant_collection, embedder.get_sentence_embedding_dimension())
        print(f"  Qdrant ready, collection={qdrant_collection!r}")

    print(f"Connecting to PostgreSQL...")
    async with _make_client() as client:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            await init_db(conn)
            print("  DB ready")

            # --- Discovery ---
            disc_bar = tqdm(total=len(sources), desc="Discovering", unit="src")
            results = await asyncio.gather(
                *[get_handler(s.handler)(client, s, max_pages=max_pages, sleep=sleep) for s in sources],
                return_exceptions=True,
            )
            disc_bar.close()

            discovered: List[DiscoveredDoc] = []
            for src, res in zip(sources, results):
                if isinstance(res, Exception):
                    tqdm.write(f"[{src.code}] discovery ERROR: {res}")
                else:
                    tqdm.write(f"[{src.code}] {len(res)} docs found")
                    discovered.extend(res)

            uniq = {(d.source_code, d.external_id): d for d in discovered}
            docs = list(uniq.values())
            if max_items is not None:
                docs = docs[:max_items]
            total = len(docs)

            # --- Ingestion ---
            counters: Dict[str, int] = {}
            sem = asyncio.Semaphore(max_concurrent)
            bar = tqdm(total=total, desc="Ingesting", unit="doc")

            with ThreadPoolExecutor(max_workers=1) as executor:
                async def process(doc: DiscoveredDoc) -> None:
                    src = sources_by_code.get(doc.source_code)
                    async with sem:
                        try:
                            status = await ingest_one(
                                client, conn, chunker, doc, data_dir, sleep,
                                qdrant, qdrant_collection, embedder,
                                src.crawl_freq_days if src else 7, executor,
                            )
                            counters[status] = counters.get(status, 0) + 1
                            bar.set_postfix(counters, refresh=False)
                        except Exception as e:
                            counters["ERROR"] = counters.get("ERROR", 0) + 1
                            tqdm.write(f"ERROR {doc.source_code}:{doc.external_id} — {e}")
                        finally:
                            bar.update(1)

                await asyncio.gather(*[process(d) for d in docs])

            bar.close()
            tqdm.write(f"Done: {counters}")


async def run_updates(
    config: Path, data_dir: Path, db_url: str,
    sleep: float, use_qdrant: bool, qdrant_url: str, qdrant_collection: str,
    embed_model: str, batch: int, max_concurrent: int,
) -> None:
    sources_by_code = {s.code: s for s in load_sources(config)}
    ensure_dir(data_dir)
    chunker = build_chunker()
    embedder: Optional[SentenceTransformer] = None
    qdrant: Optional[AsyncQdrantClient] = None

    if use_qdrant:
        embedder = load_embedder(embed_model)
        qdrant = AsyncQdrantClient(url=qdrant_url)

    async with _make_client() as client:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, source_code, external_id, canonical_url, kind,
                           http_etag, http_last_mod, content_sha256, error_count
                    FROM rag_doc
                    WHERE status='active' AND (next_check_at IS NULL OR next_check_at <= now())
                    ORDER BY next_check_at NULLS FIRST
                    LIMIT %s
                    """,
                    (batch,),
                )
                due = await cur.fetchall()

            total = len(due)
            counters: Dict[str, int] = {}
            sem = asyncio.Semaphore(max_concurrent)
            bar = tqdm(total=total, desc="Updating", unit="doc")

            with ThreadPoolExecutor(max_workers=1) as executor:
                async def process(row: dict) -> None:
                    src = sources_by_code.get(row["source_code"])
                    doc = DiscoveredDoc(
                        row["source_code"], row["canonical_url"],
                        row["external_id"] or row["canonical_url"], row["kind"],
                    )
                    async with sem:
                        try:
                            status = await ingest_one(
                                client, conn, chunker, doc, data_dir, sleep,
                                qdrant, qdrant_collection, embedder,
                                src.crawl_freq_days if src else 7, executor,
                            )
                            counters[status] = counters.get(status, 0) + 1
                            bar.set_postfix(counters, refresh=False)
                        except Exception as e:
                            counters["ERROR"] = counters.get("ERROR", 0) + 1
                            tqdm.write(f"ERROR {doc.source_code}:{doc.external_id} — {e}")
                            await mark_doc_error(conn, row["id"], str(e), src.crawl_freq_days if src else 7)
                        finally:
                            bar.update(1)

                await asyncio.gather(*[process(row) for row in due])

            bar.close()
            tqdm.write(f"Done: {counters}")


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
        p.add_argument("--sleep", type=float, default=0.9)
        p.add_argument("--use-qdrant", action="store_true", default=True)
        p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
        p.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "rag_chunks"))
        p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
        p.add_argument("--max-concurrent", type=int, default=5)

    p_run = sub.add_parser("run")
    add_common(p_run)
    p_run.add_argument("--only", nargs="*", default=None)
    p_run.add_argument("--max-pages", type=int, default=None)
    p_run.add_argument("--max-items", type=int, default=None)

    p_upd = sub.add_parser("update")
    add_common(p_upd)
    p_upd.add_argument("--batch", type=int, default=100)

    p_loop = sub.add_parser("loop")
    add_common(p_loop)
    p_loop.add_argument("--only", nargs="*", default=None)
    p_loop.add_argument("--max-pages", type=int, default=None)
    p_loop.add_argument("--max-items", type=int, default=None)
    p_loop.add_argument("--interval-seconds", type=int, default=24 * 3600)

    args = ap.parse_args()
    if not args.db_url:
        raise SystemExit("DB url is empty. Provide --db-url or set DATABASE_URL.")

    kw = dict(
        config=Path(args.config), data_dir=Path(args.data_dir), db_url=args.db_url,
        sleep=args.sleep, use_qdrant=args.use_qdrant, qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection, embed_model=args.embed_model,
        max_concurrent=args.max_concurrent,
    )

    if args.cmd == "run":
        asyncio.run(run_once(**kw, only=args.only, max_pages=args.max_pages, max_items=args.max_items))
    elif args.cmd == "update":
        asyncio.run(run_updates(**kw, batch=args.batch))
    else:
        async def _loop():
            while True:
                await run_once(**kw, only=args.only, max_pages=args.max_pages, max_items=args.max_items)
                print(f"Sleep {args.interval_seconds}s...")
                await asyncio.sleep(args.interval_seconds)
        asyncio.run(_loop())


if __name__ == "__main__":
    main()
