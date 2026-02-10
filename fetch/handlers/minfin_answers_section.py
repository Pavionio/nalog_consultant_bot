from __future__ import annotations

import re
import time
import hashlib
from typing import List, Optional, Dict, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import register


HDRS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ru,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://minfin.gov.ru/",
}


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _fetch_html(s: requests.Session, url: str, timeout: float, sleep: float) -> bytes:
    r = s.get(url, timeout=timeout, headers=HDRS, allow_redirects=True)
    time.sleep(sleep)
    r.raise_for_status()
    return r.content


def _detect_key(html: bytes) -> str:
    """
    На первой странице раздела ищем секционный ключ N:
      - ссылки содержат id_N=...
      - пагинация работает через page_N=...
    Обычно N=57, но лучше детектить.
    """
    s = html.decode("utf-8", errors="ignore")

    # сначала пытаемся page_N (самый прямой сигнал пагинации)
    m = re.search(r"\bpage_(\d+)=\d+", s)
    if m:
        return m.group(1)

    # иначе — id_N
    m = re.search(r"\bid_(\d+)=\d+", s)
    if m:
        return m.group(1)

    raise RuntimeError("Minfin Answers section: не удалось определить key (ни page_N, ни id_N).")


def _listing_url(base: str, key: str, page: int) -> str:
    base = base.rstrip("/") + "/"
    if page <= 1:
        return base
    return f"{base}?page_{key}={page}"


def _extract_doc_links(page_url: str, html: bytes, key: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    out: List[str] = []
    needle = f"id_{key}="

    # Важно: фильтруем по /Answers/ чтобы не собрать меню/другие разделы сайта
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        u = urljoin(page_url, href).split("#")[0]
        if "/perfomance/tax_relations/Answers/" in u and needle in u:
            out.append(u)

    # uniq preserve order
    seen: Dict[str, None] = {}
    for u in out:
        seen.setdefault(u, None)
    return list(seen.keys())


def _external_id(url: str, key: str) -> str:
    q = parse_qs(urlparse(url).query)
    k = f"id_{key}"
    if k in q and q[k]:
        return q[k][0]
    return _sha16(url)


@register("minfin_answers_section")
def handle(s: requests.Session, src: Source, max_pages: Optional[int], sleep: float) -> List[DiscoveredDoc]:
    # лёгкий прогрев
    try:
        s.get("https://minfin.gov.ru/", timeout=20, headers=HDRS)
        time.sleep(min(sleep, 0.4))
    except Exception:
        pass

    html1 = _fetch_html(s, src.base_url, timeout=60.0, sleep=sleep)
    key = _detect_key(html1)

    seen: Dict[str, None] = {}
    page = 1

    while True:
        if max_pages is not None and page > max_pages:
            break

        url = _listing_url(src.base_url, key, page)
        html = html1 if page == 1 else _fetch_html(s, url, timeout=60.0, sleep=sleep)

        links = _extract_doc_links(url, html, key)

        new = 0
        for u in links:
            if u not in seen:
                seen[u] = None
                new += 1

        if new == 0:
            break

        page += 1

    return [
        DiscoveredDoc(
            source_code=src.code,
            url=u,
            external_id=_external_id(u, key),
            kind=src.kind or "minfin_answers",
        )
        for u in seen.keys()
    ]
