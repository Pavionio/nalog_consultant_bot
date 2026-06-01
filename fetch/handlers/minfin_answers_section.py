from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import httpx
from bs4 import BeautifulSoup

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import register, async_fetch


_HDRS = {
    "Accept-Language": "ru,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://minfin.gov.ru/",
}


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


_MINFIN_DEFAULT_KEY = "57"


def _detect_key(html: bytes) -> str:
    """Return pagination key N from page_N= or id_N= patterns, fallback to known default."""
    text = html.decode("utf-8", errors="ignore")
    m = re.search(r"\bpage_(\d+)=\d+", text) or re.search(r"\bid_(\d+)=\d+", text)
    return m.group(1) if m else _MINFIN_DEFAULT_KEY


def _listing_url(base: str, key: str, page: int) -> str:
    base = base.rstrip("/") + "/"
    return base if page <= 1 else f"{base}?page_{key}={page}"


def _extract_links(page_url: str, html: bytes, key: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    needle = f"id_{key}="
    seen: Dict[str, None] = {}
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        u = urljoin(page_url, href).split("#")[0]
        if "/perfomance/tax_relations/Answers/" in u and needle in u:
            seen.setdefault(u, None)
    return list(seen)


def _external_id(url: str, key: str) -> str:
    q = parse_qs(urlparse(url).query)
    vals = q.get(f"id_{key}")
    return vals[0] if vals else _sha16(url)


@register("minfin_answers_section")
async def handle(
    client: httpx.AsyncClient,
    src: Source,
    max_pages: Optional[int],
    sleep: float,
) -> List[DiscoveredDoc]:
    # warmup
    try:
        await async_fetch(client, "https://minfin.gov.ru/", sleep=min(sleep, 0.4), headers=_HDRS)
    except Exception:
        pass

    html1 = await async_fetch(client, src.base_url, sleep=sleep, headers=_HDRS)
    key = _detect_key(html1)

    seen: Dict[str, None] = {}
    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break
        url = _listing_url(src.base_url, key, page)
        html = html1 if page == 1 else await async_fetch(client, url, sleep=sleep, headers=_HDRS)
        before = len(seen)
        for u in _extract_links(url, html, key):
            seen.setdefault(u, None)
        if len(seen) == before:
            break
        page += 1

    return [DiscoveredDoc(src.code, u, _external_id(u, key), src.kind or "letter") for u in seen]
