from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import httpx
from bs4 import BeautifulSoup

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import register, async_fetch


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _parse_external_id(url: str) -> str:
    m = re.search(r"/(\d+)/?$", urlparse(url).path)
    return m.group(1) if m else _sha16(url)


def _listing_url(base_url: str, page: int) -> str:
    base = base_url.rstrip("/") + "/"
    url = base if page == 1 else urljoin(base, f"{page}.html")
    u = urlparse(url)
    q = parse_qs(u.query)
    q.update({"st": ["1"], "chFederal": ["true"], "rbAllRegions": ["true"], "rbRegionSelected": ["false"]})
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q, doseq=True), u.fragment))


def _current_page(soup: BeautifulSoup) -> Optional[int]:
    active = soup.select_one("li.active, a.active, span.active")
    if active:
        m = re.search(r"\b(\d+)\b", active.get_text(" ", strip=True))
        if m:
            return int(m.group(1))
    pager = soup.select_one(".pagination, .pager")
    if pager:
        nums = re.findall(r"\b(\d+)\b", pager.get_text())
        if nums:
            return int(nums[-1])
    return None


async def _discover_listing(
    client: httpx.AsyncClient,
    base_url: str,
    item_regex: str,
    max_pages: Optional[int],
    sleep: float,
) -> List[str]:
    seen: Dict[str, None] = {}
    page = 1
    last_actual: Optional[int] = None

    while True:
        if max_pages is not None and page > max_pages:
            break
        url = _listing_url(base_url, page)
        try:
            html = await async_fetch(client, url, sleep=sleep)
        except Exception:
            break
        soup = BeautifulSoup(html, "lxml")

        actual = _current_page(soup)
        if actual is not None:
            if last_actual is not None and actual != page:
                break
            last_actual = actual

        links = [
            urljoin(url, a["href"]).split("#")[0]
            for a in soup.select("a[href]")
            if re.search(item_regex, a.get("href", ""))
        ]
        before = len(seen)
        for l in dict.fromkeys(links):
            seen[l] = None
        if len(seen) == before:
            break
        page += 1

    return list(seen)


@register("nalog_docs")
async def handle(
    client: httpx.AsyncClient,
    src: Source,
    max_pages: Optional[int],
    sleep: float,
) -> List[DiscoveredDoc]:
    urls = await _discover_listing(client, src.base_url, r"/about_fts/docs/\d+/?$", max_pages, sleep)
    return [DiscoveredDoc(src.code, u, _parse_external_id(u), src.kind) for u in urls]
