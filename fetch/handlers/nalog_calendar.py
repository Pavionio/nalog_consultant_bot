from __future__ import annotations

import datetime as dt
import hashlib
import re
from typing import List, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import register, async_fetch


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


_DATA_RE = re.compile(r"data-(\d{8})-structure-\d+\.xml", re.I)


def _parse_d8(d8: str) -> Optional[dt.date]:
    if not (d8 and len(d8) == 8 and d8.isdigit()):
        return None
    try:
        y = int(d8[:4])
        fmt = "%Y%m%d" if 2000 <= y <= 2099 else "%d%m%Y"
        return dt.datetime.strptime(d8, fmt).date()
    except Exception:
        return None


def _pick_latest(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    # row №8 = canonical "dataset hyperlink"
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2 and tds[0].get_text(" ", strip=True).replace("\xa0", " ").strip() == "8":
            a = tds[1].select_one("a[href]")
            if a and a.get("href", "").strip():
                return urljoin(base_url, a["href"].strip())

    # fallback: latest data-YYYYMMDD-structure-*.xml
    best_url, best_dt = None, None
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        m = _DATA_RE.search(abs_url)
        if not m:
            continue
        d = _parse_d8(m.group(1))
        if d and (best_dt is None or d > best_dt):
            best_dt, best_url = d, abs_url
    return best_url


@register("nalog_calendar")
async def handle(
    client: httpx.AsyncClient,
    src: Source,
    max_pages: Optional[int],
    sleep: float,
) -> List[DiscoveredDoc]:
    html = await async_fetch(client, src.base_url, sleep=sleep)
    soup = BeautifulSoup(html, "lxml")
    data_url = _pick_latest(soup, src.base_url)
    if not data_url:
        return []
    m = _DATA_RE.search(data_url)
    external_id = f"data-{m.group(1)}" if m else _sha16(data_url)
    return [DiscoveredDoc(src.code, data_url, external_id, src.kind or "xml")]
