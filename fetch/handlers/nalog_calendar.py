from __future__ import annotations

import re
import time
import hashlib
import datetime as dt
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import register


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


_DATA_RE = re.compile(r"data-(\d{8})-structure-\d+\.xml", re.I)

def _fetch_html(s: requests.Session, url: str, timeout: float, sleep: float) -> bytes:
    r = s.get(url, timeout=timeout)
    time.sleep(sleep)
    r.raise_for_status()
    return r.content


def _parse_d8(d8: str) -> Optional[dt.date]:
    if not (d8 and len(d8) == 8 and d8.isdigit()):
        return None
    try:
        y = int(d8[:4])
        if 2000 <= y <= 2099:
            return dt.datetime.strptime(d8, "%Y%m%d").date()
        return dt.datetime.strptime(d8, "%d%m%Y").date()
    except Exception:
        return None


def _pick_latest_data_link(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    # 0) Супер-надёжно для этой страницы: строка таблицы с № = 8 (актуальная "гиперссылка на набор")
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        left = tds[0].get_text(" ", strip=True).replace("\xa0", " ").strip()
        if left == "8":
            a = tds[1].select_one("a[href]")
            if a and (a.get("href") or "").strip():
                return urljoin(base_url, a["href"].strip())

    # 1) Попытка по тексту (на случай, если номера строк в будущем изменятся)
    for tr in soup.select("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) < 2:
            continue
        left = tds[0].get_text(" ", strip=True).lower()
        if "гиперссылка" in left and "набор" in left:
            a = tds[1].select_one("a[href]")
            if a and (a.get("href") or "").strip():
                return urljoin(base_url, a["href"].strip())

    # 2) Fallback: max по РЕАЛЬНОЙ дате среди всех data-XXXXXXXX-structure-*.xml
    best_url: Optional[str] = None
    best_dt: Optional[dt.date] = None

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)

        m = _DATA_RE.search(abs_url)
        if not m:
            continue
        d8 = m.group(1)
        d = _parse_d8(d8)
        if not d:
            continue

        if best_dt is None or d > best_dt:
            best_dt = d
            best_url = abs_url

    return best_url


@register("nalog_calendar")
def handle(s: requests.Session, src: Source, max_pages: Optional[int], sleep: float) -> List[DiscoveredDoc]:
    html = _fetch_html(s, src.base_url, timeout=60.0, sleep=sleep)
    soup = BeautifulSoup(html, "lxml")

    data_url = _pick_latest_data_link(soup, src.base_url)
    if not data_url:
        return []

    # external_id = дата релиза, если нашли; иначе sha
    m = _DATA_RE.search(data_url)
    external_id = f"data-{m.group(1)}" if m else _sha16(data_url)

    return [
        DiscoveredDoc(
            source_code=src.code,
            url=data_url,
            external_id=external_id,
            kind=src.kind or "xml",
        )
    ]