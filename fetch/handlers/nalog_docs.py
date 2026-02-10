from __future__ import annotations

import re
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import hashlib

import requests
from bs4 import BeautifulSoup

from fetch.models import Source, DiscoveredDoc
from fetch.handlers import register


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _parse_nalog_external_id(url: str) -> str:
    m = re.search(r"/(\d+)/?$", urlparse(url).path)
    return m.group(1) if m else _sha16(url)


def _with_query_params(url: str, params: Dict[str, str]) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    for k, v in params.items():
        q[k] = [v]
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


def _listing_page_url(base_url: str, page: int) -> str:
    """
    Nalog listing pagination:
      page 1: .../docs/   (or .../docs)
      page k: .../docs/k.html
    Always add filters.
    """
    base = base_url.rstrip("/") + "/"
    if page == 1:
        url = base
    else:
        url = urljoin(base, f"{page}.html")

    return _with_query_params(
        url,
        {
            "st": "1",
            "chFederal": "true",
            "rbAllRegions": "true",
            "rbRegionSelected": "false",
        },
    )



def _fetch_html(s: requests.Session, url: str, timeout: float, sleep: float) -> bytes:
    r = s.get(url, timeout=timeout)
    time.sleep(sleep)
    r.raise_for_status()
    return r.content


def _soup(html: bytes) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _extract_current_page(soup: BeautifulSoup) -> Optional[int]:
    """
    Try to read the current page number from pagination UI.
    Works best when there is an "active" page item.
    """
    # 1) common pattern: li.active / a.active / span.active
    active = soup.select_one("li.active, a.active, span.active")
    if active:
        txt = active.get_text(" ", strip=True)
        m = re.search(r"\b(\d+)\b", txt)
        if m:
            return int(m.group(1))

    # 2) fallback: sometimes current page is the last numeric in pagination block
    # try to locate a pagination container first
    pager = soup.select_one(".pagination, nav[aria-label*=page], nav[aria-label*=страниц], .pager")
    if pager:
        nums = re.findall(r"\b(\d+)\b", pager.get_text(" ", strip=True))
        if nums:
            # often contains many numbers; current might be last shown without link,
            # but we can't know reliably; still better than nothing.
            return int(nums[-1])

    return None


def _discover_listing(
    s: requests.Session,
    base_url: str,
    item_path_regex: str,
    max_pages: Optional[int],
    sleep: float,
) -> List[str]:
    seen: Dict[str, None] = {}
    page = 1
    last_actual_page: Optional[int] = None

    while True:
        if max_pages is not None and page > max_pages:
            break

        url = _listing_page_url(base_url, page)

        try:
            html = _fetch_html(s, url, timeout=60.0, sleep=sleep)
        except Exception:
            break

        soup = _soup(html)

        # Stop condition for "soft 404": site returns last page for out-of-range page
        actual_page = _extract_current_page(soup)
        if actual_page is not None:
            if last_actual_page is None:
                last_actual_page = actual_page
            # if requested page differs from actual -> we got clamped to last page
            if actual_page != page:
                break
            last_actual_page = actual_page

        # collect links
        links: List[str] = []
        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if href and re.search(item_path_regex, href):
                links.append(urljoin(url, href))

        links = list(dict.fromkeys([l.split("#")[0] for l in links]))
        new = 0
        for l in links:
            if l not in seen:
                seen[l] = None
                new += 1

        # if no new links -> done
        if new == 0:
            break

        page += 1

    return list(seen.keys())


@register("nalog_docs")
def handle(s: requests.Session, src: Source, max_pages: Optional[int], sleep: float) -> List[DiscoveredDoc]:
    urls = _discover_listing(
        s=s,
        base_url=src.base_url,
        item_path_regex=r"/about_fts/docs/\d+/?$",
        max_pages=max_pages,
        sleep=sleep,
    )
    return [
        DiscoveredDoc(
            source_code=src.code,
            url=u,
            external_id=_parse_nalog_external_id(u),
            kind=src.kind,
        )
        for u in urls
    ]
