from __future__ import annotations

import hashlib
from typing import List, Optional
from urllib.parse import parse_qs, urlparse, quote_plus

import httpx

from fetch.handlers import register
from fetch.models import DiscoveredDoc, Source


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _parse_nd(url: str) -> str:
    nd = parse_qs(urlparse(url).query).get("nd", [None])[0]
    return nd or _sha16(url)


def _canonical_url(url: str) -> str:
    nd = _parse_nd(url)
    return f"http://pravo.gov.ru/proxy/ips/?savertf=&nd={quote_plus(nd)}&page=all"


@register("pravo_ips")
async def handle(
    client: httpx.AsyncClient,
    src: Source,
    max_pages: Optional[int],
    sleep: float,
) -> List[DiscoveredDoc]:
    return [DiscoveredDoc(src.code, _canonical_url(src.base_url), _parse_nd(src.base_url), src.kind)]
