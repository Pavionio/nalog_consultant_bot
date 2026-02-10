from __future__ import annotations

import hashlib
from typing import List, Optional
from urllib.parse import parse_qs, urlparse, quote_plus

import requests

from fetch.handlers import register
from fetch.models import DiscoveredDoc, Source


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _parse_nd(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    nd = q.get("nd", [None])[0]
    return nd or _sha16(url)


def _force_pravo_proxy_rtf_all(url: str) -> str:
    """
    Всегда возвращает строго:
    http://pravo.gov.ru/proxy/ips/?savertf=&nd=...&page=all

    Все остальные параметры и даже исходный хост/путь игнорируются.
    """
    nd = _parse_nd(url)
    # Важно: savertf должен присутствовать и быть пустым: savertf=
    return (
        "http://pravo.gov.ru/proxy/ips/"
        f"?savertf=&nd={quote_plus(nd)}&page=all"
    )


@register("pravo_ips")
def handle(
    s: requests.Session,
    src: Source,
    max_pages: Optional[int],
    sleep: float,
) -> List[DiscoveredDoc]:
    url = _force_pravo_proxy_rtf_all(src.base_url)
    return [
        DiscoveredDoc(
            source_code=src.code,
            url=url,
            external_id=_parse_nd(src.base_url),
            kind=src.kind,
        )
    ]
