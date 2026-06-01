from __future__ import annotations

import asyncio
from typing import Optional
from urllib.parse import urlparse

import httpx

from fetch.models import Source, DiscoveredDoc


REGISTRY = {}

# Per-domain semaphores — lazily created, limit 1 concurrent request per host
_domain_semaphores: dict[str, asyncio.Semaphore] = {}


def _domain_sem(url: str) -> asyncio.Semaphore:
    domain = urlparse(url).netloc
    if domain not in _domain_semaphores:
        _domain_semaphores[domain] = asyncio.Semaphore(1)
    return _domain_semaphores[domain]


def register(name):
    def decorator(func):
        REGISTRY[name] = func
        return func
    return decorator


def get_handler(name):
    if name not in REGISTRY:
        raise KeyError("unknown handler:", name)
    return REGISTRY[name]


async def async_fetch(
    client: httpx.AsyncClient,
    url: str,
    sleep: float = 0.8,
    timeout: float = 60.0,
    headers: Optional[dict] = None,
) -> bytes:
    async with _domain_sem(url):
        resp = await client.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        content = resp.content
    await asyncio.sleep(sleep)
    return content


from . import nalog_about_nalog
from . import nalog_docs
from . import nalog_calendar
from . import pravo_ips
from . import minfin_answers_section
