"""Thin OpenAI-compatible chat client for the agent LLM (gpt-oss-20b via LM Studio).

Why not reuse src.rag.core.LlamaCppChatClient: it always injects Qwen-oriented
`enable_thinking: False` / `chat_template_kwargs`, which is wrong for gpt-oss
(uses `reasoning_effort`). This client is endpoint-agnostic and, crucially,
**fully bypasses the http(s)_proxy** — the WSL env has an active proxy at
172.18.96.1:10809 that otherwise hijacks requests to the Windows-hosted server
(this manifested as spurious HTTP 503s that never reached LM Studio).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def configure_no_proxy_for_url(url: str) -> None:
    """Add the endpoint host to NO_PROXY (belt; the suspenders is trust_env=False)."""
    host = url.removeprefix("http://").removeprefix("https://").split("/", 1)[0].split(":", 1)[0]
    hosts = ["localhost", "127.0.0.1"] if host in {"", "localhost", "127.0.0.1"} else [host]
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [p.strip() for p in current.split(",") if p.strip()]
    for item in hosts:
        if item not in parts:
            parts.append(item)
    os.environ["NO_PROXY"] = ",".join(parts)
    os.environ["no_proxy"] = ",".join(parts)


# Default endpoint: LM Studio on the Windows host, reached from WSL by its LAN IP.
# Fallback alias is the WSL gateway http://172.18.96.1:1234 (pass --base-url).
DEFAULT_BASE_URL = "http://192.168.1.8:1234"
DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass
class GPTOSSChatClient:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 2000
    timeout_s: float = 180.0
    reasoning_effort: Optional[str] = "low"  # low / medium / high / off

    def __post_init__(self) -> None:
        configure_no_proxy_for_url(self.base_url)
        self.url = self.base_url.rstrip("/") + "/v1/chat/completions"
        # trust_env=False makes requests ignore HTTP(S)_PROXY entirely.
        self._session = requests.Session()
        self._session.trust_env = False

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return {"content": str, "tool_calls": list|None, "raw": dict}."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.reasoning_effort and str(self.reasoning_effort).lower() != "off":
            payload["reasoning_effort"] = str(self.reasoning_effort).lower()
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            r = self._session.post(
                self.url,
                json=payload,
                timeout=self.timeout_s,
                proxies={"http": None, "https": None},  # hard proxy bypass
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Не удалось подключиться к LLM по {self.url}: {exc}. "
                f"Проверьте, что LM Studio запущен и доступен (IP/порт), и что прокси обойдён."
            ) from exc

        if r.status_code == 503:
            raise RuntimeError(
                f"LLM HTTP 503 ({self.url}): модель не готова к инференсу. "
                f"Загрузите '{self.model}' в LM Studio (Load) или включите JIT-loading."
            )
        if r.status_code != 200:
            raise RuntimeError(f"LLM HTTP {r.status_code} ({self.url}): {r.text[:500]}")

        data = r.json()
        msg = data["choices"][0]["message"]
        # gpt-oss returns reasoning in a separate `reasoning` field; content is clean.
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls"),
            "raw": data,
        }

    def content_only(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        return self.chat(messages, **kw)["content"]


if __name__ == "__main__":  # smoke test
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    client = GPTOSSChatClient(base_url=base)
    out = client.content_only([{"role": "user", "content": "Ответь одним словом: ОК"}], max_tokens=2000)
    print("OK, model replied:", repr(out))
