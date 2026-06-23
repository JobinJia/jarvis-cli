"""Zhipu AI (智谱) GLM provider — OpenAI-compatible chat API.

Used as a free cloud fallback for phrasing (``GLM-4.7-Flash`` is free with
real-name verification). The endpoint is ``{base_url}/chat/completions`` with
NO ``/v1`` segment — Zhipu's compatible path already ends at ``/paas/v4``, so
reusing an OpenAI client that hard-codes ``/v1`` 404s. We build the full URL
explicitly to avoid that and any base-url join ambiguity.
"""
from __future__ import annotations

import httpx

from ...config import ZhipuConfig, resolve_api_key
from .base import PhraseProvider


class ZhipuProvider(PhraseProvider):
    name = "zhipu"

    def __init__(self, cfg: ZhipuConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=self.cfg.timeout_seconds) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 80,
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg))
