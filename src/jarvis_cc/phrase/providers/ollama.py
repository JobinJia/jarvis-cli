"""Ollama local LLM provider (uses /api/chat)."""
from __future__ import annotations

import httpx

from ...config import OllamaConfig
from .base import PhraseProvider


class OllamaProvider(PhraseProvider):
    name = "ollama"

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.7, "num_predict": 200},
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["message"]["content"].strip()

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.cfg.base_url, timeout=1.0) as c:
                r = await c.get("/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
