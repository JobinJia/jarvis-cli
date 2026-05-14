"""Ollama local LLM provider (uses /api/chat)."""
from __future__ import annotations

import httpx

from ...config import OllamaConfig
from ...types import Event, Lang
from ..prompt import build_messages
from .base import PhraseProvider


class OllamaProvider(PhraseProvider):
    name = "ollama"

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str:
        messages = build_messages(event, lang, max_chars)
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "stream": False,
                    # think=False disables Qwen3/DeepSeek-R1 style chain-of-thought
                    # output so num_predict isn't consumed by <think>...</think>.
                    # Ignored by models that don't emit thinking tokens.
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
