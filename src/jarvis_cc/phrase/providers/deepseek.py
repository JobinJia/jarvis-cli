"""DeepSeek-Chat provider (OpenAI-compatible chat API)."""
from __future__ import annotations

import os

import httpx

from ...config import DeepSeekConfig
from ...types import Event, Lang
from ..prompt import build_messages
from .base import PhraseProvider


class DeepSeekProvider(PhraseProvider):
    name = "deepseek"

    def __init__(self, cfg: DeepSeekConfig) -> None:
        self.cfg = cfg

    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        messages = build_messages(event, lang, max_chars)
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
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
        return bool(os.getenv(self.cfg.api_key_env))
