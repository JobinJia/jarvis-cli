"""Anthropic Claude provider via raw HTTP (avoids SDK pinning issues)."""
from __future__ import annotations

import httpx

from ...config import AnthropicConfig, resolve_api_key
from .base import PhraseProvider


class AnthropicProvider(PhraseProvider):
    name = "anthropic"

    def __init__(self, cfg: AnthropicConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        async with httpx.AsyncClient(
            base_url="https://api.anthropic.com", timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "system": system_msg,
                    "messages": chat,
                    "max_tokens": 120,
                    "temperature": 0.7,
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["content"][0]["text"].strip()

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg))
