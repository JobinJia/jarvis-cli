"""Anthropic Claude provider via raw HTTP (avoids SDK pinning issues)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ...config import AnthropicConfig, resolve_api_key
from .base import PhraseProvider


class AnthropicProvider(PhraseProvider):
    name = "anthropic"

    def __init__(self, cfg: AnthropicConfig) -> None:
        self.cfg = cfg

    def _split_messages(
        self, messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract the system message and return (system, chat_messages)."""
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        return system_msg, chat

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        system_msg, chat = self._split_messages(messages)
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

    async def generate_stream(
        self, messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Stream from Anthropic's ``/v1/messages`` with ``stream: true``.

        Yields text deltas from ``content_block_delta`` SSE events.
        """
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        system_msg, chat = self._split_messages(messages)
        async with httpx.AsyncClient(
            base_url="https://api.anthropic.com", timeout=self.cfg.timeout_seconds
        ) as client:
            async with client.stream(
                "POST",
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
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        token = delta.get("text")
                        if token:
                            yield token
                    elif event.get("type") == "message_stop":
                        break

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg))
