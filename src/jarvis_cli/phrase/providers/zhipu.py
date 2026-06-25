"""Zhipu AI (智谱) GLM provider — OpenAI-compatible chat API.

A free cloud fallback for phrasing (``glm-4-flash`` is free with real-name
verification). Zhipu's compatible endpoint ends at ``/paas/v4`` with NO ``/v1``
segment, so the path is ``/chat/completions`` (the shared helper builds the
full URL, avoiding the /v1 404 trap).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from ...config import ZhipuConfig, resolve_api_key
from ._openai_compat import chat_completion, chat_completion_stream
from .base import PhraseProvider


class ZhipuProvider(PhraseProvider):
    name = "zhipu"

    def __init__(self, cfg: ZhipuConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        return await chat_completion(
            base_url=self.cfg.base_url,
            path="/chat/completions",
            api_key=key,
            model=self.cfg.model,
            messages=messages,
            timeout_seconds=self.cfg.timeout_seconds,
        )

    async def generate_stream(
        self, messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        async for token in chat_completion_stream(
            base_url=self.cfg.base_url,
            path="/chat/completions",
            api_key=key,
            model=self.cfg.model,
            messages=messages,
            timeout_seconds=self.cfg.timeout_seconds,
        ):
            yield token

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg))
