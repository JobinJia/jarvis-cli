"""OpenAI chat-completions provider."""
from __future__ import annotations

from collections.abc import AsyncIterator

from ...config import OpenAIConfig, resolve_api_key
from ._openai_compat import chat_completion, chat_completion_stream
from .base import PhraseProvider


class OpenAIProvider(PhraseProvider):
    name = "openai"

    def __init__(self, cfg: OpenAIConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        return await chat_completion(
            base_url="https://api.openai.com",
            path="/v1/chat/completions",
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
            base_url="https://api.openai.com",
            path="/v1/chat/completions",
            api_key=key,
            model=self.cfg.model,
            messages=messages,
            timeout_seconds=self.cfg.timeout_seconds,
        ):
            yield token

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg))
