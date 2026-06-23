"""SiliconFlow (硅基流动) provider — OpenAI-compatible chat API.

A second free cloud fallback alongside Zhipu, on an independent rate-limit
pool. Standard OpenAI endpoint (``/v1/chat/completions``); the key is resolved
from config.toml's inline ``api_key`` first, else the env var.
"""
from __future__ import annotations

from ...config import SiliconFlowConfig, resolve_api_key
from ._openai_compat import chat_completion
from .base import PhraseProvider


class SiliconFlowProvider(PhraseProvider):
    name = "siliconflow"

    def __init__(self, cfg: SiliconFlowConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        return await chat_completion(
            base_url=self.cfg.base_url,
            path="/v1/chat/completions",
            api_key=key,
            model=self.cfg.model,
            messages=messages,
            timeout_seconds=self.cfg.timeout_seconds,
        )

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg))
