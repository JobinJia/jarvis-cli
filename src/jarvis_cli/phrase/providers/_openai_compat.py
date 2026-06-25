"""Shared OpenAI-compatible chat-completions call for phrase providers.

DeepSeek, OpenAI, SiliconFlow and Zhipu all speak the same chat-completions
shape; they differ only in ``base_url`` and the path (Zhipu omits the ``/v1``
segment — a 404 trap if hard-coded). Keeping the one POST + parse here lets the
per-provider files stay thin adapters that just supply those two values.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def chat_completion(
    *,
    base_url: str,
    path: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> str:
    """POST ``messages`` to ``base_url + path`` (built explicitly to avoid any
    base-url join ambiguity) and return the assistant text."""
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        r = await client.post(
            base_url.rstrip("/") + path,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 80,
            },
        )
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()


async def chat_completion_stream(
    *,
    base_url: str,
    path: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> AsyncIterator[str]:
    """POST with ``stream: true`` and yield ``choices[0].delta.content`` tokens.

    Parses the standard OpenAI SSE format (``data: {...}`` lines).  The
    ``[DONE]`` sentinel and empty deltas are silently skipped.
    """
    url = base_url.rstrip("/") + path
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 80,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token
