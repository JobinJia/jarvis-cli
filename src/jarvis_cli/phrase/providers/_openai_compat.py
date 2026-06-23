"""Shared OpenAI-compatible chat-completions call for phrase providers.

DeepSeek, OpenAI, SiliconFlow and Zhipu all speak the same chat-completions
shape; they differ only in ``base_url`` and the path (Zhipu omits the ``/v1``
segment — a 404 trap if hard-coded). Keeping the one POST + parse here lets the
per-provider files stay thin adapters that just supply those two values.
"""
from __future__ import annotations

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
