"""LLM-based verification for MCP intent matches.

After the retriever + gate selects candidates, a lightweight LLM call
confirms whether the user's request *actually* needs each server.
Eliminates false positives that share vocabulary but not intent
(e.g. "提交代码" matching "code intelligence" because of "代码").

Falls back gracefully: if the LLM is unavailable or slow, all candidates
pass through (gate-only behavior).
"""
from __future__ import annotations

import re

import httpx
from loguru import logger

from ..config import OllamaConfig
from ..retrieval.retriever import Match

_SYSTEM = (
    "You decide which tool servers a user request actually needs. "
    "Reply ONLY with the numbers of servers that are needed "
    "(comma-separated), or \"none\". No explanation."
)


async def verify_candidates(
    text: str,
    candidates: list[Match],
    ollama_cfg: OllamaConfig,
    *,
    timeout_s: float = 1.5,
) -> list[Match]:
    """Return the subset of *candidates* the LLM confirms are relevant.

    On any failure (LLM down, timeout, unparseable reply) returns all
    candidates unchanged — gate-only is the fallback, not silence.
    """
    if not candidates:
        return []

    server_lines = "\n".join(
        f"{i + 1}. {m.record.name} — {m.record.description}"
        for i, m in enumerate(candidates)
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Request: \"{text}\"\n\nServers:\n{server_lines}"},
    ]

    try:
        async with httpx.AsyncClient(
            base_url=ollama_cfg.base_url, timeout=timeout_s,
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": ollama_cfg.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 20},
                },
            )
            r.raise_for_status()
            reply = r.json()["message"]["content"].strip().lower()
    except Exception as exc:
        logger.debug("mcp: verifier unavailable ({}); passing all candidates", exc)
        return candidates

    logger.debug("mcp: verifier reply={!r} for {!r}", reply, text[:60])

    if "none" in reply:
        return []

    indices: set[int] = set()
    for tok in re.findall(r"\d+", reply):
        idx = int(tok) - 1
        if 0 <= idx < len(candidates):
            indices.add(idx)

    if not indices:
        return candidates

    return [candidates[i] for i in sorted(indices)]
