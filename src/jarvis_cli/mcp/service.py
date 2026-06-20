"""Daemon-side MCP intent routing: registry -> index -> retrieve -> injection.

Shares the resident Embedder with SkillService so the 640 MB model loads
once.  All methods are synchronous and CPU-bound; the daemon calls them
via ``asyncio.to_thread``.
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ..config import McpConfig
from ..retrieval.embedder import Embedder, EmbedderUnavailable
from ..retrieval.index import ensure_index
from ..retrieval.retriever import Match, Retriever
from .registry import (
    McpServerRecord,
    load_registry,
    record_from_dict,
    record_to_dict,
)

_BODY_PREAMBLE = (
    "MCP server(s) auto-matched for this request. "
    "Connect via ``mcp__mcp-router__mcp_router__add_server`` if not "
    "already available, then use the server's tools via "
    "``mcp__mcp-router__mcp_router__call``:\n\n"
)
_MENU_PREAMBLE = (
    "MCP servers that may help with this request "
    "(connect with ``mcp__mcp-router__mcp_router__add_server`` if needed):\n"
)


def _connect_snippet(rec: McpServerRecord) -> str:
    params = {"name": rec.name, **rec.connect}
    return json.dumps(params, ensure_ascii=False)


_COSINE_SOLO_FLOOR = 0.50


def _has_lexical_signal(m: Match) -> bool:
    """True when the keyword / name boost contributed to the hybrid score —
    i.e. the match isn't ONLY semantic."""
    return m.score - m.cosine > 1e-4


def _build_injection(
    matches: list[Match],
    *,
    high_threshold: float,
    med_threshold: float,
) -> str | None:
    """Build injection text from pre-verified candidates.

    Candidates are already gate-filtered and LLM-verified by the daemon;
    this function only decides the presentation tier (connect vs suggest).
    """
    if not matches:
        return None

    strong = [m for m in matches if m.score >= high_threshold]
    if strong:
        parts: list[str] = []
        for m in strong[:3]:
            rec: McpServerRecord = m.record
            parts.append(
                f"### {rec.name}\n"
                f"{rec.description}\n"
                f"**Connect:** `mcp__mcp-router__mcp_router__add_server"
                f"({_connect_snippet(rec)})`"
            )
        return _BODY_PREAMBLE + "\n\n".join(parts)

    lines = [
        f"- `{m.record.name}` — {m.record.description}"
        for m in matches[:5]
    ]
    return _MENU_PREAMBLE + "\n".join(lines)


class McpService:
    def __init__(self, cfg: McpConfig, embedder: Embedder) -> None:
        self.cfg = cfg
        self._embedder = embedder
        self._retriever: Retriever | None = None
        self._ready = False
        self._unavailable = False

    def ensure_ready(self) -> bool:
        if self._ready:
            return True
        if self._unavailable:
            return False
        try:
            self._rebuild()
            self._ready = True
            return True
        except EmbedderUnavailable:
            logger.warning("mcp: embedding stack unavailable; routing disabled")
            self._unavailable = True
            return False

    def _rebuild(self) -> None:
        records = load_registry(self.cfg.registry_path)
        if not records:
            self._retriever = None
            logger.info("mcp: registry empty; routing inactive")
            return
        index = ensure_index(
            Path(self.cfg.index_dir),
            self._embedder,
            records,
            record_to_dict,
            record_from_dict,
        )
        self._retriever = Retriever(self._embedder, index)
        logger.info("mcp: service ready ({} servers)", self._retriever.size)

    def refresh(self) -> None:
        if self._unavailable:
            return
        try:
            self._rebuild()
            self._ready = True
        except EmbedderUnavailable:
            self._unavailable = True

    def query(self, text: str) -> dict:
        """Return ``{"candidates": [...], "matches": [...]}``.

        ``candidates`` are gate-passed Match objects ready for optional LLM
        verification. The daemon builds injection text after verification.
        Never raises.
        """
        empty: dict = {"candidates": [], "matches": []}
        if not self.ensure_ready() or self._retriever is None:
            return empty
        try:
            matches = self._retriever.query(text, k=self.cfg.top_k)
            gated = [
                m for m in matches
                if m.score >= self.cfg.med_threshold
                and (_has_lexical_signal(m) or m.cosine >= _COSINE_SOLO_FLOOR)
            ]
            return {
                "candidates": gated,
                "matches": [
                    {"name": m.record.name, "score": round(m.score, 4)}
                    for m in matches
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp: query failed ({})", exc)
            return empty
