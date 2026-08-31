"""Daemon-side skill service: scan -> index -> retrieve -> injection text.

Holds the resident embedding model and the per-session "already injected" sets,
so the hook stays a thin stateless client. All methods are synchronous and
CPU-bound; the daemon calls `query`/`refresh` via `asyncio.to_thread` to keep
the event loop responsive.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from loguru import logger

from ..config import SkillsConfig
from .catalog import scan_skills
from .embedder import Embedder, EmbedderUnavailable
from .index import ensure_index
from .injector import (
    InjectionPolicy,
    build_clarify,
    build_injection,
    gate_matches,
)
from .retriever import Match, SkillRetriever

# Cap how many sessions we remember injected-skill state for, so a long-lived
# daemon doesn't grow unbounded across many CC/Codex tabs.
_MAX_SESSIONS = 256


class SkillService:
    def __init__(self, cfg: SkillsConfig, embedder: Embedder | None = None) -> None:
        self.cfg = cfg
        self._embedder = embedder or Embedder(cfg.model_name, cache_dir=cfg.cache_dir)
        self._retriever: SkillRetriever | None = None
        self._ready = False
        self._unavailable = False
        self._injected: OrderedDict[str, set[str]] = OrderedDict()
        self._policy = InjectionPolicy(
            high_threshold=cfg.high_threshold,
            med_threshold=cfg.med_threshold,
            max_skills=cfg.max_skills,
            max_body_chars=cfg.max_body_chars,
            total_char_budget=cfg.total_char_budget,
        )

    def ensure_ready(self) -> bool:
        """Build the index + retriever on first use. Returns False (and stays
        false) when the embedding stack is missing — the daemon then no-ops."""
        if self._ready:
            return True
        if self._unavailable:
            return False
        try:
            self._rebuild()
            self._ready = True
            return True
        except EmbedderUnavailable:
            logger.warning("skills: embedding stack unavailable; retrieval disabled")
            self._unavailable = True
            return False

    def _rebuild(self) -> None:
        records = scan_skills()
        index = ensure_index(Path(self.cfg.index_dir), self._embedder, records)
        self._retriever = SkillRetriever(self._embedder, index)
        logger.info("skills: service ready ({} skills)", self._retriever.size)

    def refresh(self) -> None:
        """Rescan and rebuild if the on-disk skills changed. Cheap when nothing
        changed (hash compare); only re-embeds new/edited skills' set."""
        if self._unavailable:
            return
        try:
            self._rebuild()
            self._ready = True
        except EmbedderUnavailable:
            self._unavailable = True

    def _remember(self, session_id: str | None, keys: list[str]) -> None:
        if not session_id or not keys:
            return
        cur = self._injected.get(session_id)
        if cur is None:
            cur = set()
            self._injected[session_id] = cur
            while len(self._injected) > _MAX_SESSIONS:
                self._injected.popitem(last=False)
        cur.update(keys)
        self._injected.move_to_end(session_id)

    def gate(self, text: str) -> dict:
        """Retrieve + structural gate. Returns
        ``{"candidates": [Match...], "matches": [...]}``.

        ``candidates`` are gate-passed Match objects ready for optional LLM
        verification; the daemon builds injection text via ``build`` after
        verification. ``matches`` is the full retrieval list for telemetry.
        Never raises."""
        empty: dict = {"candidates": [], "matches": []}
        if not self.ensure_ready() or self._retriever is None:
            return empty
        try:
            matches = self._retriever.query(text, k=self.cfg.top_k)
            candidates = gate_matches(
                matches, med_threshold=self._policy.med_threshold
            )
            return {
                "candidates": candidates,
                "matches": [
                    {"name": m.record.name, "score": round(m.score, 4)}
                    for m in matches
                ],
            }
        except Exception as exc:  # noqa: BLE001 — hook must never break the prompt
            logger.warning("skills: gate failed ({})", exc)
            return empty

    def build(
        self,
        candidates: list[Match],
        *,
        session_id: str | None = None,
        query: str | None = None,
    ) -> dict:
        """Build injection text from intent-confirmed candidates. Applies the
        body/menu tiering, the vague-query guard (via *query*) and per-session
        dedup, then records what was injected. Returns
        ``{"context": str|None, "mode": str}``. Never raises.

        Candidates are a subset of what ``gate`` returned, so the gate inside
        ``build_injection`` re-passes them unchanged."""
        empty: dict = {"context": None, "mode": "none"}
        if not candidates:
            return empty
        try:
            already = self._injected.get(session_id or "", set())
            result = build_injection(
                candidates,
                policy=self._policy,
                already_injected=already,
                query=query,
            )
            self._remember(session_id, result.injected_keys)
            return {"context": result.text, "mode": result.mode}
        except Exception as exc:  # noqa: BLE001 — hook must never break the prompt
            logger.warning("skills: build failed ({})", exc)
            return empty

    def clarify(self, candidates: list[Match]) -> dict:
        """Render ambiguous candidates as a clarify note (verifier said
        ``unclear``). No body, no dedup — the model is told to ask the user
        first. Returns ``{"context": str|None, "mode": str}``. Never raises."""
        try:
            result = build_clarify(candidates)
            return {"context": result.text, "mode": result.mode}
        except Exception as exc:  # noqa: BLE001 — hook must never break the prompt
            logger.warning("skills: clarify failed ({})", exc)
            return {"context": None, "mode": "none"}

    def query(self, text: str, *, session_id: str | None = None) -> dict:
        """Convenience: gate + build with no LLM verification. Used for daemon
        warmup; the live request path goes through ``gate`` -> verifier ->
        ``build`` so it can drop vocabulary-only false positives."""
        gated = self.gate(text)
        built = self.build(
            gated["candidates"], session_id=session_id, query=text
        )
        return {
            "context": built["context"],
            "mode": built["mode"],
            "matches": gated["matches"],
        }
