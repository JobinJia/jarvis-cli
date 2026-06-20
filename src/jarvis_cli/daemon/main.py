"""Daemon entrypoint: assembles the full pipeline and runs forever."""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

from loguru import logger

from ..briefing import WeatherCache, compose_briefing
from ..config import DEFAULT_CONFIG_PATH, Config, load_config
from ..phrase.language import detect_for
from ..phrase.providers.anthropic import AnthropicProvider
from ..phrase.providers.base import PhraseProvider
from ..phrase.providers.deepseek import DeepSeekProvider
from ..phrase.providers.ollama import OllamaProvider
from ..phrase.providers.openai import OpenAIProvider
from ..phrase.router import PhraseRouter
from ..player import play, play_stream
from ..tts.engine import TTSEngine
from ..tts.providers.base import TTSProvider
from ..tts.providers.cosyvoice import CosyVoiceProvider
from ..tts.providers.elevenlabs import ElevenLabsProvider
from ..tts.providers.piper import PiperProvider
from ..tts.providers.say import SayProvider
from ..tts.providers.xtts import XTTSProvider
from ..types import Event
from .dedup import DedupWindow
from .health import HealthServer
from .listener import serve_unix_socket
from .queue import BoundedEventQueue


def _make_phrase_provider(name: str, cfg: Config) -> PhraseProvider | None:
    factories = {
        "deepseek": lambda: DeepSeekProvider(cfg.llm.deepseek),
        "anthropic": lambda: AnthropicProvider(cfg.llm.anthropic),
        "openai": lambda: OpenAIProvider(cfg.llm.openai),
        "ollama": lambda: OllamaProvider(cfg.llm.ollama),
    }
    factory = factories.get(name)
    if factory is None:
        if name:  # empty string means "no fallback configured", not a typo
            logger.warning("Unknown phrase provider {!r}; skipping in chain", name)
        return None
    return factory()


def _make_tts_provider(name: str, cfg: Config) -> TTSProvider | None:
    factories = {
        "xtts": lambda: XTTSProvider(cfg.tts.xtts),
        "cosyvoice": lambda: CosyVoiceProvider(cfg.tts.cosyvoice),
        "piper": lambda: PiperProvider(cfg.tts.piper),
        "elevenlabs": lambda: ElevenLabsProvider(cfg.tts.elevenlabs),
        "say": lambda: SayProvider(),
    }
    factory = factories.get(name)
    if factory is None:
        if name:
            logger.warning("Unknown TTS provider {!r}; skipping in chain", name)
        return None
    return factory()


class Daemon:
    def __init__(self, cfg: Config, health_port: int = 9527) -> None:
        self.cfg = cfg
        self.queue = BoundedEventQueue(maxsize=cfg.behavior.queue_max_size)
        self.dedup = DedupWindow(window_seconds=cfg.behavior.dedup_window_seconds)
        self.router = PhraseRouter(
            primary=_make_phrase_provider(cfg.llm.provider, cfg),
            fallback=_make_phrase_provider(cfg.llm.fallback, cfg),
            cfg=cfg,
            on_primary_fallback=self._announce_phrase_fallback,
        )
        primary_tts = _make_tts_provider(cfg.tts.provider, cfg) or SayProvider()
        fallback_tts = _make_tts_provider(cfg.tts.fallback, cfg)
        self.tts = TTSEngine(primary=primary_tts, fallback=fallback_tts)
        self.health = HealthServer(
            host="127.0.0.1",
            port=health_port,
            state_getter=self._snapshot,
        )
        self._last_text: str | None = None
        self._current_proc: asyncio.subprocess.Process | None = None
        self._current_session_id: str | None = None
        self._cancelled_sessions: set[str] = set()
        # Throttle Jarvis's voice alert when the local phrase provider
        # quietly slips onto the cloud fallback. 5 min between announcements
        # is enough for the user to notice without spamming during an
        # extended ollama outage.
        self._last_phrase_fallback_alert_at: float = 0.0
        self._phrase_fallback_alert_throttle_s: float = 300.0
        # Session briefing: shared weather cache + per-machine throttle
        # (user-configurable; default 0 = announce every session_start).
        self._weather_cache = WeatherCache(
            ttl_seconds=cfg.behavior.session_briefing.weather_ttl_seconds,
        )
        self._last_briefing_at: float = 0.0
        # Shared embedding model for RAG retrieval (skills + MCP intent
        # routing).  Created once; both services share the instance to avoid
        # loading the ~640 MB ONNX model twice.
        self.skills: object | None = None
        self.mcp: object | None = None
        _embedder = None
        if cfg.skills.enabled or cfg.mcp.enabled:
            try:
                from ..retrieval.embedder import Embedder

                _embedder = Embedder(
                    cfg.skills.model_name, cache_dir=cfg.skills.cache_dir,
                )
            except ImportError as exc:
                logger.warning(
                    "retrieval stack unavailable ({}); "
                    "install jarvis-cli[skills]", exc,
                )
        if cfg.skills.enabled and _embedder is not None:
            try:
                from ..skills.service import SkillService

                self.skills = SkillService(cfg.skills, _embedder)
                logger.info("skills: retrieval enabled")
            except ImportError as exc:
                logger.warning(
                    "skills deps missing ({}); install jarvis-cli[skills]", exc,
                )
        if cfg.mcp.enabled and _embedder is not None:
            try:
                from ..mcp.service import McpService

                self.mcp = McpService(cfg.mcp, _embedder)
                logger.info("mcp: intent routing enabled")
            except ImportError as exc:
                logger.warning("mcp deps missing ({})", exc)

    def _snapshot(self) -> dict:
        return {
            "queue_size": self.queue.size,
            "queue_capacity": self.queue.maxsize,
            "dropped": self.queue.dropped_count,
            "last_text": self._last_text,
        }

    async def _announce_phrase_fallback(self, primary_name: str) -> None:
        """Wired into PhraseRouter; fires when the primary LLM (typically
        local ollama) fails and the cloud fallback takes over. Throttle to
        one alert per `_phrase_fallback_alert_throttle_s`, then enqueue a
        pre-baked Jarvis line so the user audibly notices."""
        now = time.monotonic()
        if now - self._last_phrase_fallback_alert_at < self._phrase_fallback_alert_throttle_s:
            return
        self._last_phrase_fallback_alert_at = now
        text = (
            f"Sir, the local language model {primary_name} appears unreachable. "
            "I am falling back to the cloud."
        )
        logger.warning(
            "Phrase primary ({}) unreachable — alerting via voice", primary_name,
        )
        alert = Event(
            notification_type="idle_prompt",
            tool_name=None, tool_input={},
            text=text, lang="en",
        )
        await self.queue.put_or_drop(alert)

    async def cancel_session(self, session_id: str) -> None:
        """Cancel any in-flight audio for `session_id` and drop its queued events."""
        self._cancelled_sessions.add(session_id)
        self.queue.drop_matching(lambda e: e.session_id == session_id)
        proc = self._current_proc
        if proc is not None and self._current_session_id == session_id:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def _on_event(self, event: Event) -> None:
        dkey = event.dedup_key()
        if event.notification_type not in self.cfg.behavior.events:
            logger.debug("DROP filtered-by-events-allowlist key={}", dkey)
            return
        # session_start has its own gate (enabled flag + per-machine floor)
        # in addition to the generic dedup. Drop early so we don't waste a
        # queue slot when the user opens five tabs in a minute.
        if event.notification_type == "session_start":
            sb = self.cfg.behavior.session_briefing
            if not sb.enabled:
                logger.debug("DROP session_briefing.enabled=false key={}", dkey)
                return
            now = time.monotonic()
            if sb.min_interval_seconds > 0 and \
                    now - self._last_briefing_at < sb.min_interval_seconds:
                logger.debug(
                    "DROP session_start throttled key={} last={:.0f}s ago",
                    dkey, now - self._last_briefing_at,
                )
                return
            self._last_briefing_at = now
        if self.dedup.is_duplicate(event):
            logger.debug("DROP dedup-hit key={}", dkey)
            return
        logger.debug("QUEUE key={}", dkey)
        await self.queue.put_or_drop(event)

    async def _verify_mcp(self, text: str, candidates: list) -> list:
        """LLM-verify MCP candidates. Falls back to all candidates on failure."""
        try:
            from ..mcp.verifier import verify_candidates

            return await verify_candidates(
                text, candidates, self.cfg.llm.ollama,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("mcp: verification skipped ({})", exc)
            return candidates

    async def _on_query(self, payload: dict) -> dict:
        """Request/response handler for the hook's skill_query / skill_refresh.
        Now also runs MCP intent matching with LLM verification.
        Runs CPU-bound retrieval off the event loop. Never raises."""
        if payload.get("command") == "skill_refresh":
            if self.skills is not None:
                await asyncio.to_thread(self.skills.refresh)
            if self.mcp is not None:
                await asyncio.to_thread(self.mcp.refresh)
            return {"ok": True}

        text = payload.get("text") or ""
        sid = payload.get("session_id")

        skill_result: dict = {"context": None, "mode": "none", "matches": []}
        if self.skills is not None:
            skill_result = await asyncio.to_thread(
                self.skills.query, text, session_id=sid,
            )

        mcp_context: str | None = None
        if self.mcp is not None:
            mcp_result = await asyncio.to_thread(self.mcp.query, text)
            candidates = mcp_result.get("candidates", [])
            if candidates:
                verified = await self._verify_mcp(text, candidates)
                if verified:
                    from ..mcp.service import _build_injection

                    mcp_context = _build_injection(
                        verified,
                        high_threshold=self.cfg.mcp.high_threshold,
                        med_threshold=self.cfg.mcp.med_threshold,
                    )

        parts = [
            p for p in (skill_result.get("context"), mcp_context)
            if p
        ]
        merged = "\n\n---\n\n".join(parts) if parts else None

        return {
            "context": merged,
            "mode": skill_result.get("mode", "none"),
            "matches": skill_result.get("matches", []),
        }

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            await self._process_one(event)

    async def _process_one(self, event: Event) -> None:
        sid = event.session_id
        if sid and sid in self._cancelled_sessions:
            # Stale cancel signal preceded the event; clear and play normally.
            self._cancelled_sessions.discard(sid)

        def _register(proc: asyncio.subprocess.Process) -> None:
            self._current_proc = proc
            self._current_session_id = sid

        try:
            if event.text is not None:
                # Caller pre-baked the phrase; skip the LLM entirely.
                text = event.text
                lang = event.lang or "en"
            elif event.notification_type == "session_start":
                # Compose Jarvis briefing inline. We hand the primary
                # phrase provider (Ollama by default) in so the line is
                # freshly phrased each session — falls back to a
                # rotating template if the LLM is unreachable. Weather
                # fetch is best-effort with a TTL cache.
                text, lang = await compose_briefing(
                    self.cfg.behavior.session_briefing,
                    cache=self._weather_cache,
                    llm=self.router.primary,
                    humor_level=self.cfg.behavior.humor_level,
                )
            else:
                lang = (
                    detect_for(event.cwd)
                    if self.cfg.behavior.voice_language == "auto"
                    else self.cfg.behavior.voice_language  # type: ignore[assignment]
                )
                text = await self.router.phrase(event, lang=lang)
            self._last_text = text
            # Phrasing/briefing above can take seconds (LLM round-trip). If the
            # user already acted on this session in that window, the line is
            # stale — drop it before any play proc starts rather than speak a
            # step too late. No proc is registered yet, so the cancel that just
            # landed had nothing to kill; this guard is what honours it.
            if sid and sid in self._cancelled_sessions:
                logger.debug("DROP cancelled-before-play sid={}", sid)
                return
            logger.debug(
                "PLAY type={} sid={} text={!r}",
                event.notification_type, sid, text[:80],
            )
            if await self._try_stream(
                text, lang, event.voice_id,
                on_spawn=_register, session_id=sid,
            ):
                return
            # Skip synth fallback if a cancel arrived between stream attempt
            # and now — otherwise the same line gets re-synthesized + replayed.
            if sid and sid in self._cancelled_sessions:
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = Path(tmp.name)
            await self.tts.synthesize(text, lang, out_path, voice_id=event.voice_id)
            # synthesize() is the long pole for non-streaming providers
            # (CosyVoice ~seconds) and registers no play proc, so a cancel that
            # lands mid-synth kills nothing. Re-check before playing so the
            # finished file is discarded instead of surfacing as stale audio.
            if sid and sid in self._cancelled_sessions:
                logger.debug("DROP cancelled-after-synth sid={}", sid)
                try:
                    out_path.unlink()
                except OSError:
                    pass
                return
            try:
                await play(out_path, on_spawn=_register)
            finally:
                try:
                    out_path.unlink()
                except OSError:
                    pass
        except Exception as exc:
            if sid and sid in self._cancelled_sessions:
                logger.debug("playback cancelled for session {}", sid)
            else:
                logger.exception("worker failed: {}", exc)
        finally:
            self._current_proc = None
            self._current_session_id = None
            if sid:
                self._cancelled_sessions.discard(sid)

    async def _try_stream(
        self, text: str, lang, voice_id: str | None,
        *, on_spawn=None, session_id: str | None = None,
    ) -> bool:
        """If the primary TTS supports streaming, pipe chunks straight to
        ffplay so playback begins before synthesis completes. Returns True
        on success (or on cancel — see below); False on any other failure so
        the caller can fall back to the file-based synth+afplay path.

        When the worker kills ffplay to cancel playback, play_stream raises
        a RuntimeError. That is NOT a TTS failure — re-synthesizing and
        falling back to afplay would replay the line we just killed.
        Detect this by checking `_cancelled_sessions` and return True so the
        caller treats it as "playback already concluded".
        """
        primary = self.tts.primary
        if not getattr(primary, "supports_streaming", False):
            return False
        try:
            await play_stream(
                primary.stream(text, lang, voice_id=voice_id),
                on_spawn=on_spawn,
            )
            return True
        except Exception as exc:
            if session_id and session_id in self._cancelled_sessions:
                logger.debug("stream cancelled for session {}", session_id)
                return True
            logger.warning("Streaming TTS failed, falling back to synth: {}", exc)
            return False

    async def _prewarm_skills(self) -> None:
        """Load the model + index AND run one throwaway query at daemon start so
        the user's first prompt doesn't eat the ~1s cold path. The model load is
        cheap; the real cost is onnxruntime's first inference, which a cached
        index never triggers — so we embed once here. Best-effort: any failure
        just defers the cost to the first real query."""
        if self.skills is not None:
            try:
                ok = await asyncio.to_thread(self.skills.ensure_ready)
                if ok:
                    await asyncio.to_thread(self.skills.query, "warmup")
                logger.info("skills: prewarm {}", "ready" if ok else "unavailable")
            except Exception as exc:  # noqa: BLE001
                logger.warning("skills: prewarm failed ({})", exc)
        if self.mcp is not None:
            try:
                ok = await asyncio.to_thread(self.mcp.ensure_ready)
                if ok:
                    await asyncio.to_thread(self.mcp.query, "warmup")
                logger.info("mcp: prewarm {}", "ready" if ok else "unavailable")
            except Exception as exc:  # noqa: BLE001
                logger.warning("mcp: prewarm failed ({})", exc)

    async def run(self) -> None:
        await self.health.start()
        worker_task = asyncio.create_task(self._worker())
        prewarm_task = asyncio.create_task(self._prewarm_skills())
        try:
            await serve_unix_socket(
                Path(self.cfg.paths.socket),
                self._on_event,
                on_cancel=self.cancel_session,
                on_query=(
                    self._on_query
                    if self.skills is not None or self.mcp is not None
                    else None
                ),
            )
        finally:
            worker_task.cancel()
            prewarm_task.cancel()
            for t in (worker_task, prewarm_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            await self.health.stop()


def main() -> int:
    parser = argparse.ArgumentParser(prog="jarvis-cli-daemon")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--health-port", type=int, default=9527)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.add(cfg.paths.log, rotation="10 MB", retention="14 days", level="INFO")
    try:
        asyncio.run(Daemon(cfg, health_port=args.health_port).run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
