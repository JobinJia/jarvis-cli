"""Daemon entrypoint: assembles the full pipeline and runs forever."""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from loguru import logger

from ..briefing import WeatherCache, compose_briefing, detect_city
from ..config import DEFAULT_CONFIG_PATH, Config, load_config
from ..notify import webhook as webhook_notify
from ..phrase.language import detect_for
from ..phrase.providers.anthropic import AnthropicProvider
from ..phrase.providers.base import PhraseProvider
from ..phrase.providers.deepseek import DeepSeekProvider
from ..phrase.providers.ollama import OllamaProvider
from ..phrase.providers.openai import OpenAIProvider
from ..phrase.providers.siliconflow import SiliconFlowProvider
from ..phrase.providers.zhipu import ZhipuProvider
from ..phrase.router import PhraseRouter
from ..player import (
    Cancellable,
    PCMPlayer,
    StreamPlayer,
    open_pcm_sink,
    play,
    play_stream,
)
from ..tts.engine import TTSEngine
from ..tts.providers.base import TTSProvider
from ..tts.providers.cosyvoice import CosyVoiceProvider
from ..tts.providers.elevenlabs import ElevenLabsProvider
from ..tts.providers.piper import PiperProvider
from ..tts.providers.say import SayProvider
from ..tts.providers.xtts import XTTSProvider
from ..types import Event, Lang, emotion_for
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
        "zhipu": lambda: ZhipuProvider(cfg.llm.zhipu),
        "siliconflow": lambda: SiliconFlowProvider(cfg.llm.siliconflow),
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
        # Build the fallback chain: prefer the multi-level `fallbacks` list,
        # else wrap the singular `fallback` (back-compat). Unconstructable
        # providers (missing dep/key) are dropped here; a key-less one that
        # slips through just raises at call time and the router skips to the
        # next link.
        fb_names = cfg.llm.fallbacks or (
            [cfg.llm.fallback] if cfg.llm.fallback else []
        )
        fb_providers = [
            p for n in fb_names
            if (p := _make_phrase_provider(n, cfg)) is not None
        ]
        self.router = PhraseRouter(
            primary=_make_phrase_provider(cfg.llm.provider, cfg),
            fallbacks=fb_providers,
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
        # Keep strong refs to in-flight webhook tasks so they aren't garbage
        # collected mid-flight (asyncio only holds weak refs to tasks).
        self._webhook_tasks: set[asyncio.Task] = set()
        # The in-flight playback handle: an afplay/ffplay Process or a
        # PCMPlayer — anything cancel_session can kill() (see Cancellable).
        self._current_proc: Cancellable | None = None
        self._current_session_id: str | None = None
        self._cancelled_sessions: set[str] = set()
        # Phrase prefetch: while event N plays, the worker phrases the event
        # at the queue head so N+1 skips its LLM round-trip. Keyed by event
        # IDENTITY (the queue hands back the same object), so a cached entry
        # for a dropped/cancelled event simply never matches. No locking:
        # everything here runs on the one event loop.
        self._prefetched: tuple[Event, str, Lang] | None = None
        # Whether the event currently being dispatched got its text from the
        # prefetch cache — set by _worker right before dispatch, read by
        # _process_one for the TIMING log. An instance attr (not a kwarg)
        # so _process_one's signature — and every test that patches it —
        # stays unchanged.
        self._dispatch_prefetched: bool = False
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
        # No need to touch `_prefetched` here: dropping the queued event means
        # the worker's identity check never matches it, so the cached phrase
        # is discarded on the next dequeue.
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

    async def _verify(self, text: str, candidates: list, *, noun: str):
        """LLM-classify intent candidates (skills or MCP). Returns a
        ``VerifyResult`` with status confirmed/none/unclear. On any error the
        status is ``unclear`` — we ask the user rather than fabricate a confirm
        we couldn't actually make."""
        from ..retrieval.verifier import UNCLEAR, VerifyResult, verify_candidates

        try:
            return await verify_candidates(
                text, candidates, self.cfg.llm.ollama, noun=noun,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("{}: verification skipped ({})", noun, exc)
            return VerifyResult(UNCLEAR)

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

        # Skills and MCP are independent pipelines (separate gate -> verify ->
        # build). Run them concurrently so a prompt that hits both gates costs
        # one LLM round-trip of latency, not two.
        (skill_context, skill_mode, skill_matches), mcp_context = await asyncio.gather(
            self._skill_context(text, sid),
            self._mcp_context(text),
        )

        parts = [p for p in (skill_context, mcp_context) if p]
        merged = "\n\n---\n\n".join(parts) if parts else None

        return {
            "context": merged,
            "mode": skill_mode,
            "matches": skill_matches,
        }

    async def _skill_context(
        self, text: str, sid: str | None
    ) -> tuple[str | None, str, list]:
        """Skills pipeline: gate -> LLM-classify -> build. Confirmed intent
        body/menu-injects; "unclear" injects a clarify note (never a command);
        "none" injects nothing. Returns (context, mode, telemetry matches)."""
        from ..retrieval.verifier import CONFIRMED, UNCLEAR

        if self.skills is None:
            return None, "none", []
        gate = await asyncio.to_thread(self.skills.gate, text)
        matches = gate.get("matches", [])
        candidates = gate.get("candidates", [])
        if not candidates:
            return None, "none", matches
        vr = await self._verify(text, candidates, noun="skill")
        if vr.status == CONFIRMED and vr.matches:
            built = await asyncio.to_thread(
                self.skills.build, vr.matches, session_id=sid, query=text,
            )
        elif vr.status == UNCLEAR:
            built = await asyncio.to_thread(self.skills.clarify, candidates)
        else:
            return None, "none", matches
        return built.get("context"), built.get("mode", "none"), matches

    async def _mcp_context(self, text: str) -> str | None:
        """MCP pipeline: gate -> LLM-classify -> connect/clarify injection.
        Mirrors _skill_context but returns just the context string."""
        from ..mcp.service import _build_clarify, _build_injection
        from ..retrieval.verifier import CONFIRMED, UNCLEAR

        if self.mcp is None:
            return None
        result = await asyncio.to_thread(self.mcp.query, text)
        candidates = result.get("candidates", [])
        if not candidates:
            return None
        vr = await self._verify(text, candidates, noun="tool server")
        if vr.status == CONFIRMED and vr.matches:
            return _build_injection(
                vr.matches,
                high_threshold=self.cfg.mcp.high_threshold,
                med_threshold=self.cfg.mcp.med_threshold,
                query=text,
            )
        if vr.status == UNCLEAR:
            return _build_clarify(candidates)
        return None

    def _wants_streaming(self, event: Event) -> bool:
        """Route to the streaming pipeline when the config flag is set AND the
        event is an LLM-phrased one (not pre-baked text, not session_start
        which has its own compose path)."""
        return (
            self.cfg.behavior.streaming_pipeline
            and event.text is None
            and event.notification_type != "session_start"
        )

    def _is_stale(self, event: Event) -> bool:
        """A backlog burst (long playback blocking the queue) can dequeue a
        notification long after the user already acted on it — speaking it
        then is noise. Pre-baked text and session_start briefings are exempt
        (those should speak whenever they surface), as are events without a
        received_at stamp (0.0 — synthetic/test events carry no timestamp)."""
        max_age = self.cfg.behavior.stale_event_max_age_seconds
        if (
            max_age <= 0
            or event.text is not None
            or event.notification_type == "session_start"
            or not event.received_at
        ):
            return False
        return time.time() - event.received_at > max_age

    async def _prefetch_next(self) -> None:
        """Phrase the event at the queue head while the current one plays.

        Zero-overhead when there's nothing to do: an empty queue makes peek()
        return None and we exit immediately, so the common single-event case
        is untouched. Pre-baked text and session_start have no LLM round-trip
        to hide (the briefing composes inline), and a stale head is about to
        be dropped at dequeue anyway — skip all of those. Best-effort: any
        failure just means the worker phrases the event itself, as before.
        """
        try:
            nxt = self.queue.peek()
            if (
                nxt is None
                or nxt.text is not None
                or nxt.notification_type == "session_start"
                or self._is_stale(nxt)
            ):
                return
            lang = (
                detect_for(nxt.cwd)
                if self.cfg.behavior.voice_language == "auto"
                else self.cfg.behavior.voice_language  # type: ignore[assignment]
            )
            emotion = emotion_for(nxt.notification_type)
            text = await self.router.phrase(nxt, lang=lang, emotion=emotion)
            self._prefetched = (nxt, text, lang)
        except Exception as exc:  # noqa: BLE001
            logger.debug("prefetch failed ({})", exc)

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            # Stale check on the ORIGINAL event, before any prefetch
            # consumption — a prefetched phrase for a stale event must not
            # resurrect it via the pre-baked exemption in _is_stale.
            if self._is_stale(event):
                logger.debug(
                    "DROP stale age={:.0f}s key={}",
                    time.time() - event.received_at, event.dedup_key(),
                )
                continue
            # Consume the prefetch cache. Identity check: the queue hands
            # back the same object it stored, so a hit means THIS event was
            # phrased during the previous playback. A miss means the cached
            # event was dropped (cancel_session) or otherwise superseded —
            # the cache is garbage either way, so always clear it.
            prefetched = (
                self._prefetched is not None and self._prefetched[0] is event
            )
            if prefetched:
                _, text, lang = self._prefetched
                event = replace(event, text=text, lang=lang)
            self._prefetched = None
            # Kick off phrasing for the event now at the queue head so its
            # LLM round-trip overlaps this event's playback. Routing
            # consequence: a prefetched event carries `text`, so
            # _wants_streaming returns False and it flows through the batch
            # path's pre-baked branch — exactly what we want under backlog
            # (full text ready before playback starts beats intra-event
            # streaming).
            prefetch_task = asyncio.create_task(self._prefetch_next())
            self._dispatch_prefetched = prefetched
            if self._wants_streaming(event):
                await self._process_one_streaming(event)
            else:
                await self._process_one(event)
            # Usually already done — playback outlasts phrasing. Awaiting
            # keeps the task lifecycle clean (no orphans, errors surface).
            await prefetch_task

    async def _process_one(self, event: Event) -> None:
        sid = event.session_id
        started = time.monotonic()
        # Capture-and-reset the worker's prefetch flag so a direct call
        # (tests, future callers) never inherits a stale True.
        prefetched = self._dispatch_prefetched
        self._dispatch_prefetched = False
        if sid and sid in self._cancelled_sessions:
            # Stale cancel signal preceded the event; clear and play normally.
            self._cancelled_sessions.discard(sid)

        def _register(proc: Cancellable) -> None:
            self._current_proc = proc
            self._current_session_id = sid

        # Derive the emotion for this event once; it flows into both the
        # phrase prompt (shaping the LLM's written text) and the TTS
        # provider (ElevenLabs voice_settings presets, etc.).
        emotion = emotion_for(event.notification_type)

        try:
            phrase_started = time.monotonic()
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
                text = await self.router.phrase(
                    event, lang=lang, emotion=emotion,
                )
            # Stage timing (DEBUG): phrase covers the LLM/briefing cost above
            # (~0 for pre-baked/prefetched text); total spans through playback.
            phrase_ms = (time.monotonic() - phrase_started) * 1000

            def _log_timing() -> None:
                logger.debug(
                    "TIMING type={} phrase_ms={:.0f} total_ms={:.0f} "
                    "prefetched={}",
                    event.notification_type, phrase_ms,
                    (time.monotonic() - started) * 1000, prefetched,
                )

            self._last_text = text
            # Remote push (opt-in). Fire-and-forget as a detached task so the
            # network call runs CONCURRENTLY with audio playback below and can
            # never delay or block TTS. `text` is the final spoken line — for
            # LLM-phrased events it was built from an already-redacted summary
            # (notify.webhook itself is fail-soft and never raises).
            if self.cfg.webhook.enabled:
                self._fire_webhook(event, text)
            # Phrasing/briefing above can take seconds (LLM round-trip). If the
            # user already acted on this session in that window, the line is
            # stale — drop it before any play proc starts rather than speak a
            # step too late. No proc is registered yet, so the cancel that just
            # landed had nothing to kill; this guard is what honours it.
            if sid and sid in self._cancelled_sessions:
                logger.debug("DROP cancelled-before-play sid={}", sid)
                return
            logger.debug(
                "PLAY type={} sid={} emotion={} text={!r}",
                event.notification_type, sid, emotion, text[:80],
            )
            if await self._try_stream(
                text, lang, event.voice_id,
                on_spawn=_register, session_id=sid,
                emotion=emotion,
            ):
                _log_timing()
                return
            # Skip synth fallback if a cancel arrived between stream attempt
            # and now — otherwise the same line gets re-synthesized + replayed.
            if sid and sid in self._cancelled_sessions:
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = Path(tmp.name)
            await self.tts.synthesize(
                text, lang, out_path,
                voice_id=event.voice_id, emotion=emotion,
            )
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
            _log_timing()
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

    async def _process_one_streaming(self, event: Event) -> None:
        """Streaming pipeline: overlap LLM → sentence chunking → TTS → play.

        For each sentence the LLM produces, we synthesize and play it
        immediately — so the first sentence is audible while the LLM is still
        generating the rest. All streamed sentences feed ONE audio sink
        (in-process PCM or ffplay — see _spawn_stream_sink; spawned lazily on
        the first sentence) so consecutive sentences play gaplessly instead
        of paying a device-open/process-spawn pause between each.
        """
        sid = event.session_id
        started = time.monotonic()
        # Stage timing (DEBUG): first successful feed is our proxy for first
        # audible audio — the whole point of this pipeline is shrinking it.
        first_feed_at: float | None = None
        if sid and sid in self._cancelled_sessions:
            self._cancelled_sessions.discard(sid)

        def _register(proc: Cancellable) -> None:
            self._current_proc = proc
            self._current_session_id = sid

        # Mirror _process_one: derive the emotion once so it shapes both the
        # phrase prompt (the LLM's written tone) and every per-sentence TTS
        # call — streamed lines must not sound flatter than batch ones.
        emotion = emotion_for(event.notification_type)

        session: PCMPlayer | StreamPlayer | None = None
        try:
            lang = (
                detect_for(event.cwd)
                if self.cfg.behavior.voice_language == "auto"
                else self.cfg.behavior.voice_language  # type: ignore[assignment]
            )
            spoken_parts: list[str] = []
            async for sentence in self.router.phrase_stream(
                event, lang=lang, emotion=emotion,
            ):
                if sid and sid in self._cancelled_sessions:
                    logger.debug("DROP cancelled-mid-stream sid={}", sid)
                    break
                spoken_parts.append(sentence)
                logger.debug(
                    "STREAM-PLAY type={} sid={} emotion={} chunk={!r}",
                    event.notification_type, sid, emotion, sentence[:80],
                )
                # Try streaming TTS first (e.g. XTTS/ElevenLabs), feeding the
                # shared ffplay pipe; fall back to file-based synth+play for
                # this sentence on failure.
                primary = self.tts.primary
                if primary.supports_streaming:
                    try:
                        if session is None:
                            session = await self._spawn_stream_sink(
                                primary, on_spawn=_register,
                            )
                        await session.feed(primary.stream(
                            sentence, lang,
                            voice_id=event.voice_id, emotion=emotion,
                        ))
                        if first_feed_at is None:
                            first_feed_at = time.monotonic()
                        continue
                    except Exception as exc:
                        if sid and sid in self._cancelled_sessions:
                            # The worker killed ffplay to cancel playback —
                            # the broken pipe is the kill, not a TTS failure.
                            # Stop speaking rather than re-synthesize the
                            # line we just silenced.
                            logger.debug("stream cancelled for session {}", sid)
                            break
                        logger.warning(
                            "Streaming TTS failed, falling back to synth: {}",
                            exc,
                        )
                        # The shared pipe is in an unknown state; kill it.
                        # Later sentences may retry streaming — a fresh
                        # session gets spawned on the next attempt.
                        if session is not None:
                            await session.abort()
                            session = None
                if sid and sid in self._cancelled_sessions:
                    break
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = Path(tmp.name)
                await self.tts.synthesize(
                    sentence, lang, out_path,
                    voice_id=event.voice_id, emotion=emotion,
                )
                if sid and sid in self._cancelled_sessions:
                    try:
                        out_path.unlink()
                    except OSError:
                        pass
                    break
                try:
                    await play(out_path, on_spawn=_register)
                finally:
                    try:
                        out_path.unlink()
                    except OSError:
                        pass

            # Utterance complete (or cancelled): close the sink so it drains
            # its remaining buffer (ffplay: stdin EOF; PCM: StopStream). A
            # close error after a cancel is just the kill we issued
            # concluding — not a fault.
            if session is not None:
                try:
                    await session.close()
                except Exception as exc:
                    if sid and sid in self._cancelled_sessions:
                        logger.debug(
                            "stream session ended by cancel for session {}",
                            sid,
                        )
                    else:
                        logger.warning(
                            "stream session close failed: {}", exc,
                        )
                finally:
                    session = None

            full_text = " ".join(spoken_parts) if spoken_parts else ""
            self._last_text = full_text
            # first_feed_ms=-1 means no sentence ever streamed (every one
            # took the file-synth fallback, or the LLM produced nothing).
            logger.debug(
                "TIMING-STREAM type={} first_feed_ms={:.0f} total_ms={:.0f} "
                "sentences={}",
                event.notification_type,
                (first_feed_at - started) * 1000
                if first_feed_at is not None else -1.0,
                (time.monotonic() - started) * 1000,
                len(spoken_parts),
            )
            if full_text and self.cfg.webhook.enabled:
                self._fire_webhook(event, full_text)
        except Exception as exc:
            if sid and sid in self._cancelled_sessions:
                logger.debug("streaming playback cancelled for session {}", sid)
            else:
                logger.exception("streaming worker failed: {}", exc)
        finally:
            # Belt and braces: if an exception bailed us out with the shared
            # ffplay still live, kill it — never leave an orphan process
            # holding the pipe (abort() reaps and never raises).
            if session is not None:
                await session.abort()
            self._current_proc = None
            self._current_session_id = None
            if sid:
                self._cancelled_sessions.discard(sid)

    def _fire_webhook(self, event: Event, text: str) -> None:
        """Schedule the webhook POST as a detached task and track it so it
        isn't GC'd before completion. notify() is fail-soft (never raises),
        so the task needs no error handling of its own."""
        task = asyncio.ensure_future(
            webhook_notify.notify(self.cfg.webhook, event, text)
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)

    async def _spawn_stream_sink(
        self,
        primary: TTSProvider,
        *,
        on_spawn: Callable[[Cancellable], None] | None = None,
    ) -> PCMPlayer | StreamPlayer:
        """The one sink-selection point for both streaming call sites: raw-PCM
        providers (stream_pcm set) get the in-process sounddevice sink — which
        itself falls back to ffplay when sounddevice is unavailable — while
        container-format providers (MP3 from ElevenLabs) keep ffplay, the only
        decoder we have for those bytes.

        The PCM sink is additionally gated behind `tts.pcm_playback` (default
        off): with synthesis slower than realtime it underruns audibly, while
        ffplay buffers through the stalls — see TTSConfig for the full story.
        """
        if primary.stream_pcm is not None and self.cfg.tts.pcm_playback:
            rate, channels = primary.stream_pcm
            return await open_pcm_sink(
                rate=rate, channels=channels,
                input_args=primary.stream_input_args,
                on_spawn=on_spawn,
            )
        return await StreamPlayer.spawn(
            input_args=primary.stream_input_args, on_spawn=on_spawn,
        )

    async def _try_stream(
        self, text: str, lang, voice_id: str | None,
        *, on_spawn=None, session_id: str | None = None,
        emotion: str | None = None,
    ) -> bool:
        """If the primary TTS supports streaming, pipe chunks straight to the
        audio sink so playback begins before synthesis completes. Returns True
        on success (or on cancel — see below); False on any other failure so
        the caller can fall back to the file-based synth+afplay path.

        When the worker kill()s the sink to cancel playback, the feed raises
        (broken pipe for ffplay, PortAudioError for the PCM sink). That is
        NOT a TTS failure — re-synthesizing and falling back to afplay would
        replay the line we just killed. Detect this by checking
        `_cancelled_sessions` and return True so the caller treats it as
        "playback already concluded".
        """
        primary = self.tts.primary
        if not primary.supports_streaming:
            return False
        try:
            chunks = primary.stream(text, lang, voice_id=voice_id, emotion=emotion)
            if primary.stream_pcm is not None:
                sink = await self._spawn_stream_sink(primary, on_spawn=on_spawn)
                try:
                    await sink.feed(chunks)
                except BaseException:
                    # Conclude the sink without masking the real cause
                    # (abort never raises), then let the error decide
                    # cancel-vs-failure below.
                    await sink.abort()
                    raise
                await sink.close()
            else:
                await play_stream(
                    chunks,
                    on_spawn=on_spawn,
                    input_args=primary.stream_input_args,
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

    async def _prewarm_tts(self) -> None:
        """Let the primary provider warm its one-off state at daemon start so
        the user's first notification doesn't eat the cold path. Each provider
        knows what (if anything) it needs to warm — XTTS loads the model and
        conditioning latents; API/subprocess providers are no-ops. Best-effort:
        any failure just defers the cost to the first real event."""
        primary = self.tts.primary
        try:
            await primary.prewarm()
            logger.info("tts: prewarm ready ({})", primary.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tts: prewarm failed ({})", exc)

    async def _prewarm_briefing(self) -> None:
        """Warm the weather cache at daemon start so the first session_start
        briefing doesn't stall on the weather API. Best-effort: any failure
        just defers the fetch to the first real briefing."""
        sb = self.cfg.behavior.session_briefing
        if not sb.enabled:
            return
        try:
            city = sb.city or detect_city()
            await self._weather_cache.get(city, sb.weather_timeout_seconds)
            logger.info("briefing: weather prewarmed ({})", city)
        except Exception as exc:  # noqa: BLE001
            logger.warning("briefing: weather prewarm failed ({})", exc)

    async def run(self) -> None:
        await self.health.start()
        tasks = [
            asyncio.create_task(coro)
            for coro in (
                self._worker(),
                self._prewarm_skills(),
                self._prewarm_tts(),
                self._prewarm_briefing(),
            )
        ]
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
            for t in tasks:
                t.cancel()
            for t in tasks:
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
