"""Daemon entrypoint: assembles the full pipeline and runs forever."""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from loguru import logger

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
from ..tts.providers.elevenlabs import ElevenLabsProvider
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

    def _snapshot(self) -> dict:
        return {
            "queue_size": self.queue.size,
            "queue_capacity": self.queue.maxsize,
            "dropped": self.queue.dropped_count,
            "last_text": self._last_text,
        }

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
        if event.notification_type not in self.cfg.behavior.events:
            return
        if self.dedup.is_duplicate(event):
            logger.debug("Dedup: {}", event.dedup_key())
            return
        await self.queue.put_or_drop(event)

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
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
                else:
                    lang = (
                        detect_for(event.cwd)
                        if self.cfg.behavior.voice_language == "auto"
                        else self.cfg.behavior.voice_language  # type: ignore[assignment]
                    )
                    text = await self.router.phrase(event, lang=lang)
                self._last_text = text
                if await self._try_stream(text, lang, event.voice_id, on_spawn=_register):
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = Path(tmp.name)
                await self.tts.synthesize(text, lang, out_path, voice_id=event.voice_id)
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
        *, on_spawn=None,
    ) -> bool:
        """If the primary TTS supports streaming, pipe chunks straight to
        ffplay so playback begins before synthesis completes. Returns True
        on success; False (with a warning) on any failure so the caller can
        fall back to the file-based synth+afplay path."""
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
            logger.warning("Streaming TTS failed, falling back to synth: {}", exc)
            return False

    async def run(self) -> None:
        await self.health.start()
        worker_task = asyncio.create_task(self._worker())
        try:
            await serve_unix_socket(
                Path(self.cfg.paths.socket),
                self._on_event,
                on_cancel=self.cancel_session,
            )
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            await self.health.stop()


def main() -> int:
    parser = argparse.ArgumentParser(prog="jarvis-cc-daemon")
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
