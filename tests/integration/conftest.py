"""Shared stubs for daemon integration tests.

These tests patch `engine.synthesize` and `daemon.main.play` to exercise
wiring only — but the daemon consults `tts.primary_for(lang)` directly for
prewarm and the streaming fast path. On a dev machine with the real XTTS
model on disk, that path succeeds, bypasses both stubs (audio goes to the
stream sink, not `play`), and the tests fail — while CI, having no model,
falls back to the stubbed synth+play path and stays green. Swapping the
engine's providers for a non-streaming stub keeps the tests hermetic (and
silent) everywhere.
"""
from __future__ import annotations

import pytest

from jarvis_cli.daemon.main import Daemon


class _NoStreamTTS:
    """Provider stand-in: never streams, warms instantly."""

    name = "stub"
    supports_streaming = False

    async def prewarm(self) -> None:
        return None


@pytest.fixture
def no_stream_tts():
    """Callable that pins a daemon's TTS providers to the non-streaming stub.

    Apply after constructing the Daemon and before `d.run()`, alongside the
    usual `d.tts.synthesize` / `daemon.main.play` patches.
    """

    def _apply(daemon: Daemon) -> None:
        daemon.tts.primary = _NoStreamTTS()  # type: ignore[assignment]
        daemon.tts.overrides = {}

    return _apply
