"""TTS prewarm selection: only providers the config can actually route to.

Warming every configured provider meant the zh override (CosyVoice3, 4.7GB)
was loaded at every daemon start even though `behavior.voice_language = "en"`
never routes to it — 13 hours resident, zero syntheses, on a box already at
97% swap. These tests lock which providers get warmed for each setting, and
that the deferral is announced rather than silent.
"""
from __future__ import annotations

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.tts.engine import TTSEngine


class _FakeProvider:
    """TTS stub that records whether the daemon warmed it."""

    supports_streaming = False

    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.prewarmed = 0
        self._fail = fail

    async def prewarm(self) -> None:
        self.prewarmed += 1
        if self._fail:
            raise RuntimeError("boom")

    async def synthesize(self, text, lang, out_path, voice_id=None, emotion=None):
        return out_path


def _daemon(voice_language: str, overrides: dict | None = None):
    """A Daemon whose TTS engine is fakes — Config() alone would build the real
    XTTS/CosyVoice providers."""
    cfg = Config()
    cfg.behavior.voice_language = voice_language
    d = Daemon(cfg)
    primary = _FakeProvider("primary")
    d.tts = TTSEngine(primary=primary, fallback=None, overrides=overrides or {})
    return d, primary


def _names(providers):
    return [p.name for p in providers]


def test_pinned_en_defers_zh_override():
    zh = _FakeProvider("cosyvoice")
    d, primary = _daemon("en", {"zh": zh})

    warm, deferred = d._tts_prewarm_plan()

    assert _names(warm) == ["primary"]
    assert [p.name for p, _ in deferred] == ["cosyvoice"]
    # The reason has to name both the setting and the language it can't reach —
    # this string is what shows up in the log.
    reason = deferred[0][1]
    assert "voice_language='en'" in reason
    assert "'zh'" in reason


def test_pinned_zh_warms_the_zh_override():
    zh = _FakeProvider("cosyvoice")
    d, _ = _daemon("zh", {"zh": zh})

    warm, deferred = d._tts_prewarm_plan()

    # Primary stays warmed even under a zh pin: pre-baked events default to
    # "en" and the session_start briefing is always English.
    assert _names(warm) == ["primary", "cosyvoice"]
    assert deferred == []


def test_auto_warms_every_override():
    # "auto" resolves per event from the cwd, so any override can be picked.
    zh = _FakeProvider("cosyvoice")
    d, _ = _daemon("auto", {"zh": zh})

    warm, deferred = d._tts_prewarm_plan()

    assert _names(warm) == ["primary", "cosyvoice"]
    assert deferred == []


@pytest.mark.parametrize("voice_language", ["en", "zh", "auto"])
def test_primary_always_prewarmed(voice_language: str):
    d, primary = _daemon(voice_language)

    warm, deferred = d._tts_prewarm_plan()

    assert warm == [primary]
    assert deferred == []


@pytest.mark.asyncio
async def test_prewarm_tts_skips_the_deferred_provider():
    zh = _FakeProvider("cosyvoice")
    d, primary = _daemon("en", {"zh": zh})

    await d._prewarm_tts()

    assert primary.prewarmed == 1
    assert zh.prewarmed == 0


@pytest.mark.asyncio
async def test_prewarm_tts_logs_the_deferral():
    zh = _FakeProvider("cosyvoice")
    d, _ = _daemon("en", {"zh": zh})
    lines: list[str] = []
    from loguru import logger

    sink = logger.add(lambda m: lines.append(m), level="INFO")
    try:
        await d._prewarm_tts()
    finally:
        logger.remove(sink)

    assert any("prewarm deferred" in ln and "cosyvoice" in ln for ln in lines)
    assert any("prewarm ready" in ln and "primary" in ln for ln in lines)


@pytest.mark.asyncio
async def test_prewarm_failure_does_not_stop_the_rest():
    broken = _FakeProvider("primary", fail=True)
    zh = _FakeProvider("cosyvoice")
    d, _ = _daemon("auto", {"zh": zh})
    d.tts = TTSEngine(primary=broken, fallback=None, overrides={"zh": zh})

    await d._prewarm_tts()

    assert broken.prewarmed == 1
    assert zh.prewarmed == 1


def test_plan_reads_the_real_engine_wiring():
    """End-to-end through Daemon.__init__: `tts.provider_zh` builds the
    override, and an "en" pin leaves it cold. `say` stands in for CosyVoice —
    the wiring under test is the same, without the 4.7GB model."""
    cfg = Config()
    cfg.behavior.voice_language = "en"
    cfg.tts.provider_zh = "say"
    d = Daemon(cfg)

    warm, deferred = d._tts_prewarm_plan()

    assert _names(warm) == ["xtts"]
    assert [p.name for p, _ in deferred] == ["say"]
