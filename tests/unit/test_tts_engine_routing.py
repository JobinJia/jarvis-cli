"""Per-language TTS routing: zh override picks the native speaker."""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_cli.tts.engine import TTSEngine


class _Fake:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def synthesize(self, text, lang, out_path, voice_id=None, emotion=None):
        self.calls.append((text, lang))
        return out_path


def test_primary_for_prefers_override():
    en, zh = _Fake("xtts"), _Fake("piper")
    eng = TTSEngine(primary=en, fallback=None, overrides={"zh": zh})
    assert eng.primary_for("zh") is zh
    assert eng.primary_for("en") is en


def test_primary_for_without_override_is_global_primary():
    en = _Fake("xtts")
    eng = TTSEngine(primary=en, fallback=None)
    assert eng.primary_for("zh") is en


@pytest.mark.asyncio
async def test_synthesize_routes_zh_to_override():
    en, zh = _Fake("xtts"), _Fake("piper")
    eng = TTSEngine(primary=en, fallback=None, overrides={"zh": zh})

    await eng.synthesize("你好", "zh", Path("/tmp/x.wav"))
    await eng.synthesize("hello", "en", Path("/tmp/y.wav"))

    assert zh.calls == [("你好", "zh")]
    assert en.calls == [("hello", "en")]


@pytest.mark.asyncio
async def test_synthesize_zh_override_failure_falls_back():
    class _Broken(_Fake):
        async def synthesize(self, *a, **k):
            raise RuntimeError("boom")

    fb = _Fake("say")
    eng = TTSEngine(primary=_Fake("xtts"), fallback=fb,
                    overrides={"zh": _Broken("piper")})

    await eng.synthesize("你好", "zh", Path("/tmp/x.wav"))

    assert fb.calls == [("你好", "zh")]
