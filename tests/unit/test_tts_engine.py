from pathlib import Path
from typing import Literal

import pytest

from jarvis_cc.tts.engine import TTSEngine
from jarvis_cc.tts.providers.base import TTSProvider


class _StubTTS(TTSProvider):
    def __init__(self, name: str, mode: Literal["ok", "fail"]):
        self.name = name
        self.mode = mode
        self.calls = 0

    async def synthesize(self, text, lang, out_path, voice_id=None):
        self.calls += 1
        self.last_voice_id = voice_id
        if self.mode == "fail":
            raise RuntimeError(f"{self.name} down")
        out_path.write_bytes(b"AUDIO-" + self.name.encode())
        return out_path


@pytest.mark.asyncio
async def test_engine_uses_primary_when_ok(tmp_path: Path):
    primary, fallback = _StubTTS("p", "ok"), _StubTTS("f", "ok")
    eng = TTSEngine(primary, fallback)
    out = await eng.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")
    assert out.read_bytes() == b"AUDIO-p"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_engine_falls_back_to_secondary(tmp_path: Path):
    primary, fallback = _StubTTS("p", "fail"), _StubTTS("f", "ok")
    eng = TTSEngine(primary, fallback)
    out = await eng.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")
    assert out.read_bytes() == b"AUDIO-f"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_engine_raises_when_all_fail(tmp_path: Path):
    primary, fallback = _StubTTS("p", "fail"), _StubTTS("f", "fail")
    eng = TTSEngine(primary, fallback)
    with pytest.raises(RuntimeError):
        await eng.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")
