from pathlib import Path

import pytest
import respx

from jarvis_cc.config import ElevenLabsConfig
from jarvis_cc.tts.providers.elevenlabs import ElevenLabsProvider


@pytest.mark.asyncio
async def test_elevenlabs_writes_audio_to_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="vid")
    fake_bytes = b"BYTES-OF-MP3"
    with respx.mock(base_url="https://api.elevenlabs.io") as router:
        router.post("/v1/text-to-speech/vid").respond(200, content=fake_bytes)
        out = tmp_path / "out.mp3"
        result = await ElevenLabsProvider(cfg).synthesize("hi", lang="en", out_path=out)
    assert result == out
    assert out.read_bytes() == fake_bytes


@pytest.mark.asyncio
async def test_elevenlabs_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    cfg = ElevenLabsConfig(voice_id="vid")
    with pytest.raises(RuntimeError):
        await ElevenLabsProvider(cfg).synthesize(
            "hi", lang="en", out_path=tmp_path / "o.mp3"
        )


@pytest.mark.asyncio
async def test_elevenlabs_raises_when_voice_id_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="")
    with pytest.raises(RuntimeError):
        await ElevenLabsProvider(cfg).synthesize(
            "hi", lang="en", out_path=tmp_path / "o.mp3"
        )
