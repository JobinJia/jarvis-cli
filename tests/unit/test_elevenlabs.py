from pathlib import Path

import pytest
import respx

from jarvis.config import ElevenLabsConfig
from jarvis.tts.providers.elevenlabs import ElevenLabsProvider


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


@pytest.mark.asyncio
async def test_elevenlabs_supports_streaming_flag():
    cfg = ElevenLabsConfig(voice_id="vid")
    assert ElevenLabsProvider(cfg).supports_streaming is True


@pytest.mark.asyncio
async def test_elevenlabs_stream_yields_chunks(monkeypatch: pytest.MonkeyPatch):
    """The streaming endpoint must yield bytes incrementally so the player
    can start playback before the full audio arrives."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="vid")
    fake_bytes = b"MP3CHUNK1" + b"MP3CHUNK2" + b"MP3CHUNK3"
    with respx.mock(base_url="https://api.elevenlabs.io") as router:
        router.post("/v1/text-to-speech/vid/stream").respond(200, content=fake_bytes)
        chunks: list[bytes] = []
        async for chunk in ElevenLabsProvider(cfg).stream("hi", lang="en"):
            chunks.append(chunk)
    assert b"".join(chunks) == fake_bytes


@pytest.mark.asyncio
async def test_elevenlabs_stream_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    cfg = ElevenLabsConfig(voice_id="vid")
    with pytest.raises(RuntimeError):
        async for _ in ElevenLabsProvider(cfg).stream("hi", lang="en"):
            pass


_QUOTA_BODY = {
    "detail": {
        "type": "invalid_request",
        "code": "quota_exceeded",
        "message": "This request exceeds your quota of 10000. You have 3 credits remaining, while 165 credits are required for this request.",
        "status": "quota_exceeded",
    }
}


@pytest.mark.asyncio
async def test_elevenlabs_synth_translates_quota_401_to_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """ElevenLabs returns HTTP 401 with code=quota_exceeded when credits run
    out; the long raw HTTPStatusError message is unhelpful in logs. The
    provider must surface a single-line RuntimeError so the engine's fallback
    log reads 'TTS provider elevenlabs failed: ElevenLabs quota exhausted...'.
    """
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="vid")
    with respx.mock(base_url="https://api.elevenlabs.io") as router:
        router.post("/v1/text-to-speech/vid").respond(401, json=_QUOTA_BODY)
        with pytest.raises(RuntimeError, match="quota exhausted"):
            await ElevenLabsProvider(cfg).synthesize(
                "hi", lang="en", out_path=tmp_path / "o.mp3"
            )


@pytest.mark.asyncio
async def test_elevenlabs_stream_translates_quota_401_to_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="vid")
    with respx.mock(base_url="https://api.elevenlabs.io") as router:
        router.post("/v1/text-to-speech/vid/stream").respond(401, json=_QUOTA_BODY)
        with pytest.raises(RuntimeError, match="quota exhausted"):
            async for _ in ElevenLabsProvider(cfg).stream("hi", lang="en"):
                pass


@pytest.mark.asyncio
async def test_elevenlabs_synth_passes_through_non_quota_401(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Only quota_exceeded 401s get the special translation. Auth / scope
    401s should still surface so the engine logs the real reason."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="vid")
    other = {"detail": {"status": "invalid_api_key", "message": "bad key"}}
    with respx.mock(base_url="https://api.elevenlabs.io") as router:
        router.post("/v1/text-to-speech/vid").respond(401, json=other)
        with pytest.raises(Exception) as exc_info:
            await ElevenLabsProvider(cfg).synthesize(
                "hi", lang="en", out_path=tmp_path / "o.mp3"
            )
    # Must NOT be mis-reported as quota exhausted.
    assert "quota exhausted" not in str(exc_info.value)
