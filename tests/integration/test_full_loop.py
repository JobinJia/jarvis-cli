"""Smoke test: hook_client → real daemon (mocked LLM + TTS) → mocked player."""
import asyncio
import io
import json
import os
from pathlib import Path

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.hook_client import forward_event


@pytest.mark.asyncio
async def test_hook_to_daemon_smoke(tmp_path: Path, monkeypatch, no_stream_tts):
    cfg = Config()
    cfg.paths.socket = str(tmp_path / "j.sock")

    phrased: list[str] = []
    played: list[Path] = []

    async def fake_phrase(event, lang, emotion=None):
        s = f"Sir, {event.notification_type}"
        phrased.append(s)
        return s

    async def fake_synth(text, lang, out_path, voice_id=None, emotion=None):
        out_path.write_bytes(b"WAV")
        return out_path

    async def fake_play(audio, *, on_spawn=None):
        played.append(audio)

    d = Daemon(cfg, health_port=0)
    d.router.phrase = fake_phrase  # type: ignore[assignment]
    d.tts.synthesize = fake_synth  # type: ignore[assignment]
    no_stream_tts(d)
    monkeypatch.setattr("jarvis_cli.daemon.main.play", fake_play)

    daemon_task = asyncio.create_task(d.run())
    for _ in range(100):
        if Path(cfg.paths.socket).exists():
            break
        await asyncio.sleep(0.02)
    assert Path(cfg.paths.socket).exists()

    # Use the real hook_client end (runs sync IO, so do it in a thread)
    payload = json.dumps(
        {
            "notification_type": "permission_prompt",
            "tool_name": "Bash",
            "cwd": str(tmp_path),
        }
    )
    ok = await asyncio.to_thread(forward_event, io.StringIO(payload), cfg.paths.socket)
    assert ok is True

    for _ in range(200):
        if played:
            break
        await asyncio.sleep(0.02)

    daemon_task.cancel()
    try:
        await daemon_task
    except asyncio.CancelledError:
        pass

    assert phrased and "permission_prompt" in phrased[0]
    assert played
