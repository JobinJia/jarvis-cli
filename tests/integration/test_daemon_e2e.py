"""End-to-end: socket → dedup → queue → router → engine → player.

Stubs the LLM router, TTS engine, and player so we exercise wiring only.
"""
import asyncio
import json
import socket
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.types import Event


@pytest.mark.asyncio
async def test_daemon_handles_event_end_to_end(tmp_path: Path, monkeypatch):
    cfg = Config()
    cfg.paths.socket = str(tmp_path / "j.sock")
    cfg.paths.log = str(tmp_path / "d.log")
    cfg.paths.missed_log = str(tmp_path / "m.log")

    phrased: list[str] = []

    async def fake_phrase(event: Event, lang, emotion=None):
        s = f"Sir, {event.tool_name}!"
        phrased.append(s)
        return s

    played: list[Path] = []

    async def fake_play(audio: Path, *, on_spawn=None) -> None:
        played.append(audio)

    async def fake_synth(text: str, lang, out_path: Path, voice_id=None, emotion=None):
        out_path.write_bytes(b"FAKE")
        return out_path

    d = Daemon(cfg, health_port=0)
    d.router.phrase = fake_phrase  # type: ignore[assignment]
    d.tts.synthesize = fake_synth  # type: ignore[assignment]
    monkeypatch.setattr("jarvis_cli.daemon.main.play", fake_play)

    task = asyncio.create_task(d.run())
    # wait for socket
    for _ in range(100):
        if Path(cfg.paths.socket).exists():
            break
        await asyncio.sleep(0.02)

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(cfg.paths.socket)
    s.sendall(
        (
            json.dumps(
                {
                    "notification_type": "permission_prompt",
                    "tool_name": "Bash",
                    "cwd": str(tmp_path),
                }
            )
            + "\n"
        ).encode()
    )
    s.close()

    # give the daemon time to process
    for _ in range(200):
        if played:
            break
        await asyncio.sleep(0.02)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert phrased == ["Sir, Bash!"]
    assert len(played) == 1


@pytest.mark.asyncio
async def test_daemon_dedups_within_window(tmp_path: Path, monkeypatch):
    cfg = Config()
    cfg.behavior.dedup_window_seconds = 10
    cfg.paths.socket = str(tmp_path / "j.sock")

    phrased: list[str] = []

    async def fake_phrase(event, lang, emotion=None):
        phrased.append(event.tool_name or "?")
        return "ok"

    async def fake_synth(text, lang, out_path, voice_id=None, emotion=None):
        out_path.write_bytes(b"x")
        return out_path

    async def fake_play(audio, *, on_spawn=None):
        pass

    d = Daemon(cfg, health_port=0)
    d.router.phrase = fake_phrase  # type: ignore[assignment]
    d.tts.synthesize = fake_synth  # type: ignore[assignment]
    monkeypatch.setattr("jarvis_cli.daemon.main.play", fake_play)

    task = asyncio.create_task(d.run())
    for _ in range(100):
        if Path(cfg.paths.socket).exists():
            break
        await asyncio.sleep(0.02)

    payload = {
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "cwd": str(tmp_path),
    }
    for _ in range(3):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(cfg.paths.socket)
        s.sendall((json.dumps(payload) + "\n").encode())
        s.close()
        await asyncio.sleep(0.05)

    # wait for processing
    await asyncio.sleep(0.4)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert phrased == ["Bash"], f"expected dedup to 1 call, got {phrased}"
