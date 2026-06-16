import asyncio
import json
import socket
from pathlib import Path

import pytest

from jarvis_cli.daemon.listener import parse_payload, serve_unix_socket
from jarvis_cli.types import Event


def test_parse_payload_maps_known_fields():
    raw = json.dumps(
        {
            "notification_type": "permission_prompt",
            "tool_name": "Bash",
            "tool_input": {"command": "rm foo"},
            "cwd": "/x/y",
            "session_id": "s1",
            "_received_at": 12345.67,
        }
    )
    ev = parse_payload(raw)
    assert isinstance(ev, Event)
    assert ev.notification_type == "permission_prompt"
    assert ev.tool_name == "Bash"
    assert ev.tool_input == {"command": "rm foo"}
    assert ev.cwd == "/x/y"
    assert ev.received_at == 12345.67


def test_parse_payload_returns_none_for_unknown_type():
    raw = json.dumps({"notification_type": "made_up_event", "tool_name": "X"})
    assert parse_payload(raw) is None


def test_parse_payload_returns_none_for_malformed_json():
    assert parse_payload("not json") is None


@pytest.mark.asyncio
async def test_serve_unix_socket_yields_events(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received: list[Event] = []

    async def collect(ev: Event):
        received.append(ev)

    server_task = asyncio.create_task(serve_unix_socket(sock_path, collect))
    # Wait for socket to exist
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall(
        (json.dumps({"notification_type": "idle_prompt", "tool_name": None}) + "\n").encode()
    )
    s.close()

    await asyncio.sleep(0.1)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert len(received) == 1
    assert received[0].notification_type == "idle_prompt"


@pytest.mark.asyncio
async def test_serve_unix_socket_routes_cancel_command(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    events: list[Event] = []
    cancels: list[str] = []

    async def on_event(ev: Event):
        events.append(ev)

    async def on_cancel(sid: str):
        cancels.append(sid)

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_cancel=on_cancel)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall(
        (json.dumps({"command": "cancel", "session_id": "abc"}) + "\n").encode()
    )
    s.close()

    await asyncio.sleep(0.1)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert cancels == ["abc"]
    assert events == []


@pytest.mark.asyncio
async def test_serve_unix_socket_replies_to_skill_query(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    seen: list[dict] = []

    async def on_event(ev: Event): pass

    async def on_query(payload: dict) -> dict:
        seen.append(payload)
        return {"context": "BODY", "mode": "body"}

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_query=on_query)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    def _roundtrip() -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(str(sock_path))
        s.sendall(
            (json.dumps({"command": "skill_query", "text": "hi"}) + "\n").encode()
        )
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        return json.loads(buf.split(b"\n", 1)[0].decode())

    reply = await asyncio.to_thread(_roundtrip)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert reply == {"context": "BODY", "mode": "body"}
    assert seen and seen[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_serve_unix_socket_ignores_cancel_without_session_id(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    cancels: list[str] = []

    async def on_event(ev: Event): pass
    async def on_cancel(sid: str): cancels.append(sid)

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_cancel=on_cancel)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall((json.dumps({"command": "cancel"}) + "\n").encode())
    s.close()
    await asyncio.sleep(0.1)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert cancels == []
