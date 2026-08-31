import asyncio
import json
import socket
from pathlib import Path

import pytest

from jarvis.daemon.listener import parse_payload, serve_unix_socket
from jarvis.types import Event


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


@pytest.mark.asyncio
async def test_reply_write_survives_vanished_client(tmp_path: Path):
    """A hook that timed out and exited leaves drain() raising
    ConnectionResetError. Regression for the 20 tracebacks asyncio's default
    handler dumped into the daemon log in 13h on 2026-08-25."""
    sock_path = tmp_path / "j.sock"
    unhandled: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, ctx: unhandled.append(ctx)
    )
    events: list[Event] = []
    holder: dict[str, socket.socket] = {}

    async def on_event(ev: Event):
        events.append(ev)

    async def on_query(payload: dict) -> dict:
        # The client gives up while the daemon is still composing the reply.
        holder["sock"].close()
        await asyncio.sleep(0.05)
        # Big enough that drain() actually has to flush to the dead peer.
        return {"context": "x" * 2_000_000}

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_query=on_query)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    holder["sock"] = s
    s.sendall((json.dumps({"command": "skill_query", "text": "hi"}) + "\n").encode())

    await asyncio.sleep(0.3)

    # The listener must still be serving after the failed write.
    s2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s2.connect(str(sock_path))
    s2.sendall(
        (json.dumps({"notification_type": "idle_prompt", "tool_name": None}) + "\n").encode()
    )
    s2.close()
    await asyncio.sleep(0.1)

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert unhandled == []
    assert [ev.notification_type for ev in events] == ["idle_prompt"]


@pytest.mark.asyncio
async def test_serve_unix_socket_reassembles_split_query(tmp_path: Path):
    """A request larger than one segment used to reach the parser truncated and
    be dropped as malformed JSON (2026-08-25 14:16:45, a skill_query carrying a
    <task-notification> block)."""
    sock_path = tmp_path / "j.sock"
    seen: list[dict] = []

    async def on_event(ev: Event): pass

    async def on_query(payload: dict) -> dict:
        seen.append(payload)
        return {"context": "BODY"}

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_query=on_query)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    text = "<task-notification>" + "y" * 200_000 + "</task-notification>"
    raw = (json.dumps({"command": "skill_query", "text": text}) + "\n").encode()
    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    for i in range(0, len(raw), 8192):
        writer.write(raw[i:i + 8192])
        await writer.drain()
        await asyncio.sleep(0)
    reply = json.loads((await reader.readline()).decode())
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert reply == {"context": "BODY"}
    assert len(seen) == 1
    assert seen[0]["text"] == text


@pytest.mark.asyncio
async def test_serve_unix_socket_reassembles_split_event(tmp_path: Path):
    """Same reassembly on the fire-and-forget event path."""
    sock_path = tmp_path / "j.sock"
    received: list[Event] = []

    async def on_event(ev: Event):
        received.append(ev)

    server_task = asyncio.create_task(serve_unix_socket(sock_path, on_event))
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    raw = (
        json.dumps(
            {
                "notification_type": "permission_prompt",
                "tool_name": "Bash",
                "tool_input": {"command": "z" * 100_000},
            }
        )
        + "\n"
    ).encode()
    # asyncio streams, not a blocking socket: a payload this size exceeds the
    # unix-socket buffer, and sendall() from the event-loop thread would wedge
    # against the server reading on that same loop.
    _, writer = await asyncio.open_unix_connection(str(sock_path))
    half = len(raw) // 2
    writer.write(raw[:half])
    await writer.drain()
    await asyncio.sleep(0.05)
    writer.write(raw[half:])
    await writer.drain()
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass

    await asyncio.sleep(0.2)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert len(received) == 1
    assert received[0].tool_name == "Bash"
    assert received[0].tool_input["command"] == "z" * 100_000
