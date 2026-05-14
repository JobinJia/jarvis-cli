"""Unix-socket listener: accepts NDJSON lines, normalizes into Event."""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import get_args

from loguru import logger

from ..types import Event, NotificationType

_ALLOWED_TYPES: set[str] = set(get_args(NotificationType))


def parse_payload(raw: str) -> Event | None:
    """Parse a single NDJSON line into a normalized Event, or None on bad data."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    ntype = data.get("notification_type")
    if ntype not in _ALLOWED_TYPES:
        return None
    lang = data.get("lang")
    if lang not in (None, "zh", "en"):
        lang = None
    return Event(
        notification_type=ntype,
        tool_name=data.get("tool_name"),
        tool_input=data.get("tool_input") or {},
        cwd=data.get("cwd"),
        session_id=data.get("session_id"),
        raw_message=data.get("message") or data.get("raw_message"),
        received_at=float(data.get("_received_at", time.time())),
        text=data.get("text"),
        lang=lang,
        voice_id=data.get("voice_id"),
    )


async def serve_unix_socket(
    sock_path: Path,
    on_event: Callable[[Event], Awaitable[None]],
) -> None:
    """Run a unix-socket server forever, dispatching parsed events to `on_event`."""
    sock_path = Path(sock_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await reader.read(65536)
            for line in data.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                ev = parse_payload(line)
                if ev is None:
                    logger.warning("Dropped malformed/unknown event: {!r}", line[:120])
                    continue
                await on_event(ev)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    server = await asyncio.start_unix_server(handle, path=str(sock_path))
    os.chmod(sock_path, 0o600)
    logger.info("Listener bound to {}", sock_path)
    try:
        async with server:
            await server.serve_forever()
    finally:
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass
