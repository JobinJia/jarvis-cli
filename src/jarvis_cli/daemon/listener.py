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
    *,
    on_cancel: Callable[[str], Awaitable[None]] | None = None,
    on_query: Callable[[dict], Awaitable[dict]] | None = None,
) -> None:
    """Run a unix-socket server forever, dispatching parsed rows.

    Event rows (with `notification_type`) go to `on_event`.
    `{"command":"cancel","session_id":"..."}` rows go to `on_cancel`.
    `{"command":"skill_query"|"skill_refresh",...}` rows go to `on_query`,
    whose returned dict is written back as a single JSON line — the only
    request/response path on this socket (everything else is fire-and-forget).
    Rows missing session_id on cancel are dropped silently.
    """
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
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Dropped malformed JSON: {!r}", line[:120])
                    continue
                if isinstance(payload, dict) and payload.get("command") == "cancel":
                    sid = payload.get("session_id")
                    if sid and on_cancel is not None:
                        await on_cancel(sid)
                    continue
                if isinstance(payload, dict) and \
                        payload.get("command") in ("skill_query", "skill_refresh"):
                    if on_query is not None:
                        try:
                            reply = await on_query(payload)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("skill_query handler failed: {}", exc)
                            reply = {}
                        writer.write(
                            (json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8")
                        )
                        await writer.drain()
                    continue
                ev = parse_payload(line)
                if ev is None:
                    logger.warning("Dropped malformed/unknown event: {!r}", line[:120])
                    continue
                logger.debug(
                    "RX event type={} tool={} cwd={} sid={} text={}",
                    ev.notification_type, ev.tool_name, ev.cwd, ev.session_id,
                    "pre-baked" if ev.text else "(via LLM)",
                )
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
