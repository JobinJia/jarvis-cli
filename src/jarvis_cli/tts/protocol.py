"""Framing for the daemon <-> synthesis-worker pipe.

One JSON header line, optionally followed by exactly `size` raw bytes. Audio
travels as those raw bytes rather than inside the JSON: a piece of XTTS output
is ~0.5-2 MB, and base64-ing it through a text channel would cost a third more
bytes and a pointless encode/decode on both sides of every utterance.

Kept in its own module so the daemon-side client and the worker agree on the
wire by construction, and so importing the framing never drags in a provider.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any


async def write_frame(
    writer: asyncio.StreamWriter, header: dict[str, Any], payload: bytes = b"",
) -> None:
    """Emit one frame. `size` is derived here, never passed in, so a header can
    never disagree with the payload that follows it."""
    if payload:
        header = {**header, "size": len(payload)}
    writer.write(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n")
    if payload:
        writer.write(payload)
    await writer.drain()


async def read_frame(
    reader: asyncio.StreamReader,
) -> tuple[dict[str, Any], bytes] | None:
    """Read one frame, or None at EOF (the peer exited).

    A truncated payload is EOF too, not a short read: `readexactly` raising
    IncompleteReadError means the peer died mid-frame, and returning the
    partial bytes would hand the caller half an utterance.
    """
    line = await reader.readline()
    if not line:
        return None
    try:
        header = json.loads(line)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"malformed frame header: {line[:120]!r}") from exc
    size = int(header.get("size", 0))
    if size <= 0:
        return header, b""
    try:
        return header, await reader.readexactly(size)
    except asyncio.IncompleteReadError:
        return None
