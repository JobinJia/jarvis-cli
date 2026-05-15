"""Audio playback helpers.

`play(path)`            — afplay; reads a finished audio file from disk.
`play_stream(chunks)`   — ffplay; reads MP3 chunks from stdin so playback
                          starts before synthesis completes.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path


async def play(audio: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "afplay", str(audio),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"afplay failed: {err.decode(errors='replace')}")


async def play_stream(chunks: AsyncIterator[bytes]) -> None:
    """Spawn ffplay reading MP3 from stdin and feed it chunks as they arrive.

    ffplay buffers internally and starts playback once it has enough audio
    for a frame, so first sound usually plays within a few hundred ms of
    the first chunk — long before synthesis completes.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffplay",
        "-loglevel", "error",
        "-nodisp",
        "-autoexit",
        "-i", "pipe:0",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None  # PIPE was requested above
    try:
        async for chunk in chunks:
            if not chunk:
                continue
            proc.stdin.write(chunk)
            await proc.stdin.drain()
    finally:
        try:
            if not proc.stdin.is_closing():
                proc.stdin.close()
                await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffplay exited with code {rc}")
