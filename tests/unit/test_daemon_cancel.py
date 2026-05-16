"""Daemon.cancel_session: kill current proc + drop same-sid queued events."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis_cc.config import Config
from jarvis_cc.daemon.main import Daemon
from jarvis_cc.types import Event


def _ev(sid: str | None, tool: str = "T") -> Event:
    return Event(
        notification_type="permission_prompt",
        tool_name=tool,
        cwd=f"/{sid}",
        session_id=sid,
    )


@pytest.mark.asyncio
async def test_cancel_session_kills_current_proc_for_matching_sid():
    d = Daemon(Config())
    proc = MagicMock()
    proc.kill = MagicMock()
    d._current_proc = proc
    d._current_session_id = "abc"

    await d.cancel_session("abc")

    proc.kill.assert_called_once()
    assert "abc" in d._cancelled_sessions


@pytest.mark.asyncio
async def test_cancel_session_does_not_kill_proc_for_other_sid():
    d = Daemon(Config())
    proc = MagicMock()
    proc.kill = MagicMock()
    d._current_proc = proc
    d._current_session_id = "abc"

    await d.cancel_session("xyz")

    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_session_drops_matching_queued_events():
    d = Daemon(Config())
    await d.queue.put_or_drop(_ev("abc", tool="T1"))
    await d.queue.put_or_drop(_ev("xyz", tool="T2"))
    await d.queue.put_or_drop(_ev("abc", tool="T3"))

    await d.cancel_session("abc")

    survivors = []
    while d.queue.size:
        survivors.append((await d.queue.get()).tool_name)
    assert survivors == ["T2"]


@pytest.mark.asyncio
async def test_cancel_session_handles_process_lookup_error():
    d = Daemon(Config())

    proc = MagicMock()
    proc.kill = MagicMock(side_effect=ProcessLookupError())
    d._current_proc = proc
    d._current_session_id = "abc"

    # Should not raise
    await d.cancel_session("abc")
