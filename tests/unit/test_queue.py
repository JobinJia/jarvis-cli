import asyncio

import pytest

from jarvis_cc.daemon.queue import BoundedEventQueue
from jarvis_cc.types import Event


def _ev(i: int) -> Event:
    return Event(notification_type="idle_prompt", tool_name=f"T{i}", cwd=f"/{i}")


@pytest.mark.asyncio
async def test_queue_preserves_order():
    q = BoundedEventQueue(maxsize=10)
    await q.put_or_drop(_ev(1))
    await q.put_or_drop(_ev(2))
    a = await q.get()
    b = await q.get()
    assert a.tool_name == "T1"
    assert b.tool_name == "T2"


@pytest.mark.asyncio
async def test_queue_drops_oldest_when_full():
    q = BoundedEventQueue(maxsize=2)
    for i in range(5):
        await q.put_or_drop(_ev(i))
    # Only last 2 survive
    a = await q.get()
    b = await q.get()
    assert a.tool_name == "T3"
    assert b.tool_name == "T4"
    assert q.dropped_count == 3
