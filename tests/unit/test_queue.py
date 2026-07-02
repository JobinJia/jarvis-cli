import pytest

from jarvis_cli.daemon.queue import BoundedEventQueue
from jarvis_cli.types import Event


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


@pytest.mark.asyncio
async def test_queue_peek_returns_head_without_removing():
    q = BoundedEventQueue(maxsize=10)
    await q.put_or_drop(_ev(1))
    await q.put_or_drop(_ev(2))
    head = q.peek()
    assert head is not None and head.tool_name == "T1"
    assert q.size == 2
    # peek() must hand back the SAME object get() will return — the worker's
    # prefetch cache is keyed on event identity.
    assert (await q.get()) is head


def test_queue_peek_empty_returns_none():
    q = BoundedEventQueue(maxsize=10)
    assert q.peek() is None


@pytest.mark.asyncio
async def test_queue_drop_matching_removes_only_matches():
    q = BoundedEventQueue(maxsize=10)
    for i in range(5):
        await q.put_or_drop(_ev(i))
    removed = q.drop_matching(lambda e: e.tool_name in {"T1", "T3"})
    assert removed == 2
    survivors = []
    while q.size:
        survivors.append((await q.get()).tool_name)
    assert survivors == ["T0", "T2", "T4"]


@pytest.mark.asyncio
async def test_queue_drop_matching_returns_zero_when_no_match():
    q = BoundedEventQueue(maxsize=10)
    await q.put_or_drop(_ev(0))
    assert q.drop_matching(lambda e: e.tool_name == "nope") == 0
    assert q.size == 1
