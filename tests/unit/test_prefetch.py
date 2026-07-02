"""Phrase prefetch: overlap LLM phrasing of event N+1 with playback of N.

Under a backlog the worker used to be fully serial — phrase(N) → play(N) →
phrase(N+1) — so every event paid a full LLM round-trip that could have run
during the previous event's playback. `_prefetch_next` now phrases the queue
head while the current event plays and stores it in `_prefetched`, keyed by
event IDENTITY (the queue hands back the same object). These tests lock the
cache hit, the identity-miss discard (cancel path), and the skip conditions
(pre-baked / session_start / stale / empty queue).
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import patch

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.types import Event


def _llm_event(
    i: int = 0, *, sid: str = "s1", ntype: str = "permission_prompt",
    text: str | None = None, received_at: float = 0.0,
) -> Event:
    return Event(
        notification_type=ntype,
        tool_name=f"T{i}",
        cwd="/repo",
        session_id=sid,
        received_at=received_at,
        text=text,
    )


def _daemon_with_fake_phrase() -> tuple[Daemon, list[Event]]:
    """Daemon whose router.phrase is an async fake recording its events."""
    d = Daemon(Config())
    phrase_calls: list[Event] = []

    async def _phrase(event: Event, *, lang, emotion=None) -> str:
        phrase_calls.append(event)
        return "prefetched line"

    d.router.phrase = _phrase
    return d, phrase_calls


async def _drive_worker(d: Daemon, n_events: int) -> list[Event]:
    """Run the real `_worker` with `_process_one` patched to record, until
    `n_events` events were dispatched. Events must already be queued."""
    processed: list[Event] = []
    done = asyncio.Event()

    async def _record(event: Event) -> None:
        processed.append(event)
        if len(processed) == n_events:
            done.set()

    with patch.object(d, "_process_one", side_effect=_record):
        worker = asyncio.create_task(d._worker())
        await asyncio.wait_for(done.wait(), timeout=2)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
    return processed


# --- cache hit ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_hands_second_event_pre_phrased_text():
    """Two LLM events queued: the first is phrased in-band (text None at
    dispatch), but while it 'plays', prefetch phrases the second — which must
    arrive at _process_one with text already set. router.phrase runs exactly
    once (the prefetch); the pre-baked branch never re-phrases."""
    d, phrase_calls = _daemon_with_fake_phrase()
    e1, e2 = _llm_event(1), _llm_event(2)
    await d.queue.put_or_drop(e1)
    await d.queue.put_or_drop(e2)

    processed = await _drive_worker(d, 2)

    assert processed[0].text is None
    assert processed[1].text == "prefetched line"
    assert processed[1].tool_name == "T2"
    assert phrase_calls == [e2]


# --- identity miss (cancel) --------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_identity_miss_discards_cache():
    """A prefetched entry whose event got dropped (cancel_session's
    drop_matching) must never leak its text onto a different event — the
    identity check fails and the cache is cleared."""
    d, phrase_calls = _daemon_with_fake_phrase()
    doomed = _llm_event(1, sid="dead")
    await d.queue.put_or_drop(doomed)
    await d._prefetch_next()
    assert d._prefetched is not None and d._prefetched[0] is doomed

    # Cancel path: the cached event leaves the queue; a different one arrives.
    d.queue.drop_matching(lambda e: e.session_id == "dead")
    replacement = _llm_event(2, sid="alive")
    await d.queue.put_or_drop(replacement)

    processed = await _drive_worker(d, 1)

    assert processed[0].text is None  # no cached text leaked
    assert processed[0] is replacement
    assert d._prefetched is None  # miss still clears the cache
    assert phrase_calls == [doomed]  # only the original prefetch ran


# --- skip conditions ---------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_skips_empty_queue():
    d, phrase_calls = _daemon_with_fake_phrase()
    await d._prefetch_next()
    assert d._prefetched is None
    assert phrase_calls == []


@pytest.mark.asyncio
async def test_prefetch_skips_prebaked_text():
    """Pre-baked text has no LLM round-trip to hide."""
    d, phrase_calls = _daemon_with_fake_phrase()
    await d.queue.put_or_drop(_llm_event(1, text="already baked"))
    await d._prefetch_next()
    assert d._prefetched is None
    assert phrase_calls == []


@pytest.mark.asyncio
async def test_prefetch_skips_session_start():
    """The briefing composes inline in _process_one, not via router.phrase."""
    d, phrase_calls = _daemon_with_fake_phrase()
    await d.queue.put_or_drop(
        Event(
            notification_type="session_start",
            tool_name=None,
            cwd="/repo",
            session_id="s1",
        )
    )
    await d._prefetch_next()
    assert d._prefetched is None
    assert phrase_calls == []


@pytest.mark.asyncio
async def test_prefetch_skips_stale_head():
    """A stale head is about to be dropped at dequeue — phrasing it would
    waste the round-trip AND resurrect it via the pre-baked stale exemption."""
    d, phrase_calls = _daemon_with_fake_phrase()
    await d.queue.put_or_drop(_llm_event(1, received_at=time.time() - 120))
    await d._prefetch_next()
    assert d._prefetched is None
    assert phrase_calls == []


@pytest.mark.asyncio
async def test_prefetch_failure_is_best_effort():
    """A phrase failure during prefetch must not crash the worker — it just
    means the event is phrased in-band on dequeue, as before."""
    d = Daemon(Config())

    async def _boom(event: Event, *, lang, emotion=None) -> str:
        raise RuntimeError("ollama down")

    d.router.phrase = _boom
    await d.queue.put_or_drop(_llm_event(1))
    await d._prefetch_next()  # must not raise
    assert d._prefetched is None
