"""Stale-event drop (`_is_stale` + `_worker`) and briefing prewarm.

Under a backlog burst (long playback blocking the queue) the worker used to
speak every dequeued event, including a permission_prompt the user already
acted on 30+ seconds ago. `behavior.stale_event_max_age_seconds` now drops
LLM-phrased events at dequeue time; pre-baked text and session_start
briefings are exempt. `_prewarm_briefing` warms the weather cache at daemon
start so the first briefing doesn't stall on the weather API.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, patch

import pytest

from jarvis.config import Config
from jarvis.daemon.main import Daemon
from jarvis.types import Event


def _llm_event(
    received_at: float, *, ntype: str = "permission_prompt",
    text: str | None = None,
) -> Event:
    return Event(
        notification_type=ntype,
        tool_name="Bash",
        cwd="/repo",
        session_id="s1",
        received_at=received_at,
        text=text,
    )


# --- _is_stale predicate -----------------------------------------------------


def test_is_stale_fresh_event_is_not_stale():
    d = Daemon(Config())
    assert d._is_stale(_llm_event(time.time())) is False


def test_is_stale_old_llm_event_is_stale():
    """120s past a 60s floor: an LLM-phrased notification this old describes a
    prompt the user already answered — drop, don't speak."""
    d = Daemon(Config())
    assert d._is_stale(_llm_event(time.time() - 120)) is True


def test_is_stale_prebaked_text_is_exempt():
    """`say --text` lines were composed by the caller and should speak
    whenever they surface, however delayed."""
    d = Daemon(Config())
    old = _llm_event(time.time() - 120, text="pre-baked line")
    assert d._is_stale(old) is False


def test_is_stale_session_start_is_exempt():
    """The briefing has its own throttle gates; a delayed one is still worth
    hearing (greeting + time + weather stay accurate at compose time)."""
    d = Daemon(Config())
    old = Event(
        notification_type="session_start",
        tool_name=None,
        cwd="/repo",
        session_id="s1",
        received_at=time.time() - 120,
    )
    assert d._is_stale(old) is False


def test_is_stale_zero_received_at_is_exempt():
    """Synthetic/test events carry no timestamp (received_at=0.0) — never
    treat 'unstamped' as 'ancient'."""
    d = Daemon(Config())
    assert d._is_stale(_llm_event(0.0)) is False


def test_is_stale_config_zero_disables():
    cfg = Config()
    cfg.behavior.stale_event_max_age_seconds = 0
    d = Daemon(cfg)
    assert d._is_stale(_llm_event(time.time() - 3600)) is False


# --- worker-level drop -------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_drops_stale_event_and_plays_fresh_one():
    """Drive the real `_worker` loop: enqueue an old event then a fresh one
    and assert only the fresh one reaches processing — the drop decision
    lives in production code, not in the test."""
    d = Daemon(Config())

    processed: list[Event] = []
    done = asyncio.Event()

    async def _record(event: Event) -> None:
        processed.append(event)
        done.set()

    stale = _llm_event(time.time() - 120)
    fresh = _llm_event(time.time())

    with patch.object(d, "_process_one", side_effect=_record):
        worker = asyncio.create_task(d._worker())
        await d.queue.put_or_drop(stale)
        await d.queue.put_or_drop(fresh)
        await asyncio.wait_for(done.wait(), timeout=2)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    assert processed == [fresh]


# --- briefing prewarm --------------------------------------------------------


@pytest.mark.asyncio
async def test_prewarm_briefing_fetches_configured_city():
    cfg = Config()
    cfg.behavior.session_briefing.city = "Gotham"
    d = Daemon(cfg)

    with patch.object(d._weather_cache, "get", new=AsyncMock()) as get:
        await d._prewarm_briefing()

    get.assert_awaited_once_with(
        "Gotham", cfg.behavior.session_briefing.weather_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_prewarm_briefing_falls_back_to_detected_city():
    """Empty `city` derives the location the same way compose_briefing does,
    so the prewarmed cache entry is the one the briefing will hit."""
    d = Daemon(Config())

    with patch.object(d._weather_cache, "get", new=AsyncMock()) as get, \
            patch(
                "jarvis.daemon.main.detect_city", return_value="Metropolis",
            ):
        await d._prewarm_briefing()

    get.assert_awaited_once_with(
        "Metropolis", d.cfg.behavior.session_briefing.weather_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_prewarm_briefing_disabled_skips_fetch():
    cfg = Config()
    cfg.behavior.session_briefing.enabled = False
    d = Daemon(cfg)

    with patch.object(d._weather_cache, "get", new=AsyncMock()) as get:
        await d._prewarm_briefing()

    get.assert_not_awaited()
