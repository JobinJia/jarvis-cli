"""The daemon honours the mute state file, and says so in /health."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis import mute
from jarvis.config import Config
from jarvis.daemon.main import Daemon
from jarvis.types import Event


def _daemon(tmp_path) -> Daemon:
    cfg = Config()
    cfg.paths.mute_state = str(tmp_path / "mute.json")
    cfg.behavior.events = ["permission_prompt", "task_complete"]
    return Daemon(cfg)


def _ev(ntype: str = "permission_prompt", text: str | None = None) -> Event:
    return Event(
        notification_type=ntype, tool_name="T", cwd="/w", session_id="s",
        text=text,
    )


@pytest.mark.asyncio
async def test_events_queue_normally_when_nothing_is_muted(tmp_path):
    d = _daemon(tmp_path)
    await d._on_event(_ev())
    assert d.queue.size == 1


@pytest.mark.asyncio
async def test_a_global_mute_drops_every_event(tmp_path):
    d = _daemon(tmp_path)
    mute.set_global(d._mute_path, 600)
    await d._on_event(_ev("permission_prompt"))
    await d._on_event(_ev("task_complete"))
    assert d.queue.size == 0


@pytest.mark.asyncio
async def test_a_mute_set_after_startup_takes_effect_with_no_reload(tmp_path):
    # The daemon re-reads the file per event, which is what lets the CLI mute
    # a running daemon (and a daemon that was down when the mute was set).
    d = _daemon(tmp_path)
    await d._on_event(_ev())
    mute.set_global(d._mute_path, 600)
    await d._on_event(_ev())
    assert d.queue.size == 1


@pytest.mark.asyncio
async def test_a_type_mute_drops_only_that_type(tmp_path):
    d = _daemon(tmp_path)
    mute.set_type(d._mute_path, "task_complete", 600)
    await d._on_event(_ev("task_complete"))
    await d._on_event(_ev("permission_prompt"))
    assert d.queue.size == 1
    assert (await d.queue.get()).notification_type == "permission_prompt"


@pytest.mark.asyncio
async def test_mute_outranks_pre_baked_text(tmp_path):
    # `say --text` skips the LLM but not the mute: the user asked for silence.
    d = _daemon(tmp_path)
    mute.set_global(d._mute_path, 600)
    await d._on_event(_ev(text="Sir, the kettle has boiled."))
    assert d.queue.size == 0


@pytest.mark.asyncio
async def test_a_lapsed_mute_lets_events_through_again(tmp_path):
    # Hand-written because save() prunes a lapsed expiry on the way out; the
    # daemon must also ignore one it finds on the way in.
    (tmp_path / "mute.json").write_text('{"global": 1.0}')
    d = _daemon(tmp_path)
    await d._on_event(_ev())
    assert d.queue.size == 1


@pytest.mark.asyncio
async def test_muting_cuts_the_line_already_in_flight(tmp_path):
    d = _daemon(tmp_path)
    proc = MagicMock()
    d._current_proc = proc
    await d.queue.put_or_drop(_ev())
    mute.set_global(d._mute_path, 600)

    reply = d._apply_mute()

    proc.kill.assert_called_once()
    assert reply["dropped"] == 1
    assert d.queue.size == 0


def test_a_type_mute_does_not_cut_the_line_in_flight(tmp_path):
    # Only a blanket mute means "stop talking now"; silencing one event type
    # is about what comes next.
    d = _daemon(tmp_path)
    proc = MagicMock()
    d._current_proc = proc
    mute.set_type(d._mute_path, "task_complete", 600)

    reply = d._apply_mute()

    proc.kill.assert_not_called()
    assert list(reply["muted"]) == ["task_complete"]
    assert "dropped" not in reply


def test_health_reports_what_is_muted(tmp_path):
    d = _daemon(tmp_path)
    assert d._snapshot()["muted"] == {}
    mute.set_global(d._mute_path, 600)
    assert "all" in d._snapshot()["muted"]
