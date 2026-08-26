"""PortAudio wedge self-heal: the daemon restarts itself instead of staying
degraded until a human notices.

A wedge (leaked release thread parked on CoreAudio's HAL mutex) cannot be
cleared from inside the process. Before 2026-08-25 the only signal was one
WARNING line, and the daemon ran degraded for 75 minutes until the user
reported the audio by ear. launchd's KeepAlive gives us a free respawn — these
tests lock when we take it and, more importantly, when we must not.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.types import Event


def _event(sid: str = "s1") -> Event:
    return Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        cwd="/repo",
        session_id=sid,
    )


@pytest.mark.asyncio
async def test_no_restart_without_a_wedge():
    d = Daemon(Config())
    with patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()
    hard_exit.assert_not_called()


@pytest.mark.asyncio
async def test_wedge_on_a_long_lived_daemon_restarts_it():
    d = Daemon(Config())
    d._note_pcm_wedge("device release stuck past 5.0s")
    d._started_at -= 3600.0  # an hour of healthy playback before the wedge

    with patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()

    hard_exit.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_wedge_early_in_the_process_does_not_restart():
    """The loop guard. A device that wedges on every open would respawn
    forever under KeepAlive; staying up degraded still speaks, via
    synth+afplay."""
    d = Daemon(Config())
    d._note_pcm_wedge("device open stuck past 5.0s")  # seconds into the process

    with patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()

    hard_exit.assert_not_called()


@pytest.mark.asyncio
async def test_restart_waits_for_the_queue_to_drain():
    """Exiting with work queued would drop those events on the floor, and
    exiting mid-utterance would cut it. The check re-arms until idle."""
    d = Daemon(Config())
    d._note_pcm_wedge("device release stuck past 5.0s")
    d._started_at -= 3600.0
    await d.queue.put_or_drop(_event())

    with patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()
        hard_exit.assert_not_called()
        assert d._self_heal_checked is False  # still pending, not ruled out

        await d.queue.get()
        await d._self_heal_if_wedged()

    hard_exit.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_declined_restart_is_decided_once():
    """Below the uptime floor we settle for degraded — and must not re-litigate
    it on every subsequent event, which would spam the log for hours."""
    d = Daemon(Config())
    d._note_pcm_wedge("device open stuck past 5.0s")

    with patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()
        assert d._self_heal_checked is True
        d._started_at -= 3600.0  # crossing the floor later must not revive it
        await d._self_heal_if_wedged()

    hard_exit.assert_not_called()


@pytest.mark.asyncio
async def test_first_wedge_reason_is_the_one_reported():
    """Later wedges are echoes of the same stuck device; the first reason is
    the diagnostic one."""
    d = Daemon(Config())
    d._note_pcm_wedge("device release stuck past 5.0s")
    d._note_pcm_wedge("device open stuck past 5.0s")
    assert d._wedge_reason == "device release stuck past 5.0s"


def test_daemon_registers_itself_for_wedge_notifications():
    """Without this wiring the self-heal never fires at all."""
    import jarvis_cli.player as player_mod

    with patch.object(player_mod, "_wedge_callback", None):
        d = Daemon(Config())
        assert player_mod._wedge_callback == d._note_pcm_wedge


@pytest.mark.asyncio
async def test_restart_drains_in_flight_pushes_first():
    """An empty speech queue does not mean nothing is in flight: webhook and
    ntfy pushes are detached tasks that outlive playback on purpose, and the
    actionable ones carry the Approve/Deny buttons. os._exit runs no atexit
    hooks and cancels nothing, so a POST still on the wire would vanish."""
    d = Daemon(Config())
    d._note_pcm_wedge("device release stuck past 5.0s")
    d._started_at -= 3600.0

    finished = False

    async def _push() -> None:
        nonlocal finished
        await asyncio.sleep(0.05)
        finished = True

    d._webhook_tasks.add(asyncio.create_task(_push()))

    with patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()

    hard_exit.assert_called_once_with(1)
    assert finished is True, "exited before the push completed"


@pytest.mark.asyncio
async def test_restart_is_not_held_hostage_by_a_hung_push():
    """The drain is bounded — a push that never returns must not keep the
    daemon wedged forever, which is the very state we are recovering from."""
    d = Daemon(Config())
    d._note_pcm_wedge("device release stuck past 5.0s")
    d._started_at -= 3600.0

    hung = asyncio.create_task(asyncio.sleep(3600))
    d._webhook_tasks.add(hung)

    with patch("jarvis_cli.daemon.main._SELF_HEAL_PUSH_DRAIN_SECONDS", 0.05), \
            patch("jarvis_cli.daemon.main.os._exit") as hard_exit:
        await d._self_heal_if_wedged()

    hard_exit.assert_called_once_with(1)
    hung.cancel()
