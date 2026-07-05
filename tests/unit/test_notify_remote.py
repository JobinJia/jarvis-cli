"""notify/remote.py: ntfy actionable pushes + reply-topic decision listener."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from jarvis_cli.config import Config, RemoteConfig
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.notify import remote
from jarvis_cli.types import Event


def _event(**kw) -> Event:
    base = dict(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={},
        cwd="/Users/me/proj",
        session_id="sid-42",
    )
    base.update(kw)
    return Event(**base)


def _cfg(**kw) -> RemoteConfig:
    base = dict(
        enabled=True,
        ntfy_base="https://ntfy.sh",
        topic_notify="secret-notify",
        topic_reply="secret-reply",
    )
    base.update(kw)
    return RemoteConfig(**base)


def _msg(body: str) -> str:
    return json.dumps({"event": "message", "message": body})


# --- push_actionable -------------------------------------------------------


@pytest.mark.asyncio
async def test_push_actionable_posts_body_and_action_buttons():
    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        seen["headers"] = dict(request.headers)
        return httpx.Response(200)

    with respx.mock(base_url="https://ntfy.sh") as router:
        router.post("/secret-notify").mock(side_effect=_handler)
        await remote.push_actionable(_cfg(), _event(), "May I run rm, sir?")

    assert seen["body"] == "May I run rm, sir?"
    h = seen["headers"]
    assert h["title"] == "Jarvis - proj"
    assert h["priority"] == "high"  # permission_prompt needs a decision
    assert h["tags"] == "robot"
    actions = h["actions"]
    approve, deny = actions.split("; ")
    assert approve.startswith(
        "http, Approve, https://ntfy.sh/secret-reply, method=POST, "
        "body=approve sid-42 "
    )
    assert deny.startswith(
        "http, Deny, https://ntfy.sh/secret-reply, method=POST, "
        "body=deny sid-42 "
    )
    # Both buttons carry the SAME nonce so either tap consumes it.
    assert approve.rsplit(" ", 1)[1] == deny.rsplit(" ", 1)[1]


@pytest.mark.asyncio
async def test_push_actionable_no_actions_without_session_id():
    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200)

    with respx.mock(base_url="https://ntfy.sh") as router:
        router.post("/secret-notify").mock(side_effect=_handler)
        await remote.push_actionable(_cfg(), _event(session_id=None), "hi")

    assert "actions" not in seen["headers"]


@pytest.mark.asyncio
async def test_push_actionable_default_priority_for_non_decision_types():
    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200)

    cfg = _cfg(events=["idle_prompt"])
    with respx.mock(base_url="https://ntfy.sh") as router:
        router.post("/secret-notify").mock(side_effect=_handler)
        await remote.push_actionable(
            cfg, _event(notification_type="idle_prompt"), "hi",
        )

    assert seen["headers"]["priority"] == "default"


@pytest.mark.asyncio
async def test_push_actionable_respects_event_filter():
    with respx.mock(
        base_url="https://ntfy.sh", assert_all_called=False,
    ) as router:
        route = router.post("/secret-notify").respond(200)
        await remote.push_actionable(
            _cfg(), _event(notification_type="idle_prompt"), "hi",
        )
    assert route.called is False


@pytest.mark.asyncio
async def test_push_actionable_non_ascii_project_title_falls_back():
    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200)

    with respx.mock(base_url="https://ntfy.sh") as router:
        router.post("/secret-notify").mock(side_effect=_handler)
        await remote.push_actionable(_cfg(), _event(cwd="/Users/me/项目"), "hi")

    assert seen["headers"]["title"] == "Jarvis"


@pytest.mark.asyncio
async def test_push_actionable_fails_soft_on_connection_error():
    with respx.mock(base_url="https://ntfy.sh") as router:
        router.post("/secret-notify").mock(
            side_effect=httpx.ConnectError("boom")
        )
        # Must not raise — logged and swallowed.
        await remote.push_actionable(_cfg(), _event(), "hi")


@pytest.mark.asyncio
async def test_push_actionable_disabled_is_noop():
    with respx.mock(
        base_url="https://ntfy.sh", assert_all_called=False,
    ) as router:
        route = router.post("/secret-notify").respond(200)
        await remote.push_actionable(_cfg(enabled=False), _event(), "hi")
    assert route.called is False


# --- listen_replies --------------------------------------------------------


@pytest.mark.asyncio
async def test_listen_replies_parses_skips_malformed_and_dedups_nonce():
    lines = "\n".join([
        json.dumps({"event": "open"}),          # non-message event
        "not json at all {{{",                    # malformed JSON
        _msg("approve sid-1 aaaa1111"),
        _msg("approve sid-1 aaaa1111"),           # duplicate nonce → ignored
        _msg("launch missiles now"),              # unknown verb
        _msg("approve onlytwo"),                  # wrong arity
        _msg("deny sid-2 bbbb2222"),
    ])
    calls: list[tuple[str, str]] = []
    done = asyncio.Event()

    async def on_decision(decision: str, sid: str) -> None:
        calls.append((decision, sid))
        if len(calls) == 2:
            done.set()

    with respx.mock(base_url="https://ntfy.sh") as router:
        router.get("/secret-reply/json").mock(
            return_value=httpx.Response(200, text=lines)
        )
        task = asyncio.create_task(remote.listen_replies(_cfg(), on_decision))
        try:
            await asyncio.wait_for(done.wait(), timeout=5)
        finally:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == [("approve", "sid-1"), ("deny", "sid-2")]


@pytest.mark.asyncio
async def test_listen_replies_reconnects_after_connection_error():
    """First connect blows up; the listener must swallow it, back off, and
    succeed on the second attempt — never raising out of the task."""
    calls: list[tuple[str, str]] = []
    done = asyncio.Event()

    async def on_decision(decision: str, sid: str) -> None:
        calls.append((decision, sid))
        done.set()

    with respx.mock(base_url="https://ntfy.sh") as router:
        router.get("/secret-reply/json").mock(
            side_effect=[
                httpx.ConnectError("down"),
                httpx.Response(200, text=_msg("approve sid-9 cccc3333")),
            ]
        )
        task = asyncio.create_task(remote.listen_replies(_cfg(), on_decision))
        try:
            # Generous budget: covers the 1s backoff after the first failure.
            await asyncio.wait_for(done.wait(), timeout=10)
        finally:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == [("approve", "sid-9")]


def test_parse_reply_accepts_only_wellformed_decisions():
    assert remote._parse_reply(_msg("approve sid-1 abcd1234")) == (
        "approve", "sid-1", "abcd1234",
    )
    assert remote._parse_reply(_msg("deny s n")) == ("deny", "s", "n")
    assert remote._parse_reply("") is None
    assert remote._parse_reply("garbage") is None
    assert remote._parse_reply(json.dumps({"event": "keepalive"})) is None
    assert remote._parse_reply(_msg("approve too many words here")) is None
    assert remote._parse_reply(json.dumps({"event": "message"})) is None


# --- daemon wiring ---------------------------------------------------------


@pytest.mark.asyncio
async def test_on_remote_decision_enqueues_spoken_ack():
    d = Daemon(Config())
    await d._on_remote_decision("approve", "sid-9")
    ev = await d.queue.get()
    assert ev.text == "Sir, remote approval received."
    assert ev.lang == "en"
    assert ev.notification_type == "idle_prompt"
    # session_id stays None so a pending cancel can't silence the ack.
    assert ev.session_id is None


@pytest.mark.asyncio
async def test_on_remote_decision_deny_ack_text():
    d = Daemon(Config())
    await d._on_remote_decision("deny", "sid-9")
    ev = await d.queue.get()
    assert ev.text == "Understood, sir — request denied remotely."


@pytest.mark.asyncio
async def test_on_remote_decision_spawns_bridge_cmd_with_env():
    d = Daemon(Config())
    d.cfg.remote.on_decision_cmd = "./bridge.sh"
    with patch(
        "jarvis_cli.daemon.main.asyncio.create_subprocess_shell",
        new_callable=AsyncMock,
    ) as spawn:
        await d._on_remote_decision("approve", "sid-9")
    spawn.assert_awaited_once()
    args, kwargs = spawn.call_args
    assert args[0] == "./bridge.sh"
    env = kwargs["env"]
    assert env["JARVIS_SESSION_ID"] == "sid-9"
    assert env["JARVIS_DECISION"] == "approve"
    assert env["JARVIS_CWD"] == ""


@pytest.mark.asyncio
async def test_on_remote_decision_no_cmd_no_spawn():
    d = Daemon(Config())  # on_decision_cmd defaults to ""
    with patch(
        "jarvis_cli.daemon.main.asyncio.create_subprocess_shell",
        new_callable=AsyncMock,
    ) as spawn:
        await d._on_remote_decision("approve", "sid-9")
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_remote_decision_spawn_failure_is_swallowed():
    d = Daemon(Config())
    d.cfg.remote.on_decision_cmd = "./bridge.sh"
    with patch(
        "jarvis_cli.daemon.main.asyncio.create_subprocess_shell",
        new_callable=AsyncMock,
        side_effect=OSError("no such file"),
    ):
        # Must not raise — the listener stays alive.
        await d._on_remote_decision("deny", "sid-9")
    # The spoken ack still landed despite the broken bridge.
    assert d.queue.size == 1
