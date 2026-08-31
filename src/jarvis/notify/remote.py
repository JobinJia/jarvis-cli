"""ntfy actionable approvals: a watch → Mac decision loop with no inbound port.

When a session blocks on something that needs the user (permission prompt,
question, elicitation), the daemon pushes an ntfy notification carrying
Approve/Deny **action buttons**. Tapping a button makes the *phone* POST a
tiny command string ("approve <sid> <nonce>") to a second ntfy topic; the
daemon holds a long-lived streaming subscription to that reply topic and
reacts. Both directions therefore ride ntfy over HTTPS — the phone/watch
never needs inbound network access to the Mac, no tunnel, no port forward.

Security model: ntfy topics are unauthenticated — the topic NAME is the
credential. Both ``topic_notify`` and ``topic_reply`` must be long random
strings (bearer secrets): anyone who learns the notify topic can read your
pushes, and anyone who learns the reply topic can inject approve/deny
decisions. Generate them like passwords (e.g. ``openssl rand -hex 16``),
never reuse them across machines, and for the paranoid: self-host ntfy so
the secrets never transit a third party. Each push additionally embeds a
short random nonce; the listener remembers seen nonces, so a duplicated or
replayed button tap is ignored.

What a decision *does* is deliberately pluggable: the daemon speaks an ack
and (optionally) runs ``on_decision_cmd`` with JARVIS_SESSION_ID /
JARVIS_DECISION / JARVIS_CWD in the environment — the bridge toward actually
unblocking a session (e.g. wiring an orchestrator's send-input.sh).

Fail-soft contract, same as notify/webhook.py: push_actionable never raises
and never blocks audio; listen_replies reconnects forever and only exits on
task cancellation. A remote-approval problem must NEVER break local audio.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable

import httpx
from loguru import logger

from ..config import RemoteConfig
from ..types import Event
from .webhook import project_name

# Decision events block a session on the user, so they warrant an ntfy
# priority that vibrates/breaks through; anything else the user opted into
# rides at default priority.
_HIGH_PRIORITY_TYPES = frozenset({
    "permission_prompt", "ask_user_question", "elicitation_dialog",
})

# How many reply nonces we remember for replay protection. Decisions are
# rare (human-tapped), so even a small bound covers days; the bound exists
# only so an always-on daemon can't grow the set forever.
_SEEN_NONCE_CAP = 256


def _ascii_title(event: Event) -> str:
    """ntfy titles travel as HTTP headers, and httpx requires header values
    to be latin-1-encodable (and ntfy re-encodes them as UTF-8 JSON, which
    mangles raw latin-1 bytes). Rather than juggling RFC 2047 encoded-words,
    keep the title pure ASCII: use the project dir name when it is ASCII,
    else fall back to plain "Jarvis". The full text still arrives in the
    message body, which is sent as UTF-8 and survives any alphabet."""
    project = project_name(event)
    if project.isascii():
        return f"Jarvis - {project}"
    return "Jarvis"


async def push_actionable(cfg: RemoteConfig, event: Event, text: str) -> None:
    """Push ``text`` to the notify topic with Approve/Deny action buttons.

    Fail-soft: never raises, never blocks audio — the daemon fires this as a
    detached task. Buttons are only attached when the event carries a
    session_id (without one there is nothing a decision could unblock).
    """
    if not cfg.enabled or not cfg.topic_notify:
        return
    if cfg.events and event.notification_type not in cfg.events:
        logger.debug(
            "remote: skip type={} (not in events)", event.notification_type,
        )
        return

    base = cfg.ntfy_base.rstrip("/")
    headers = {
        "Title": _ascii_title(event),
        "Priority": (
            "high"
            if event.notification_type in _HIGH_PRIORITY_TYPES
            else "default"
        ),
        "Tags": "robot",
    }
    sid = event.session_id
    if sid and cfg.topic_reply:
        # ntfy's Actions header grammar: `; `-separated actions, each a
        # comma-separated list — so labels must not contain commas. The
        # nonce lets the listener drop duplicate/replayed taps.
        nonce = uuid.uuid4().hex[:8]
        reply_url = f"{base}/{cfg.topic_reply}"
        headers["Actions"] = (
            f"http, Approve, {reply_url}, method=POST, "
            f"body=approve {sid} {nonce}; "
            f"http, Deny, {reply_url}, method=POST, "
            f"body=deny {sid} {nonce}"
        )
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            r = await client.post(
                f"{base}/{cfg.topic_notify}",
                content=text.encode("utf-8"),
                headers=headers,
            )
            r.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — fail-soft: never break audio
        logger.warning(
            "remote push failed ({}): {}", type(exc).__name__, exc,
        )
        return
    logger.debug("remote push delivered type={}", event.notification_type)


def _parse_reply(line: str) -> tuple[str, str, str] | None:
    """Parse one line of the ntfy JSON stream into (decision, sid, nonce).

    Returns None for anything that isn't a well-formed decision: keepalive/
    open events, malformed JSON, or bodies that don't match
    ``approve|deny <sid> <nonce>``. The reply topic is world-writable to
    anyone holding the secret, so we treat every line as untrusted input and
    silently drop what we don't recognize.
    """
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("event") != "message":
        return None
    body = obj.get("message")
    if not isinstance(body, str):
        return None
    parts = body.split()
    if len(parts) != 3 or parts[0] not in ("approve", "deny"):
        return None
    decision, sid, nonce = parts
    return decision, sid, nonce


async def listen_replies(
    cfg: RemoteConfig,
    on_decision: Callable[[str, str], Awaitable[None]],
) -> None:
    """Subscribe to the reply topic forever; call ``on_decision(decision, sid)``
    for each valid, first-seen decision line.

    Reconnects on ANY error with capped exponential backoff (1s → 60s) and
    never raises out — the daemon runs this for its whole lifetime and a
    flaky network must not kill the task. The only exit is CancelledError
    (daemon shutdown), which propagates cleanly.
    """
    url = f"{cfg.ntfy_base.rstrip('/')}/{cfg.topic_reply}/json"
    # Insertion-ordered so eviction drops the OLDEST nonce once we hit cap.
    seen: OrderedDict[str, None] = OrderedDict()
    backoff = 1.0
    while True:
        try:
            # The stream is intentionally endless: connect promptly, then
            # read without a deadline (ntfy sends keepalive lines).
            timeout = httpx.Timeout(cfg.timeout_seconds, read=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    # Connected: a healthy stream earns a fresh backoff.
                    backoff = 1.0
                    async for line in resp.aiter_lines():
                        parsed = _parse_reply(line)
                        if parsed is None:
                            continue
                        decision, sid, nonce = parsed
                        if nonce in seen:
                            logger.debug(
                                "remote: duplicate nonce {} ignored", nonce,
                            )
                            continue
                        seen[nonce] = None
                        while len(seen) > _SEEN_NONCE_CAP:
                            seen.popitem(last=False)
                        await on_decision(decision, sid)
        except asyncio.CancelledError:
            raise  # daemon shutdown — the one clean exit
        except Exception as exc:  # noqa: BLE001 — reconnect forever
            logger.warning(
                "remote listener error ({}): {} — reconnecting in {:.0f}s",
                type(exc).__name__, exc, backoff,
            )
        else:
            # Server closed an otherwise-healthy stream; reconnect politely.
            logger.debug(
                "remote listener stream ended — reconnecting in {:.0f}s",
                backoff,
            )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)
