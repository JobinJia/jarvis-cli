"""Fire-and-forget webhook notifier.

When enabled, the daemon POSTs the spoken Jarvis line plus a little event
metadata to a configured URL so a phone/IM (Bark, ntfy, Slack/Discord webhook,
or any generic JSON endpoint) can surface it while the user is away.

Payload shape (``application/json``, ``format = "generic"``)::

    {
      "text": "Sir, the build has completed.",   # the spoken Jarvis line
      "notification_type": "idle_prompt",          # Event.notification_type
      "tool_name": "Bash" | null,                  # Event.tool_name
      "cwd": "/path/to/project" | null,            # Event.cwd
      "received_at": 1718000000.0 | null           # epoch secs, passed in by caller
    }

We deliberately keep the payload generic (a flat JSON object) and let the
target be customized purely by URL + headers, so the same notifier targets:

  - Bark:    POST https://api.day.app/<key>  (Bark reads `title`/`body`, but
             also surfaces arbitrary JSON; for a clean Bark title/body use a
             URL like https://api.day.app/<key>/Jarvis/<text> instead — the
             generic POST still delivers).
  - ntfy:    POST https://ntfy.sh/<topic> with header `Title: Jarvis`; ntfy
             shows the JSON body as the message.
  - Slack:   POST https://hooks.slack.com/services/... — Slack expects a
             `{"text": ...}` object, which is exactly the top-level `text`
             field here, so it renders out of the box.
  - Discord: POST https://discord.com/api/webhooks/... — Discord expects
             `{"content": ...}`; set a header or use a relay if you need that
             exact key, otherwise the raw JSON is still logged on Discord's
             side. (Slack-compatible by appending `/slack` to the URL.)

Bark native mode (``format = "bark"``): the generic POST *delivers* to Bark
but renders as a JSON blob. With ``format = "bark"`` we instead emit Bark's
own fields — ``title`` ("Jarvis · <project>"), ``body`` (the spoken line),
``group`` (per-project notification stacking) and ``level`` — so the iOS push
(and its automatic Apple Watch mirror) reads like a real notification. The
URL embeds the device key (https://api.day.app/<key>). ``level`` maps from
the notification type: attention-needed events (permission prompts,
questions, failures, rate limits, overflow) are ``timeSensitive`` so they
punch through iOS Focus modes; everything else is ``active``.

Privacy note: by the time the daemon calls this, ``text`` is the final spoken
line. For LLM-phrased events that line was produced from an already
extract-+-redacted summary (see phrase/router.py, gated by
``behavior.privacy.cloud_redaction``), so no raw tool input leaks through the
phrasing. The ``cwd`` and ``tool_name`` we attach here are raw, unredacted
metadata — they never pass through the phrase redactor — so only enable the
webhook against an endpoint you trust with your project paths. Bark mode
exposes strictly less: only the project *basename* (title/group), never the
full path or tool name — but Bark pushes route through Apple's APNs plus the
Bark server unless you self-host, so the same trust call applies.

Fail-soft contract: every error (timeout, connection refused, non-2xx, bad
config) is caught and logged. This function NEVER raises and NEVER blocks the
audio path — the daemon fires it as a detached task.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from loguru import logger

from ..config import WebhookConfig
from ..types import Event

# Event types where the user's attention is actually needed (a session is
# blocked or something went wrong). In Bark terms these are "timeSensitive":
# iOS lets them punch through Focus modes; the rest ride as ordinary "active"
# pushes. Kept module-level so notify/remote.py's priority mapping and tests
# can share the same judgement call.
ATTENTION_TYPES: frozenset[str] = frozenset({
    "permission_prompt",
    "ask_user_question",
    "elicitation_dialog",
    "tool_failure",
    "api_error",
    "rate_limited",
    "max_turns_reached",
    "context_overflow",
})


def project_name(event: Event) -> str:
    """The project's directory basename, or "jarvis" when the event carries
    no cwd (pre-baked `say --text` lines, daemon self-announcements)."""
    return Path(event.cwd).name if event.cwd else "jarvis"


def _resolve_headers(cfg: WebhookConfig) -> dict[str, str]:
    """Merge static ``headers`` with an optional auth header whose value is
    read from an environment variable (so tokens stay out of config.toml).

    ``auth_header`` names the header (e.g. ``Authorization`` or
    ``X-Api-Key``); ``auth_env`` names the env var holding its value. If the
    env var is unset the auth header is simply omitted — the POST still goes
    out, matching the rest of the codebase's "missing key, fall through"
    posture.
    """
    headers = dict(cfg.headers)
    if cfg.auth_header and cfg.auth_env:
        value = os.getenv(cfg.auth_env)
        if value:
            headers[cfg.auth_header] = value
    return headers


def _build_payload(event: Event, text: str) -> dict[str, object | None]:
    return {
        "text": text,
        "notification_type": event.notification_type,
        "tool_name": event.tool_name,
        "cwd": event.cwd,
        # The daemon forbids wall-clock reads in some paths, so we never call
        # time.time() here — we surface whatever the Event already carries.
        "received_at": event.received_at or None,
    }


def _build_bark_payload(event: Event, text: str) -> dict[str, object]:
    """Bark-native shape: title/body render as a real iOS notification,
    `group` stacks pushes per project in Notification Center, and `level`
    decides whether the push may break through a Focus mode."""
    project = project_name(event)
    level = (
        "timeSensitive"
        if event.notification_type in ATTENTION_TYPES
        else "active"
    )
    return {
        "title": f"Jarvis · {project}",
        "body": text,
        "group": project,
        "level": level,
    }


async def notify(cfg: WebhookConfig, event: Event, text: str) -> bool:
    """POST the spoken ``text`` + ``event`` metadata to the configured webhook.

    Returns True on a 2xx response, False otherwise. Never raises: any error
    is logged and swallowed so a webhook problem cannot affect local audio or
    crash the daemon worker.
    """
    if not cfg.enabled or not cfg.url:
        return False
    # Optional allowlist: when set, only the listed notification types are
    # pushed. Empty list (default) means "push every event that reaches here".
    if cfg.events and event.notification_type not in cfg.events:
        logger.debug(
            "webhook: skip type={} (not in allowlist)", event.notification_type,
        )
        return False

    headers = _resolve_headers(cfg)
    # Unknown format strings fall through to generic rather than erroring —
    # a config typo must not silence remote pushes entirely.
    if cfg.format == "ntfy":
        # ntfy wants the message as the raw body and metadata as headers —
        # this lets ONE app cover both stacks: buttonless notify-only pushes
        # here, actionable decision pushes via notify/remote.py, same topic
        # or separate ones as the user prefers. Title must stay latin-1-safe
        # (httpx header constraint), so non-ASCII project names fall back.
        project = project_name(event)
        title = f"Jarvis - {project}" if project.isascii() else "Jarvis"
        headers.update({
            "Title": title,
            "Priority": (
                "high"
                if event.notification_type in ATTENTION_TYPES
                else "default"
            ),
            "Tags": "robot",
        })
        body: bytes | None = text.encode("utf-8")
        payload: dict | None = None
    elif cfg.format == "bark":
        body = None
        payload = _build_bark_payload(event, text)
    else:
        body = None
        payload = _build_payload(event, text)
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            if body is not None:
                r = await client.post(cfg.url, content=body, headers=headers)
            else:
                r = await client.post(cfg.url, json=payload, headers=headers)
            r.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — fail-soft: never break audio
        logger.warning("webhook notify failed ({}): {}", type(exc).__name__, exc)
        return False
    logger.debug("webhook delivered type={}", event.notification_type)
    return True
