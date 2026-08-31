"""Mute state — a small JSON file the CLI writes and the daemon reads.

Two scopes live here: a global mute (`jarvis-cli mute 30m`) that silences
everything, and per-event-type mutes (`jarvis-cli events off task_complete
2h`) that silence one kind of announcement. Both carry an expiry.

Why a file rather than a field in config.toml, which already hot-reloads:
config.toml is a hand-written preferences file, and a mute is transient
state with a timestamp in it — `install --reconfigure` rewrites that file
from a template and would drop or resurrect a mute either way. A separate
file also means `jarvis-cli mute` works while the daemon is down or
restarting, and that a launchd respawn (or the TTS self-heal) cannot
silently un-mute a machine its owner muted for a meeting.

Every mute is expiring by default. This repo has lost days to silences
nobody noticed (2026-07-06, 2026-07-17, 2026-08-15), so "quiet forever"
has to be asked for explicitly, and `describe()` surfaces whatever is in
force through /health.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# `forever` is stored in place of an expiry timestamp. It reads plainly in
# the file, and no clock skew can make it lapse.
FOREVER = "forever"

# How long a bare `jarvis-cli mute` lasts. Long enough for a meeting, short
# enough that forgetting to unmute costs one afternoon, not one week.
DEFAULT_MUTE_SECONDS = 30 * 60

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])?$", re.IGNORECASE)

Expiry = float | str  # epoch seconds, or FOREVER


def parse_duration(text: str) -> float:
    """Seconds for `60s` / `30m` / `1h` / `7d`; a bare number means minutes.

    Raises ValueError with a user-facing message on anything else — the CLI
    prints it verbatim.
    """
    m = _DURATION_RE.match(text.strip())
    if not m:
        raise ValueError(
            f"cannot read {text!r} as a duration — use 60s, 30m, 1h or 7d"
        )
    seconds = float(m.group(1)) * _UNITS[(m.group(2) or "m").lower()]
    if seconds <= 0:
        raise ValueError("duration must be greater than zero")
    return seconds


def format_duration(seconds: float) -> str:
    """`5400` -> `1h30m`. Rounds to whole seconds; never returns empty."""
    seconds = int(round(seconds))
    if seconds <= 0:
        return "0s"
    parts: list[str] = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            parts.append(f"{seconds // size}{unit}")
            seconds %= size
    # Two units is enough to be useful ("1h30m", "7d"); more is noise.
    return "".join(parts[:2])


def format_expiry(expiry: Expiry, now: float | None = None) -> str:
    """Human-readable form of one stored expiry, for CLI output and /health."""
    if expiry == FOREVER:
        return "forever"
    now = time.time() if now is None else now
    left = float(expiry) - now
    end = time.localtime(float(expiry))
    # A bare "until 02:03" is a lie for `mute 7d`, so anything landing on a
    # later day carries the date too.
    same_day = end[:3] == time.localtime(now)[:3]
    clock = time.strftime("%H:%M" if same_day else "%m-%d %H:%M", end)
    if left <= 0:
        return f"expired at {clock}"
    return f"until {clock} ({format_duration(left)} left)"


def expiry_for(seconds: float | None) -> Expiry:
    """Turn a duration (or None for indefinite) into a stored expiry."""
    return FOREVER if seconds is None else time.time() + seconds


def _is_active(expiry: Expiry | None, now: float) -> bool:
    if expiry is None:
        return False
    return expiry == FOREVER or float(expiry) > now


# --- state file ------------------------------------------------------------


def load(path: str | Path) -> dict[str, Any]:
    """Read the state file. A missing, empty, or corrupt file reads as
    "nothing is muted" — a state file the daemon cannot parse must never be
    able to silence it, and must never be able to crash it either."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    types = data.get("types")
    return {
        "global": data.get("global"),
        "types": types if isinstance(types, dict) else {},
    }


def save(path: str | Path, state: dict[str, Any]) -> None:
    """Write the state file, dropping anything that has already lapsed so the
    file stays a readable record of what is actually in force."""
    now = time.time()
    pruned: dict[str, Any] = {}
    if _is_active(state.get("global"), now):
        pruned["global"] = state["global"]
    types = {
        k: v for k, v in (state.get("types") or {}).items()
        if _is_active(v, now)
    }
    if types:
        pruned["types"] = types
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pruned, indent=2) + "\n", encoding="utf-8")


def set_global(path: str | Path, seconds: float | None) -> Expiry:
    """Mute everything for `seconds` (None = indefinitely). Returns the expiry."""
    state = load(path)
    expiry = expiry_for(seconds)
    state["global"] = expiry
    save(path, state)
    return expiry


def set_type(
    path: str | Path, notification_type: str, seconds: float | None,
) -> Expiry:
    """Mute one event type for `seconds` (None = indefinitely)."""
    state = load(path)
    expiry = expiry_for(seconds)
    state.setdefault("types", {})[notification_type] = expiry
    save(path, state)
    return expiry


def clear_type(path: str | Path, notification_type: str) -> bool:
    """Lift a per-type mute. Returns True when one was actually in force."""
    state = load(path)
    had = _is_active(state.get("types", {}).get(notification_type), time.time())
    state.get("types", {}).pop(notification_type, None)
    save(path, state)
    return had


def clear_all(path: str | Path) -> bool:
    """Lift every mute, global and per-type. `unmute` means "speak again",
    so it does not leave a per-type mute behind to puzzle over later.
    Returns True when anything was in force."""
    active = describe(load(path))
    save(path, {})
    return bool(active)


# --- queries ---------------------------------------------------------------


def muted_reason(
    state: dict[str, Any], notification_type: str, now: float | None = None,
) -> str | None:
    """Why this event must stay silent, or None to let it speak.

    The string is log/status copy, so it names the scope and the expiry.
    """
    now = time.time() if now is None else now
    g = state.get("global")
    if _is_active(g, now):
        return f"muted {format_expiry(g, now)}"
    t = (state.get("types") or {}).get(notification_type)
    if _is_active(t, now):
        return f"{notification_type} muted {format_expiry(t, now)}"
    return None


def describe(state: dict[str, Any], now: float | None = None) -> dict[str, str]:
    """What is in force right now, as {scope: human-readable expiry}, for
    /health and `jarvis-cli status`. Empty dict when nothing is muted."""
    now = time.time() if now is None else now
    out: dict[str, str] = {}
    g = state.get("global")
    if _is_active(g, now):
        out["all"] = format_expiry(g, now)
    for ntype, expiry in (state.get("types") or {}).items():
        if _is_active(expiry, now):
            out[ntype] = format_expiry(expiry, now)
    return out
