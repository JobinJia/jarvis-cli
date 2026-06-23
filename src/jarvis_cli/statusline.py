"""Claude Code `statusLine` command — token/cost observability for Jarvis.

Registered as the `jarvis-cli-statusline` console_script. Claude Code pipes a
JSON session blob to stdin (see the verified schema below) and renders whatever
this prints to stdout, on every status-line refresh. It runs on a HOT path, so:

  * stdlib only — no httpx/torch/loguru imports on the print path.
  * never raise — like hook_client, a traceback would corrupt CC's channel.
    Everything is caught, a safe fallback line is printed, and we exit 0.

Output (single line)::

    [Opus] 🧠 15.5k tok · 8% · 💰 $0.12 · 🌿 main

The compact line is always printed. The OPTIONAL spoken cost milestone (see
CostConfig) is forwarded to the daemon only when `cost.announce_milestones` is
enabled — and only the FIRST time the session crosses each `milestone_usd`
step, tracked via a per-session high-water file.

Verified statusLine stdin schema (subset we read), from
https://code.claude.com/docs/en/statusline (Full JSON schema accordion)::

    {
      "session_id": "abc123",
      "model": {"id": "claude-opus-4-8", "display_name": "Opus"},
      "workspace": {"current_dir": "/path"},
      "cost": {"total_cost_usd": 0.01234, ...},
      "context_window": {
        "total_input_tokens": 15500,
        "total_output_tokens": 1200,
        "used_percentage": 8,
        ...
      }
    }

`cost.total_cost_usd` and `context_window.*` may be null/absent before the
first API response; every read here tolerates that.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path


def _fmt_tokens(n: int) -> str:
    """Compact token count: 1500 -> '1.5k', 2_300_000 -> '2.3M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _git_branch(cwd: str | None) -> str | None:
    """Current branch via a fast plumbing call. Returns None outside a repo or
    on any failure — never raises. `git symbolic-ref` avoids the cost of a full
    `git branch` and prints nothing on a detached HEAD."""
    try:
        out = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = out.stdout.strip()
    return branch or None


def render_line(data: dict) -> str:
    """Build the compact status line from a parsed statusLine payload.

    Pure + total: every field is optional, so this never raises on a partial
    payload (which CC sends before the first API response)."""
    model = ((data.get("model") or {}).get("display_name")) or "Claude"

    cw = data.get("context_window") or {}
    in_tok = cw.get("total_input_tokens") or 0
    out_tok = cw.get("total_output_tokens") or 0
    total_tok = int(in_tok) + int(out_tok)
    pct = cw.get("used_percentage")

    cost = (data.get("cost") or {}).get("total_cost_usd")

    parts = [f"[{model}]"]
    parts.append(f"🧠 {_fmt_tokens(total_tok)} tok")
    if pct is not None:
        parts.append(f"{int(pct)}%")
    if cost is not None:
        parts.append(f"💰 ${float(cost):.2f}")

    cwd = data.get("cwd") or ((data.get("workspace") or {}).get("current_dir"))
    branch = _git_branch(cwd)
    if branch:
        parts.append(f"🌿 {branch}")

    return " · ".join(parts).replace("] · ", "] ", 1)


def _crossed_milestone(cost: float, step: float, last: float) -> float | None:
    """Return the new milestone threshold just crossed, or None.

    `last` is the highest threshold already announced for this session. We
    announce the LARGEST whole multiple of `step` at-or-below `cost` that is
    strictly greater than `last`, so a single render that vaults several steps
    (rare) still only speaks once — for the level actually reached."""
    if step <= 0 or cost < step:
        return None
    reached = (int(cost / step)) * step
    if reached > last + 1e-9:
        return float(reached)
    return None


def _milestone_state_path(state_dir: str, session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "anon"
    return Path(state_dir) / f"{safe}.json"


def _read_last_milestone(path: Path) -> float:
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("last", 0.0))
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        return 0.0


def _write_last_milestone(path: Path, value: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last": value}), encoding="utf-8")
    except OSError:
        pass


def _announce(text: str, sock_path: str, session_id: str | None) -> None:
    """Fire-and-forget a pre-baked idle_prompt to the daemon socket so Jarvis
    speaks the milestone. Mirrors hook_client's _send: short timeout, swallow
    every socket error (daemon may be down — the statusline must not stall)."""
    payload = {
        "notification_type": "idle_prompt",
        "tool_name": None,
        "tool_input": {},
        "text": text,
        "lang": "en",
        # No session_id: the milestone is a fire-once aside, not an
        # "awaiting input" line, so it should be immune to the session's
        # cancel signal (same reasoning as the session_start briefing).
        "session_id": None,
    }
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.3)
        s.connect(sock_path)
        s.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    except OSError:
        pass
    finally:
        try:
            s.close()
        except OSError:
            pass


def _maybe_announce_cost(data: dict) -> None:
    """If cost-milestone announcements are enabled and this render crosses a
    new spend threshold, send the spoken line to the daemon. Loads config
    lazily so the common (disabled) path never touches the filesystem twice.

    The daemon CANNOT detect this on its own: `cost.total_cost_usd` is only
    present in the statusLine payload, never in the lifecycle/notification
    hooks. So milestone detection has to live here, where the number is."""
    cost = (data.get("cost") or {}).get("total_cost_usd")
    if cost is None:
        return
    cost = float(cost)

    # Lazy, stdlib-only import — config.py pulls no heavy deps.
    from .config import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config(DEFAULT_CONFIG_PATH)
    if not cfg.cost.announce_milestones:
        return

    session_id = data.get("session_id") or "anon"
    state_path = _milestone_state_path(cfg.cost.state_dir, session_id)
    last = _read_last_milestone(state_path)
    reached = _crossed_milestone(cost, cfg.cost.milestone_usd, last)
    if reached is None:
        return

    dollars = int(reached) if reached == int(reached) else round(reached, 2)
    text = f"Sir, this session has passed {dollars} dollars."
    _announce(text, cfg.paths.socket, session_id)
    _write_last_milestone(state_path, reached)


def main() -> int:
    """Entry point for the `jarvis-cli-statusline` console_script.

    MUST never crash Claude Code: read stdin, print one line, exit 0. On any
    failure print a minimal fallback so the status row is never blank, since CC
    blanks the line when the script errors or prints nothing."""
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            data = {}
    except (json.JSONDecodeError, ValueError):
        data = {}

    try:
        line = render_line(data)
    except Exception:  # noqa: BLE001 — never blank the status row
        line = "[Claude]"
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

    # Milestone announcement is strictly secondary; isolate it so a failure
    # here can never affect the already-printed status line or the exit code.
    try:
        _maybe_announce_cost(data)
    except Exception:  # noqa: BLE001 — structural guarantee
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
