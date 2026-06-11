# JARVIS_MUTE: silence hook events for spawned sub-Claude sessions

**Date:** 2026-06-11
**Status:** Approved

## Problem

Every `claude` CLI process fires the `SessionStart` hook with `source="startup"`,
and the payload carries no field that distinguishes a programmatically spawned
session from one the user opened (verified against code.claude.com/docs hooks
reference). When the orchestrate skill dispatches tasks via `spawn-agent.sh`,
each terminal panel is a fresh `claude` process, so every dispatched sub-Claude
triggers a full Jarvis briefing — plus permission prompts, idle reminders, and
question announcements as it runs. Dispatching N tasks produces N briefings.

In-process subagents (Agent tool) are not affected: `SessionStart` only fires
for the main session (subagents get `SubagentStart`, which jarvis-cli does not
listen to), so they need no handling.

## Decision

Mute **all** Jarvis events for spawned sub-Claude sessions (user choice:
full mute, not briefing-only). The user's own interactive sessions are
untouched.

## Mechanism

An environment variable acts as a per-session mute switch, checked at the hook
entry point. Hooks run as child processes of the `claude` process, so an env
var injected at spawn time is inherited by every hook invocation of that
session — no daemon changes, and any spawner (orchestrate, scripts, CI) can
opt in the same way.

Rejected alternatives:

- **Payload-based detection** — no distinguishing field exists in the
  `SessionStart` payload; not implementable.
- **Daemon-side filtering** (by cwd / session registry) — requires state
  registration and a protocol change; fragile for no added benefit.

## Changes

### 1. `src/jarvis_cli/hook_client.py`

At the top of `forward_event()`: if `os.environ.get("JARVIS_MUTE")` is
non-empty and not `"0"`/`"false"` (case-insensitive), return `False`
immediately — before reading the stream and before touching the socket.
A muted session is invisible to the daemon: no events, no cancels.

Dropping cancels from muted sessions is harmless: a muted session never
enqueues audio, so its `session_id` never matches anything in the daemon.

### 2. `~/.claude/skills/orchestrate/scripts/spawn-agent.sh` (outside this repo)

Both command-construction branches prefix the spawned command with the
inline variable:

```bash
COMMAND="JARVIS_MUTE=1 claude $SKIP_PERMISSIONS ..."
```

Inline (`VAR=1 cmd`) scoping guarantees the variable exists only in the
spawned `claude` process tree — the user's shell and main sessions are
unaffected.

## Behavior matrix

| Session | Briefing | Other events (permission/idle/question) |
|---|---|---|
| User-opened `claude` | speaks | speaks |
| orchestrate-spawned panel | silent | silent |
| Agent-tool subagent | n/a (never fired) | n/a |
| Manual `JARVIS_MUTE=1 claude` | silent | silent |

## Testing

`tests/unit/test_hook_client.py`:

- `JARVIS_MUTE=1` → `forward_event` returns `False` without reading the
  stream or connecting to the socket (pass a stream whose `read` raises to
  prove it is never consumed).
- `JARVIS_MUTE=0`, `JARVIS_MUTE=false`, unset → existing forwarding behavior
  unchanged.
- A `session_start` payload with `JARVIS_MUTE=1` set is dropped end-to-end.
