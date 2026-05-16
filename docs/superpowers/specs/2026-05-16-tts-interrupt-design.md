# TTS Interrupt on User Action — Design

Status: approved (2026-05-16)

## Problem

When Claude Code's Notification (or AskUserQuestion) hook fires, the daemon
speaks a sentence. If the user returns to the originating terminal and
responds *during* playback, audio keeps playing past the moment it is useful
— up to several seconds of wasted noise for longer phrases (AskUserQuestion
in particular).

Goal: as soon as the user takes a CC-side action in the originating session,
cut any in-flight audio for that session and drop any same-session events
still waiting in the queue.

## Non-goals

- Detecting that the user merely focused the terminal window. Focus alone
  shouldn't silence the prompt; the user may still want to hear it while
  deciding.
- Cross-session cancellation. Responding in Terminal B must not affect
  Terminal A's audio.
- Per-event "skip just this one" semantics. We always drop all pending
  same-session events; this matches the intent ("I have processed this
  prompt").

## Architecture

```
Claude Code
  ├─ Notification ───────────────► jarvis-cc-hook ──► socket: event   (existing)
  ├─ PreToolUse(AskUserQuestion) ► jarvis-cc-hook ──► socket: event   (existing)
  ├─ UserPromptSubmit ───────────► jarvis-cc-hook ──► socket: CANCEL  (new)
  └─ PostToolUse ────────────────► jarvis-cc-hook ──► socket: CANCEL  (new)
                                                            │
                                                            ▼
                                                  daemon listener
                                                            │
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                  kill current play proc       drop same-session events
                                  (afplay / ffplay)             from queue
```

Trigger choice — `UserPromptSubmit` covers "user typed a reply";
`PostToolUse` covers "user allowed a tool" and "user clicked an
AskUserQuestion option" (CC fires PostToolUse(AskUserQuestion) when the
dialog closes). Together they cover every CC-observable user response with
no separate signal needed.

`PostToolUse` also fires for auto-allowed tools that never produced a voice
prompt. Those cancels are cheap no-ops in the daemon.

## Wire protocol

The Unix socket stream stays NDJSON. A new row type is introduced:

```jsonc
// existing event rows (unchanged)
{"notification_type": "permission_prompt", "session_id": "abc", ...}

// new cancel row
{"command": "cancel", "session_id": "abc"}
```

`session_id` is required. Missing/empty `session_id` ⇒ hook drops the
event silently rather than send a cancel that could only be global.

## hook_client.py

`_translate_cc_payload` gains a branch:

```python
if payload.get("hook_event_name") in {"UserPromptSubmit", "PostToolUse"}:
    sid = payload.get("session_id")
    if not sid:
        return None
    return {"command": "cancel", "session_id": sid}
```

Existing PreToolUse(AskUserQuestion) and event-shape paths are unchanged.

When `behavior.cancel_on_user_action = false` is set in config, the hook
short-circuits these two event types and returns None before opening the
socket — zero overhead when disabled.

## listener.py

`serve_unix_socket` signature grows one parameter:

```python
async def serve_unix_socket(sock_path, on_event, on_cancel): ...
```

Per-line dispatch:

```python
payload = json.loads(line)
if payload.get("command") == "cancel":
    sid = payload.get("session_id")
    if sid:
        await on_cancel(sid)
    continue
ev = parse_payload(line)
# ... existing path
```

`parse_payload` is untouched.

## Daemon

New state:

```python
self._current_proc: asyncio.subprocess.Process | None = None
self._current_session_id: str | None = None
self._cancelled_sessions: set[str] = set()   # suppress noisy error logs
```

`_worker` registers/clears the proc around playback via `on_spawn`:

```python
def _register(proc):
    self._current_proc = proc
    self._current_session_id = event.session_id

try:
    if await self._try_stream(text, lang, event.voice_id, on_spawn=_register):
        pass
    else:
        # synth to file, then play
        await play(out_path, on_spawn=_register)
except Exception as exc:
    if event.session_id and event.session_id in self._cancelled_sessions:
        logger.debug("playback cancelled for {}", event.session_id)
    else:
        logger.exception("worker failed: {}", exc)
finally:
    self._current_proc = None
    self._current_session_id = None
```

After a successful start of the next event for a session, that sid is
removed from `_cancelled_sessions` (so the set never grows unboundedly).

New method:

```python
async def cancel_session(self, session_id: str) -> None:
    self._cancelled_sessions.add(session_id)
    self.queue.drop_matching(lambda e: e.session_id == session_id)
    proc = self._current_proc
    if proc is not None and self._current_session_id == session_id:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
```

## player.py

`play()` and `play_stream()` gain an optional `on_spawn`:

```python
async def play(audio, *, on_spawn: Callable[[Process], None] | None = None) -> None:
    proc = await asyncio.create_subprocess_exec("afplay", str(audio), ...)
    if on_spawn:
        on_spawn(proc)
    ...
```

Callers that don't pass `on_spawn` (including all existing tests) behave
identically.

## queue.py

`BoundedEventQueue` gains:

```python
def drop_matching(self, predicate: Callable[[Event], bool]) -> int:
    """Remove all queued events matching predicate. Returns count dropped."""
```

Implemented by rebuilding the internal deque under the existing lock (no
new lock needed; ops are on a single asyncio loop).

## install.py

`merge_claude_settings` extended to register the new hook types alongside
`Notification`. The same prune-then-append idiom applies for each:

```python
for hook_type in ("UserPromptSubmit", "PostToolUse"):
    ...  # prune our old entries, append one with matcher=""
```

`remove_from_claude_settings` symmetrically strips our entries from these
types so `uninstall` stays clean.

Existing `PreToolUse(AskUserQuestion)` entries in user settings are not
touched (that hook is currently configured manually; this design does not
change that).

## Config

New key in `~/.jarvis-cc/config.toml`:

```toml
[behavior]
cancel_on_user_action = true   # default
```

Loaded into `Config.behavior.cancel_on_user_action` (bool, default True).
Consumed only by `hook_client.py` to skip socket writes when disabled.

## Edge cases

| Case | Behavior |
|---|---|
| Event arrives with no `session_id` | Hook sends event normally; cannot be cancelled later. |
| Cancel arrives before any matching event | `_cancelled_sessions` records the sid; queue drain is a no-op; if a same-sid event arrives later it plays normally (set cleared on next successful start). |
| Cancel arrives mid-stream | `proc.kill()` ⇒ ffplay/afplay exits non-zero ⇒ worker swallows the exception because sid ∈ `_cancelled_sessions`. |
| Multiple cancels in a row | Idempotent. |
| `proc` already exited | `ProcessLookupError` caught. |
| Cancel races with new same-sid event being enqueued | Queue drain only sees what's already enqueued at cancel time. New event plays. |
| `cancel_on_user_action = false` | Hook returns None for the two cancel events; daemon path unchanged. |
| Old settings.json (no new hook entries) | Re-run `jarvis-cc install`; documented in README upgrade note. |

## Tests

- `tests/unit/test_hook_client.py` — translate UserPromptSubmit /
  PostToolUse into cancel row; drop when no sid; drop when config disables.
- `tests/unit/test_listener.py` — cancel row routed to `on_cancel`, not
  `on_event`.
- `tests/unit/test_install.py` — merge adds UserPromptSubmit + PostToolUse;
  remove strips them; existing PreToolUse blocks preserved.
- `tests/unit/test_queue.py` — `drop_matching` removes only matching items,
  returns count.
- `tests/unit/test_daemon_cancel.py` (new) — `cancel_session` kills
  current proc, drops same-sid queued events, leaves other sids alone;
  `_cancelled_sessions` set/cleared correctly.
- `tests/unit/test_player.py` — `on_spawn` invoked with the live Process
  handle.

## README

- "How it works" diagram: add the cancel arrows on the new hook types.
- "Configuration": document `cancel_on_user_action` default.
- "Install": add upgrade note — existing users re-run `jarvis-cc install`
  to register the new hooks.
