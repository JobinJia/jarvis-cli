# TTS Interrupt on User Action — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When voice is playing for a CC notification, cancel the audio (and drop pending same-session events) the moment the user responds in the originating Claude Code session.

**Architecture:** New hook events `UserPromptSubmit` and `PostToolUse` send a `{"command":"cancel","session_id":...}` row over the existing Unix socket. The daemon tracks the currently-playing subprocess plus its session_id, and on cancel kills the proc and drains the queue of same-session events.

**Tech Stack:** Python 3.11, asyncio, Unix socket NDJSON, pytest. Same stack as the rest of `jarvis-cc`.

**Spec:** `docs/superpowers/specs/2026-05-16-tts-interrupt-design.md`

---

## File Structure

**Modify (existing, in dependency order):**

- `src/jarvis_cc/config.py` — add `BehaviorConfig.cancel_on_user_action: bool = True`.
- `src/jarvis_cc/daemon/queue.py` — add `BoundedEventQueue.drop_matching(predicate)`.
- `src/jarvis_cc/player.py` — both `play()` and `play_stream()` accept optional `on_spawn` callback so the worker can capture the subprocess handle.
- `src/jarvis_cc/daemon/listener.py` — route `{"command":"cancel",...}` rows to a new `on_cancel` callback parameter.
- `src/jarvis_cc/daemon/main.py` — track current proc + session_id, implement `cancel_session`, wire `on_cancel` into the listener, swallow expected exceptions when sid is in `_cancelled_sessions`.
- `src/jarvis_cc/hook_client.py` — translate `UserPromptSubmit` / `PostToolUse` payloads into cancel rows; honor `cancel_on_user_action`.
- `src/jarvis_cc/install.py` — register new hook types in `merge_claude_settings`; clean them in `remove_from_claude_settings`.
- `README.md` — document `cancel_on_user_action`; add upgrade note.

**Modify (existing tests):**

- `tests/unit/test_queue.py`
- `tests/unit/test_player.py`
- `tests/unit/test_listener.py`
- `tests/unit/test_hook_client.py`
- `tests/unit/test_install.py`
- `tests/unit/test_config.py` (only if it asserts BehaviorConfig fields)

**Create:**

- `tests/unit/test_daemon_cancel.py` — exercises `Daemon.cancel_session`.

---

## Task 1: Config flag `cancel_on_user_action`

**Files:**
- Modify: `src/jarvis_cc/config.py:78-99` (BehaviorConfig)
- Test: `tests/unit/test_config.py`

- [ ] **Step 1.1: Write the failing test**

Append to `tests/unit/test_config.py`:

```python
def test_behavior_default_has_cancel_on_user_action_true(tmp_path):
    from jarvis_cc.config import load_config
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("")  # empty → all defaults
    cfg = load_config(cfg_path)
    assert cfg.behavior.cancel_on_user_action is True


def test_behavior_cancel_on_user_action_overridable(tmp_path):
    from jarvis_cc.config import load_config
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[behavior]\ncancel_on_user_action = false\n")
    cfg = load_config(cfg_path)
    assert cfg.behavior.cancel_on_user_action is False
```

- [ ] **Step 1.2: Run test to verify it fails**

```
uv run pytest tests/unit/test_config.py::test_behavior_default_has_cancel_on_user_action_true -v
```

Expected: FAIL with `AttributeError: 'BehaviorConfig' object has no attribute 'cancel_on_user_action'`.

- [ ] **Step 1.3: Add the field**

Edit `src/jarvis_cc/config.py` — inside `BehaviorConfig`, add after `phrase_hard_cap`:

```python
    # When True (default), the hook sends a cancel signal on UserPromptSubmit /
    # PostToolUse so the daemon stops any in-flight audio for that session.
    cancel_on_user_action: bool = True
```

- [ ] **Step 1.4: Run both new tests to verify they pass**

```
uv run pytest tests/unit/test_config.py -k cancel_on_user_action -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```
git add src/jarvis_cc/config.py tests/unit/test_config.py
git commit -m "feat(config): add behavior.cancel_on_user_action flag (default true)"
```

---

## Task 2: `BoundedEventQueue.drop_matching`

**Files:**
- Modify: `src/jarvis_cc/daemon/queue.py`
- Test: `tests/unit/test_queue.py`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/unit/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_queue_drop_matching_removes_only_matches():
    q = BoundedEventQueue(maxsize=10)
    for i in range(5):
        await q.put_or_drop(_ev(i))
    # Drop events whose tool_name is T1 or T3
    removed = q.drop_matching(lambda e: e.tool_name in {"T1", "T3"})
    assert removed == 2
    survivors = []
    while q.size:
        survivors.append((await q.get()).tool_name)
    assert survivors == ["T0", "T2", "T4"]


@pytest.mark.asyncio
async def test_queue_drop_matching_returns_zero_when_no_match():
    q = BoundedEventQueue(maxsize=10)
    await q.put_or_drop(_ev(0))
    assert q.drop_matching(lambda e: e.tool_name == "nope") == 0
    assert q.size == 1
```

- [ ] **Step 2.2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_queue.py -k drop_matching -v
```

Expected: FAIL with `AttributeError: 'BoundedEventQueue' object has no attribute 'drop_matching'`.

- [ ] **Step 2.3: Implement `drop_matching`**

In `src/jarvis_cc/daemon/queue.py`, add:

```python
    def drop_matching(self, predicate) -> int:
        """Remove every queued event for which predicate(event) is True.

        Synchronous because callers (the cancel path) run on the same asyncio
        loop as put/get. Returns count removed.
        """
        before = len(self._deque)
        kept = [e for e in self._deque if not predicate(e)]
        self._deque.clear()
        self._deque.extend(kept)
        return before - len(self._deque)
```

Also add to the top of the file (next to existing imports), if not already present, the typing import — but `predicate` is left untyped to avoid a dependency on `Callable` here; the docstring documents the contract.

- [ ] **Step 2.4: Run tests to verify they pass**

```
uv run pytest tests/unit/test_queue.py -v
```

Expected: all queue tests pass (existing 2 + new 2 = 4).

- [ ] **Step 2.5: Commit**

```
git add src/jarvis_cc/daemon/queue.py tests/unit/test_queue.py
git commit -m "feat(queue): add drop_matching for selective in-place removal"
```

---

## Task 3: `player.on_spawn` callback

**Files:**
- Modify: `src/jarvis_cc/player.py`
- Test: `tests/unit/test_player.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/unit/test_player.py`:

```python
@pytest.mark.asyncio
async def test_play_invokes_on_spawn_with_proc(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    seen = []

    class _P:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

        async def wait(self):
            return 0

    async def _fake_exec(*args, **kwargs):
        return _P()

    with patch("jarvis_cc.player.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        await play(audio, on_spawn=seen.append)

    assert len(seen) == 1
    assert isinstance(seen[0], _P)


@pytest.mark.asyncio
async def test_play_stream_invokes_on_spawn_with_proc():
    from jarvis_cc.player import play_stream
    seen = []

    class _Stdin:
        async def drain(self): return None
        def write(self, data: bytes): pass
        def close(self): pass
        async def wait_closed(self): return None
        def is_closing(self): return False

    class _P:
        returncode = 0
        stdin = _Stdin()

        async def wait(self): return 0

    async def _fake_exec(*args, **kwargs):
        return _P()

    async def _chunks():
        yield b"x"

    with patch(
        "jarvis_cc.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        await play_stream(_chunks(), on_spawn=seen.append)

    assert len(seen) == 1
    assert isinstance(seen[0], _P)
```

- [ ] **Step 3.2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_player.py -k on_spawn -v
```

Expected: FAIL with `TypeError: play() got an unexpected keyword argument 'on_spawn'`.

- [ ] **Step 3.3: Implement on_spawn in `play` and `play_stream`**

Rewrite `src/jarvis_cc/player.py` to:

```python
"""Audio playback helpers.

`play(path)`            — afplay; reads a finished audio file from disk.
`play_stream(chunks)`   — ffplay; reads MP3 chunks from stdin so playback
                          starts before synthesis completes.

Both accept an optional `on_spawn(proc)` callback so callers (the daemon
worker) can capture the subprocess handle for external cancellation.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path


async def play(
    audio: Path,
    *,
    on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "afplay", str(audio),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if on_spawn is not None:
        on_spawn(proc)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"afplay failed: {err.decode(errors='replace')}")


async def play_stream(
    chunks: AsyncIterator[bytes],
    *,
    on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
) -> None:
    """Spawn ffplay reading MP3 from stdin and feed it chunks as they arrive."""
    proc = await asyncio.create_subprocess_exec(
        "ffplay",
        "-loglevel", "error",
        "-nodisp",
        "-autoexit",
        "-i", "pipe:0",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None  # PIPE was requested above
    if on_spawn is not None:
        on_spawn(proc)
    try:
        async for chunk in chunks:
            if not chunk:
                continue
            proc.stdin.write(chunk)
            await proc.stdin.drain()
    finally:
        try:
            if not proc.stdin.is_closing():
                proc.stdin.close()
                await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffplay exited with code {rc}")
```

- [ ] **Step 3.4: Run all player tests**

```
uv run pytest tests/unit/test_player.py -v
```

Expected: all player tests pass (4 existing + 2 new = 6).

- [ ] **Step 3.5: Commit**

```
git add src/jarvis_cc/player.py tests/unit/test_player.py
git commit -m "feat(player): optional on_spawn callback to expose play subprocess"
```

---

## Task 4: Listener routes cancel rows

**Files:**
- Modify: `src/jarvis_cc/daemon/listener.py`
- Test: `tests/unit/test_listener.py`

- [ ] **Step 4.1: Write the failing test**

Append to `tests/unit/test_listener.py`:

```python
@pytest.mark.asyncio
async def test_serve_unix_socket_routes_cancel_command(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    events: list[Event] = []
    cancels: list[str] = []

    async def on_event(ev: Event):
        events.append(ev)

    async def on_cancel(sid: str):
        cancels.append(sid)

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_cancel=on_cancel)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall(
        (json.dumps({"command": "cancel", "session_id": "abc"}) + "\n").encode()
    )
    s.close()

    await asyncio.sleep(0.1)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert cancels == ["abc"]
    assert events == []


@pytest.mark.asyncio
async def test_serve_unix_socket_ignores_cancel_without_session_id(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    cancels: list[str] = []

    async def on_event(ev: Event): pass
    async def on_cancel(sid: str): cancels.append(sid)

    server_task = asyncio.create_task(
        serve_unix_socket(sock_path, on_event, on_cancel=on_cancel)
    )
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall((json.dumps({"command": "cancel"}) + "\n").encode())
    s.close()
    await asyncio.sleep(0.1)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert cancels == []
```

- [ ] **Step 4.2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_listener.py -k cancel -v
```

Expected: FAIL with `TypeError: serve_unix_socket() got an unexpected keyword argument 'on_cancel'`.

- [ ] **Step 4.3: Implement cancel routing**

Edit `src/jarvis_cc/daemon/listener.py` — change `serve_unix_socket` signature and dispatch:

```python
async def serve_unix_socket(
    sock_path: Path,
    on_event: Callable[[Event], Awaitable[None]],
    *,
    on_cancel: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Run a unix-socket server forever, dispatching parsed rows.

    Event rows (with `notification_type`) go to `on_event`.
    `{"command":"cancel","session_id":"..."}` rows go to `on_cancel`.
    Rows missing session_id on cancel are dropped silently.
    """
    sock_path = Path(sock_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await reader.read(65536)
            for line in data.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Dropped malformed JSON: {!r}", line[:120])
                    continue
                if isinstance(payload, dict) and payload.get("command") == "cancel":
                    sid = payload.get("session_id")
                    if sid and on_cancel is not None:
                        await on_cancel(sid)
                    continue
                ev = parse_payload(line)
                if ev is None:
                    logger.warning("Dropped malformed/unknown event: {!r}", line[:120])
                    continue
                await on_event(ev)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    server = await asyncio.start_unix_server(handle, path=str(sock_path))
    os.chmod(sock_path, 0o600)
    logger.info("Listener bound to {}", sock_path)
    try:
        async with server:
            await server.serve_forever()
    finally:
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass
```

The existing test `test_serve_unix_socket_yields_events` calls `serve_unix_socket(sock_path, collect)` with only the positional `on_event`; `on_cancel` is keyword-only with a default of `None`, so that test keeps working.

- [ ] **Step 4.4: Run all listener tests**

```
uv run pytest tests/unit/test_listener.py -v
```

Expected: all listener tests pass (existing 4 + new 2 = 6).

- [ ] **Step 4.5: Commit**

```
git add src/jarvis_cc/daemon/listener.py tests/unit/test_listener.py
git commit -m "feat(listener): route cancel command rows to on_cancel callback"
```

---

## Task 5: `Daemon.cancel_session` + worker integration

**Files:**
- Modify: `src/jarvis_cc/daemon/main.py`
- Create: `tests/unit/test_daemon_cancel.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/unit/test_daemon_cancel.py`:

```python
"""Daemon.cancel_session: kill current proc + drop same-sid queued events."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from jarvis_cc.config import Config
from jarvis_cc.daemon.main import Daemon
from jarvis_cc.types import Event


def _ev(sid: str | None, tool: str = "T") -> Event:
    return Event(
        notification_type="permission_prompt",
        tool_name=tool,
        cwd=f"/{sid}",
        session_id=sid,
    )


@pytest.mark.asyncio
async def test_cancel_session_kills_current_proc_for_matching_sid():
    d = Daemon(Config())
    proc = MagicMock()
    proc.kill = MagicMock()
    d._current_proc = proc
    d._current_session_id = "abc"

    await d.cancel_session("abc")

    proc.kill.assert_called_once()
    assert "abc" in d._cancelled_sessions


@pytest.mark.asyncio
async def test_cancel_session_does_not_kill_proc_for_other_sid():
    d = Daemon(Config())
    proc = MagicMock()
    proc.kill = MagicMock()
    d._current_proc = proc
    d._current_session_id = "abc"

    await d.cancel_session("xyz")

    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_session_drops_matching_queued_events():
    d = Daemon(Config())
    await d.queue.put_or_drop(_ev("abc", tool="T1"))
    await d.queue.put_or_drop(_ev("xyz", tool="T2"))
    await d.queue.put_or_drop(_ev("abc", tool="T3"))

    await d.cancel_session("abc")

    survivors = []
    while d.queue.size:
        survivors.append((await d.queue.get()).tool_name)
    assert survivors == ["T2"]


@pytest.mark.asyncio
async def test_cancel_session_handles_process_lookup_error():
    d = Daemon(Config())

    proc = MagicMock()
    proc.kill = MagicMock(side_effect=ProcessLookupError())
    d._current_proc = proc
    d._current_session_id = "abc"

    # Should not raise
    await d.cancel_session("abc")
```

- [ ] **Step 5.2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_daemon_cancel.py -v
```

Expected: FAIL with `AttributeError` on `_current_proc` / `cancel_session`.

- [ ] **Step 5.3: Implement state + `cancel_session` in `Daemon`**

Edit `src/jarvis_cc/daemon/main.py`:

In `Daemon.__init__`, after `self._last_text = None`, add:

```python
        self._current_proc: asyncio.subprocess.Process | None = None
        self._current_session_id: str | None = None
        self._cancelled_sessions: set[str] = set()
```

Add a new method on `Daemon` (place it after `_snapshot`):

```python
    async def cancel_session(self, session_id: str) -> None:
        """Cancel any in-flight audio for `session_id` and drop its queued events."""
        self._cancelled_sessions.add(session_id)
        self.queue.drop_matching(lambda e: e.session_id == session_id)
        proc = self._current_proc
        if proc is not None and self._current_session_id == session_id:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
```

- [ ] **Step 5.4: Run test_daemon_cancel.py**

```
uv run pytest tests/unit/test_daemon_cancel.py -v
```

Expected: 4 passed.

- [ ] **Step 5.5: Wire `on_spawn` and cancel-aware error swallowing in `_worker`**

Replace `_worker` and `_try_stream` in `src/jarvis_cc/daemon/main.py` with:

```python
    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            sid = event.session_id
            if sid and sid in self._cancelled_sessions:
                # Stale cancel signal preceded the event; clear and play normally.
                self._cancelled_sessions.discard(sid)

            def _register(proc: asyncio.subprocess.Process) -> None:
                self._current_proc = proc
                self._current_session_id = sid

            try:
                if event.text is not None:
                    text = event.text
                    lang = event.lang or "en"
                else:
                    lang = (
                        detect_for(event.cwd)
                        if self.cfg.behavior.voice_language == "auto"
                        else self.cfg.behavior.voice_language  # type: ignore[assignment]
                    )
                    text = await self.router.phrase(event, lang=lang)
                self._last_text = text
                if await self._try_stream(text, lang, event.voice_id, on_spawn=_register):
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = Path(tmp.name)
                await self.tts.synthesize(text, lang, out_path, voice_id=event.voice_id)
                try:
                    await play(out_path, on_spawn=_register)
                finally:
                    try:
                        out_path.unlink()
                    except OSError:
                        pass
            except Exception as exc:
                if sid and sid in self._cancelled_sessions:
                    logger.debug("playback cancelled for session {}", sid)
                else:
                    logger.exception("worker failed: {}", exc)
            finally:
                self._current_proc = None
                self._current_session_id = None
                if sid:
                    self._cancelled_sessions.discard(sid)

    async def _try_stream(
        self, text: str, lang, voice_id: str | None,
        *, on_spawn=None,
    ) -> bool:
        """If the primary TTS supports streaming, pipe chunks straight to
        ffplay so playback begins before synthesis completes. Returns True
        on success; False (with a warning) on any failure so the caller can
        fall back to the file-based synth+afplay path."""
        primary = self.tts.primary
        if not getattr(primary, "supports_streaming", False):
            return False
        try:
            await play_stream(
                primary.stream(text, lang, voice_id=voice_id),
                on_spawn=on_spawn,
            )
            return True
        except Exception as exc:
            logger.warning("Streaming TTS failed, falling back to synth: {}", exc)
            return False
```

- [ ] **Step 5.6: Wire `on_cancel` into the listener call in `run`**

In `Daemon.run`, change:

```python
            await serve_unix_socket(Path(self.cfg.paths.socket), self._on_event)
```

to:

```python
            await serve_unix_socket(
                Path(self.cfg.paths.socket),
                self._on_event,
                on_cancel=self.cancel_session,
            )
```

- [ ] **Step 5.7: Run the full unit suite to verify nothing regressed**

```
uv run pytest tests/unit/ -v
```

Expected: all previously-passing tests still pass, plus the new ones from Tasks 1–5.

- [ ] **Step 5.8: Commit**

```
git add src/jarvis_cc/daemon/main.py tests/unit/test_daemon_cancel.py
git commit -m "feat(daemon): cancel_session — kill current proc + drain queue"
```

---

## Task 6: `hook_client` translates cancel events

**Files:**
- Modify: `src/jarvis_cc/hook_client.py`
- Test: `tests/unit/test_hook_client.py`

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/unit/test_hook_client.py`:

```python
def test_forward_event_userpromptsubmit_sends_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "abc-123",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row == {"command": "cancel", "session_id": "abc-123"}


def test_forward_event_posttooluse_sends_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "sess-9",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row == {"command": "cancel", "session_id": "sess-9"}


def test_forward_event_userpromptsubmit_without_session_id_is_dropped(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {"hook_event_name": "UserPromptSubmit", "cwd": "/x"}
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is False


def test_forward_event_cancel_disabled_by_flag(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "abc",
    }
    ok = forward_event(
        io.StringIO(json.dumps(payload)),
        sock_path,
        cancel_on_user_action=False,
    )
    assert ok is False
```

- [ ] **Step 6.2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_hook_client.py -k "cancel or PromptSubmit or PostToolUse" -v
```

Expected: FAIL — `forward_event` either ignores `cancel_on_user_action` kwarg or doesn't translate the new event types.

- [ ] **Step 6.3: Extend `_translate_cc_payload` and `forward_event`**

In `src/jarvis_cc/hook_client.py`:

(a) Add a new branch to `_translate_cc_payload`. Place it BEFORE the existing `PreToolUse / AskUserQuestion` branch:

```python
    hook_event = payload.get("hook_event_name")
    if hook_event in ("UserPromptSubmit", "PostToolUse"):
        sid = payload.get("session_id")
        if not sid:
            return None
        return {"command": "cancel", "session_id": sid}
```

(b) Change `forward_event` signature to accept `cancel_on_user_action` and short-circuit when it's False AND the payload is one of the cancel-trigger events. Replace its current top portion with:

```python
def forward_event(
    stream: IO[str],
    sock_path: str | Path,
    *,
    lang_mode: str = "en",
    cancel_on_user_action: bool = True,
) -> bool:
    """Forward an NDJSON event from `stream` to the unix socket at `sock_path`.

    `lang_mode` ("en" | "zh" | "auto") only affects AskUserQuestion translation.
    `cancel_on_user_action`: when False, UserPromptSubmit / PostToolUse hook
    payloads are dropped without contacting the daemon.

    Returns True if successfully sent. Returns False on any failure
    (invalid JSON, socket missing, write error, dropped by policy) —
    never raises.
    """
    sock_path = Path(sock_path)
    try:
        raw = stream.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False

    if not isinstance(payload, dict):
        return False

    if (
        not cancel_on_user_action
        and payload.get("hook_event_name") in ("UserPromptSubmit", "PostToolUse")
    ):
        return False
```

Then the rest of `forward_event` (the `payload = _translate_cc_payload(...)` line and the socket send) stays unchanged.

(c) Update `main()` to pass the flag through:

```python
def main() -> int:
    try:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        mode = getattr(cfg.behavior, "voice_language", "en") or "en"
        cancel = getattr(cfg.behavior, "cancel_on_user_action", True)
        forward_event(
            sys.stdin,
            cfg.paths.socket,
            lang_mode=mode,
            cancel_on_user_action=cancel,
        )
    except Exception:
        pass
    return 0
```

- [ ] **Step 6.4: Run the full hook_client test file**

```
uv run pytest tests/unit/test_hook_client.py -v
```

Expected: all existing tests still pass, plus the 4 new ones (total = old count + 4).

- [ ] **Step 6.5: Commit**

```
git add src/jarvis_cc/hook_client.py tests/unit/test_hook_client.py
git commit -m "feat(hook): translate UserPromptSubmit/PostToolUse into cancel rows"
```

---

## Task 7: Install / uninstall the new hooks

**Files:**
- Modify: `src/jarvis_cc/install.py`
- Test: `tests/unit/test_install.py`

- [ ] **Step 7.1: Write the failing tests**

Append to `tests/unit/test_install.py`:

```python
from jarvis_cc.install import remove_from_claude_settings


def test_merge_settings_registers_userpromptsubmit_and_posttooluse():
    out = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    for hook_type in ("UserPromptSubmit", "PostToolUse"):
        entries = out["hooks"][hook_type]
        assert len(entries) == 1
        assert entries[0]["hooks"][0]["command"] == "jarvis-cc-hook"


def test_merge_settings_idempotent_for_new_hooks():
    out1 = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    out2 = merge_claude_settings(out1, hook_command="jarvis-cc-hook")
    for hook_type in ("UserPromptSubmit", "PostToolUse"):
        entries = out2["hooks"][hook_type]
        count = sum(
            1 for matcher in entries for h in matcher["hooks"]
            if h["command"] == "jarvis-cc-hook"
        )
        assert count == 1


def test_merge_settings_preserves_existing_userpromptsubmit_entries():
    existing = {
        "hooks": {
            "UserPromptSubmit": [
                {"matcher": "", "hooks": [{"type": "command", "command": "other-hook"}]}
            ]
        }
    }
    out = merge_claude_settings(existing, hook_command="jarvis-cc-hook")
    cmds = [
        h["command"]
        for m in out["hooks"]["UserPromptSubmit"]
        for h in m["hooks"]
    ]
    assert "other-hook" in cmds
    assert "jarvis-cc-hook" in cmds


def test_remove_strips_our_userpromptsubmit_and_posttooluse_entries():
    existing = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    out = remove_from_claude_settings(existing, hook_command="jarvis-cc-hook")
    for hook_type in ("UserPromptSubmit", "PostToolUse", "Notification"):
        entries = out.get("hooks", {}).get(hook_type, [])
        cmds = [h["command"] for m in entries for h in m["hooks"]]
        assert "jarvis-cc-hook" not in cmds
```

- [ ] **Step 7.2: Run tests to verify they fail**

```
uv run pytest tests/unit/test_install.py -k "userpromptsubmit or posttooluse or new_hooks or remove_strips" -v
```

Expected: FAIL — the new hook types are not registered yet.

- [ ] **Step 7.3: Update `merge_claude_settings` and `remove_from_claude_settings`**

Edit `src/jarvis_cc/install.py`. Replace `merge_claude_settings` with:

```python
_OUR_HOOK_TYPES = ("Notification", "UserPromptSubmit", "PostToolUse")


def merge_claude_settings(existing: dict, hook_command: str) -> dict:
    """Install our hook entries into Notification, UserPromptSubmit, and
    PostToolUse, replacing any prior jarvis-cc-hook entries in those buckets.
    """
    out = copy.deepcopy(existing)
    hooks = out.setdefault("hooks", {})
    for hook_type in _OUR_HOOK_TYPES:
        entries = hooks.setdefault(hook_type, [])
        pruned: list[dict] = []
        for matcher in entries:
            kept = [h for h in matcher.get("hooks", []) if not _is_our_hook(h)]
            if kept:
                pruned.append({**matcher, "hooks": kept})
        pruned.append(
            {"matcher": "", "hooks": [{"type": "command", "command": hook_command}]}
        )
        out["hooks"][hook_type] = pruned
    return out
```

Replace `remove_from_claude_settings` with:

```python
def remove_from_claude_settings(existing: dict, hook_command: str) -> dict:
    """Strip our jarvis-cc-hook entries from every hook bucket we install into.

    `hook_command` is accepted for signature compat; matching is by basename.
    """
    out = copy.deepcopy(existing)
    hooks = out.get("hooks", {})
    for hook_type in _OUR_HOOK_TYPES:
        entries = hooks.get(hook_type, [])
        filtered = []
        for matcher in entries:
            kept = [h for h in matcher.get("hooks", []) if not _is_our_hook(h)]
            if kept:
                filtered.append({**matcher, "hooks": kept})
        if filtered:
            hooks[hook_type] = filtered
        else:
            hooks.pop(hook_type, None)
    return out
```

- [ ] **Step 7.4: Run install tests**

```
uv run pytest tests/unit/test_install.py -v
```

Expected: all install tests pass (existing + 4 new).

- [ ] **Step 7.5: Commit**

```
git add src/jarvis_cc/install.py tests/unit/test_install.py
git commit -m "feat(install): register UserPromptSubmit + PostToolUse hooks"
```

---

## Task 8: README upgrade note

**Files:**
- Modify: `README.md`

- [ ] **Step 8.1: Add a config row + upgrade note**

In `README.md`, in the default config block (`[behavior]` section, around lines 151-158), append after `phrase_hard_cap = 120`:

```toml
cancel_on_user_action = true   # stop playback when you respond in the originating CC session
```

In the "How it works" diagram (around line 16), add one line under the hook arrow:

```
                                                                                ↳ also: UserPromptSubmit + PostToolUse → cancel
```

In the "Install" section, after the "Now restart any running Claude Code sessions" line (around line 87), add:

```
> **Upgrading from an older install?** Re-run `uv run jarvis-cc install` to
> register the new `UserPromptSubmit` and `PostToolUse` hooks that drive
> "stop voice when I respond" behavior.
```

- [ ] **Step 8.2: Commit**

```
git add README.md
git commit -m "docs: document cancel_on_user_action and upgrade step"
```

---

## Task 9: End-to-end manual verification

This task is non-coded; it confirms the wiring works against the live daemon. Skip steps you've already verified.

- [ ] **Step 9.1: Reinstall**

```
uv run jarvis-cc install
```

Expected output includes "patched ~/.claude/settings.json" and the daemon reloads.

- [ ] **Step 9.2: Inspect `settings.json`**

```
cat ~/.claude/settings.json | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['hooks'], indent=2, ensure_ascii=False))"
```

Expected: `Notification`, `UserPromptSubmit`, and `PostToolUse` each contain an entry pointing to the absolute `jarvis-cc-hook` path.

- [ ] **Step 9.3: Trigger interrupt manually**

Run a synthetic event in one terminal:

```
uv run jarvis-cc test --event permission_prompt --tool Bash
```

Within a second of voice starting, in any CC session, submit any prompt. Expected: voice cuts within ~100 ms of submission. Daemon log (`tail -f ~/.jarvis-cc/logs/daemon.stderr.log`) should show a debug-level "playback cancelled for session …" line and NO traceback.

- [ ] **Step 9.4: Verify cross-session isolation**

Open two CC sessions. Trigger a voice event in session A. Submit a prompt in session B (different session_id). Expected: A's voice continues uninterrupted.

(If a second CC session is awkward to spin up, this can be exercised by hand-crafting two events with different `session_id` via the test CLI plus a manual cancel row sent through `nc -U ~/.jarvis-cc/jarvis.sock` — but the install hook path is the realistic check.)

---

## Self-Review Checklist (already applied)

**Spec coverage:** Every section of `2026-05-16-tts-interrupt-design.md` is mapped to a task — config → T1, queue → T2, player → T3, listener → T4, daemon + worker → T5, hook_client → T6, install → T7, README → T8, E2E → T9.

**Placeholder scan:** No TBD/TODO; every code step shows the actual code.

**Type consistency:** `cancel_session(session_id: str)` matches the `on_cancel: Callable[[str], Awaitable[None]]` parameter in `serve_unix_socket`. `drop_matching` returns `int` and accepts `predicate(Event) -> bool` consistently across queue impl and daemon caller. `on_spawn(proc)` signature is identical in `play`, `play_stream`, and the worker's `_register`.

**Edge cases mapped:** missing session_id (hook drops; listener also drops), early cancel (set + queue drain; `_cancelled_sessions` cleared on next play), ProcessLookupError (T5 has a dedicated test).
