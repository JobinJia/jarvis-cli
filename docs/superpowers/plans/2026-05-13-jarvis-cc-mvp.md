# jarvis-cc MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MVP of `jarvis-cc` — a Python daemon that listens to Claude Code Notification hooks and plays a Jarvis-voiced (zero-shot voice-cloned) short sentence describing each decision-point event, in Chinese or English based on project context.

**Architecture:** A one-shot `hook_client` is invoked by Claude Code and forwards the event JSON over a Unix socket to a long-running `jarvis-daemon` (managed by launchd). The daemon dedups, queues, calls an LLM provider (DeepSeek → Ollama fallback chain) to phrase the event in Jarvis tone, runs local XTTS-v2 zero-shot voice cloning for TTS, and plays the audio via `afplay`. All providers (LLM and TTS) are abstracted behind small ABCs so the chain is configurable in `~/.jarvis-cc/config.toml`.

**Tech Stack:** Python 3.11+, `uv` package manager, `asyncio` (daemon event loop), `httpx` + `respx` (HTTP clients/mocks), `coqui-tts` (XTTS-v2), `langdetect` (zh/en detection), `tomllib` (stdlib config), `pytest` + `pytest-asyncio` (tests), `loguru` (logging), macOS `launchd` (daemon supervision), `afplay` (audio playback).

**Branch & flow:** Work on a fresh `feat/mvp` branch off `main`. Each task ends with a commit. PR back to `main` only when the smoke test at the end passes.

**Spec reference:** `docs/superpowers/specs/2026-05-13-jarvis-cc-design.md`

---

## Pre-flight

Run once before Task 1:

```bash
cd ~/myself/jarvis-cc
git checkout -b feat/mvp
```

---

## Task 1: Bootstrap project (pyproject, src layout, Event type)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/jarvis_cc/__init__.py`
- Create: `src/jarvis_cc/types.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_types.py`

- [ ] **Step 1: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.coverage
.DS_Store
~/.jarvis-cc/      # operator runtime dir (never commit)
*.wav
*.mp3
build/
dist/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "jarvis-cc"
version = "0.1.0"
description = "Jarvis-voiced notification layer for Claude Code"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Jobin" }]
dependencies = [
    "httpx>=0.27",
    "loguru>=0.7",
    "langdetect>=1.0.9",
    "coqui-tts>=0.24",
    "anthropic>=0.40",
    "openai>=1.50",
    "elevenlabs>=1.10",
]

[project.scripts]
jarvis-cc = "jarvis_cc.install:main"
jarvis-cc-hook = "jarvis_cc.hook_client:main"
jarvis-cc-daemon = "jarvis_cc.daemon.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jarvis_cc"]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "respx>=0.21",
    "ruff>=0.7",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Write minimal `README.md`**

```markdown
# jarvis-cc

Jarvis-voiced notification layer for Claude Code. See [docs/superpowers/specs/2026-05-13-jarvis-cc-design.md](docs/superpowers/specs/2026-05-13-jarvis-cc-design.md).

## Install

\`\`\`bash
uv sync
uv run jarvis-cc install
\`\`\`
```

- [ ] **Step 4: Write `src/jarvis_cc/__init__.py`**

```python
"""jarvis-cc: Jarvis-voiced notification layer for Claude Code."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Write `src/jarvis_cc/types.py`**

```python
"""Shared dataclasses used across hook_client, daemon, phrase, tts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NotificationType = Literal[
    "permission_prompt",
    "idle_prompt",
    "elicitation_dialog",
]

Lang = Literal["zh", "en"]


@dataclass(frozen=True)
class Event:
    """A single notification event from Claude Code, normalized for daemon use."""

    notification_type: NotificationType
    tool_name: str | None
    tool_input: dict[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    session_id: str | None = None
    raw_message: str | None = None
    received_at: float = 0.0  # epoch seconds; filled by listener

    def dedup_key(self) -> str:
        """Hash key for dedup window: same (cwd, type, tool) collapses."""
        return f"{self.cwd or ''}::{self.notification_type}::{self.tool_name or ''}"
```

- [ ] **Step 6: Write test `tests/unit/test_types.py`**

```python
from jarvis_cc.types import Event


def test_dedup_key_combines_cwd_type_tool():
    e = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        cwd="/x/y",
    )
    assert e.dedup_key() == "/x/y::permission_prompt::Bash"


def test_dedup_key_handles_none_fields():
    e = Event(notification_type="idle_prompt", tool_name=None, cwd=None)
    assert e.dedup_key() == "::idle_prompt::"


def test_dedup_key_differs_by_cwd():
    a = Event(notification_type="permission_prompt", tool_name="Bash", cwd="/a")
    b = Event(notification_type="permission_prompt", tool_name="Bash", cwd="/b")
    assert a.dedup_key() != b.dedup_key()
```

- [ ] **Step 7: Install and run tests**

Run:
```bash
uv sync
uv run pytest tests/unit/test_types.py -v
```
Expected: 3 PASSED.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore README.md src tests
git commit -m "feat: bootstrap project with Event dataclass"
```

---

## Task 2: Config loader (`config.py`)

**Files:**
- Create: `src/jarvis_cc/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write test `tests/unit/test_config.py`**

```python
from pathlib import Path

import pytest

from jarvis_cc.config import Config, load_config


def test_load_config_returns_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.llm.provider == "deepseek"
    assert cfg.llm.fallback == "ollama"
    assert cfg.tts.provider == "xtts"
    assert cfg.behavior.dedup_window_seconds == 10
    assert cfg.behavior.queue_max_size == 5
    assert cfg.behavior.voice_language == "auto"
    assert cfg.behavior.events == [
        "permission_prompt",
        "idle_prompt",
        "elicitation_dialog",
    ]


def test_load_config_reads_toml(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(
        """
[llm]
provider = "anthropic"
fallback = "deepseek"

[behavior]
dedup_window_seconds = 30
queue_max_size = 9
"""
    )
    cfg = load_config(p)
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.fallback == "deepseek"
    assert cfg.behavior.dedup_window_seconds == 30
    assert cfg.behavior.queue_max_size == 9
    # Untouched fields stay default
    assert cfg.tts.provider == "xtts"


def test_load_config_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.paths.socket.startswith(str(tmp_path))
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: ImportError (module not yet defined).

- [ ] **Step 3: Implement `src/jarvis_cc/config.py`**

```python
"""TOML-backed config with safe defaults. Layered: file > defaults."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 5.0


@dataclass
class AnthropicConfig:
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-haiku-4-5-20251001"
    timeout_seconds: float = 5.0


@dataclass
class OpenAIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 5.0


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout_seconds: float = 10.0


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    fallback: str = "ollama"
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class XTTSConfig:
    model_dir: str = "~/.jarvis-cc/models/xtts-v2"
    ref_audio_zh: str = "~/.jarvis-cc/voices/jarvis_zh.wav"
    ref_audio_en: str = "~/.jarvis-cc/voices/jarvis_en.wav"
    device: str = "mps"


@dataclass
class ElevenLabsConfig:
    api_key_env: str = "ELEVENLABS_API_KEY"
    voice_id: str = ""
    model: str = "eleven_turbo_v2_5"


@dataclass
class TTSConfig:
    provider: str = "xtts"
    fallback: str = "say"
    xtts: XTTSConfig = field(default_factory=XTTSConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)


@dataclass
class BehaviorConfig:
    dedup_window_seconds: int = 10
    queue_max_size: int = 5
    voice_language: str = "auto"
    events: list[str] = field(
        default_factory=lambda: [
            "permission_prompt",
            "idle_prompt",
            "elicitation_dialog",
        ]
    )
    phrase_max_chars: int = 30


@dataclass
class PathsConfig:
    socket: str = "~/.jarvis-cc/jarvis.sock"
    log: str = "~/.jarvis-cc/daemon.log"
    missed_log: str = "~/.jarvis-cc/missed.log"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def expanduser(p: str) -> str:
    return str(Path(os.path.expanduser(p)))


def _merge(dst, src: dict) -> None:
    """Shallow-recursive merge: dict src into dataclass dst, mutating dst."""
    for key, val in src.items():
        if not hasattr(dst, key):
            continue
        cur = getattr(dst, key)
        if isinstance(val, dict) and hasattr(cur, "__dataclass_fields__"):
            _merge(cur, val)
        else:
            setattr(dst, key, val)


def load_config(path: str | Path) -> Config:
    """Load TOML config from `path`, fall back to defaults if missing/empty.

    Path-like values under `[paths]` and `[tts.xtts]` get `~` expanded.
    """
    cfg = Config()
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        _merge(cfg, data)

    cfg.paths.socket = expanduser(cfg.paths.socket)
    cfg.paths.log = expanduser(cfg.paths.log)
    cfg.paths.missed_log = expanduser(cfg.paths.missed_log)
    cfg.tts.xtts.model_dir = expanduser(cfg.tts.xtts.model_dir)
    cfg.tts.xtts.ref_audio_zh = expanduser(cfg.tts.xtts.ref_audio_zh)
    cfg.tts.xtts.ref_audio_en = expanduser(cfg.tts.xtts.ref_audio_en)
    return cfg


DEFAULT_CONFIG_PATH = expanduser("~/.jarvis-cc/config.toml")
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/config.py tests/unit/test_config.py
git commit -m "feat(config): typed TOML config loader with defaults"
```

---

## Task 3: Hook client (`hook_client.py`)

**Files:**
- Create: `src/jarvis_cc/hook_client.py`
- Create: `tests/unit/test_hook_client.py`

- [ ] **Step 1: Write test `tests/unit/test_hook_client.py`**

```python
import io
import json
import socket
import threading
from pathlib import Path

from jarvis_cc.hook_client import forward_event


def _start_unix_echo_server(path: Path) -> list[bytes]:
    received: list[bytes] = []
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(1)

    def _serve():
        conn, _ = sock.accept()
        received.append(conn.recv(4096))
        conn.close()
        sock.close()

    threading.Thread(target=_serve, daemon=True).start()
    return received


def test_forward_event_writes_ndjson_to_socket(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)

    payload = {
        "session_id": "abc",
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)

    assert ok is True
    # Allow the server thread to write `received`
    for _ in range(50):
        if received:
            break
        import time
        time.sleep(0.01)
    assert len(received) == 1
    line = received[0].decode().strip()
    assert json.loads(line)["tool_name"] == "Bash"


def test_forward_event_returns_false_when_socket_missing(tmp_path: Path):
    sock_path = tmp_path / "does-not-exist.sock"
    ok = forward_event(io.StringIO('{"notification_type":"idle_prompt"}'), sock_path)
    assert ok is False


def test_forward_event_handles_invalid_json(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    ok = forward_event(io.StringIO("not-json"), sock_path)
    assert ok is False
```

- [ ] **Step 2: Run test (expect ImportError)**

Run: `uv run pytest tests/unit/test_hook_client.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/jarvis_cc/hook_client.py`**

```python
"""Thin client invoked by Claude Code Notification hook.

Reads JSON payload from stdin, writes a single NDJSON line over the
configured Unix socket, and exits. Must never raise to stdout — Claude
Code reads stdout for hook decisions.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import IO

from .config import DEFAULT_CONFIG_PATH, load_config


def forward_event(stream: IO[str], sock_path: str | Path) -> bool:
    """Forward an NDJSON event from `stream` to the unix socket at `sock_path`.

    Returns True if successfully sent. Returns False on any failure
    (invalid JSON, socket missing, write error) — never raises.
    """
    sock_path = Path(sock_path)
    try:
        raw = stream.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False

    if not isinstance(payload, dict):
        return False

    payload["_received_at"] = time.time()
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(str(sock_path))
        s.sendall(line)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def main() -> int:
    """Entry point registered as `jarvis-cc-hook` console_script.

    Must NEVER raise — Claude Code reads stdout for hook decisions and a
    traceback would corrupt that channel. All failures are silent and
    exit 0.
    """
    try:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        forward_event(sys.stdin, cfg.paths.socket)
    except Exception:  # noqa: BLE001 — structural guarantee
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test (expect PASS)**

Run: `uv run pytest tests/unit/test_hook_client.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/hook_client.py tests/unit/test_hook_client.py
git commit -m "feat(hook): one-shot hook_client forwarding stdin JSON to unix socket"
```

---

## Task 4: Daemon listener (`daemon/listener.py`)

**Files:**
- Create: `src/jarvis_cc/daemon/__init__.py`
- Create: `src/jarvis_cc/daemon/listener.py`
- Create: `tests/unit/test_listener.py`

- [ ] **Step 1: Write test `tests/unit/test_listener.py`**

```python
import asyncio
import json
import socket
from pathlib import Path

import pytest

from jarvis_cc.daemon.listener import parse_payload, serve_unix_socket
from jarvis_cc.types import Event


def test_parse_payload_maps_known_fields():
    raw = json.dumps(
        {
            "notification_type": "permission_prompt",
            "tool_name": "Bash",
            "tool_input": {"command": "rm foo"},
            "cwd": "/x/y",
            "session_id": "s1",
            "_received_at": 12345.67,
        }
    )
    ev = parse_payload(raw)
    assert isinstance(ev, Event)
    assert ev.notification_type == "permission_prompt"
    assert ev.tool_name == "Bash"
    assert ev.tool_input == {"command": "rm foo"}
    assert ev.cwd == "/x/y"
    assert ev.received_at == 12345.67


def test_parse_payload_returns_none_for_unknown_type():
    raw = json.dumps({"notification_type": "made_up_event", "tool_name": "X"})
    assert parse_payload(raw) is None


def test_parse_payload_returns_none_for_malformed_json():
    assert parse_payload("not json") is None


@pytest.mark.asyncio
async def test_serve_unix_socket_yields_events(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received: list[Event] = []

    async def collect(ev: Event):
        received.append(ev)

    server_task = asyncio.create_task(serve_unix_socket(sock_path, collect))
    # Wait for socket to exist
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall(
        (json.dumps({"notification_type": "idle_prompt", "tool_name": None}) + "\n").encode()
    )
    s.close()

    await asyncio.sleep(0.1)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    assert len(received) == 1
    assert received[0].notification_type == "idle_prompt"
```

- [ ] **Step 2: Run test (expect ImportError)**

Run: `uv run pytest tests/unit/test_listener.py -v`

- [ ] **Step 3: Write `src/jarvis_cc/daemon/__init__.py`**

```python
"""Daemon package: long-running asyncio loop for jarvis-cc."""
```

- [ ] **Step 4: Implement `src/jarvis_cc/daemon/listener.py`**

```python
"""Unix-socket listener: accepts NDJSON lines, normalizes into Event."""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import get_args

from loguru import logger

from ..types import Event, NotificationType

_ALLOWED_TYPES: set[str] = set(get_args(NotificationType))


def parse_payload(raw: str) -> Event | None:
    """Parse a single NDJSON line into a normalized Event, or None on bad data."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    ntype = data.get("notification_type")
    if ntype not in _ALLOWED_TYPES:
        return None
    return Event(
        notification_type=ntype,
        tool_name=data.get("tool_name"),
        tool_input=data.get("tool_input") or {},
        cwd=data.get("cwd"),
        session_id=data.get("session_id"),
        raw_message=data.get("message") or data.get("raw_message"),
        received_at=float(data.get("_received_at", time.time())),
    )


async def serve_unix_socket(
    sock_path: Path,
    on_event: Callable[[Event], Awaitable[None]],
) -> None:
    """Run a unix-socket server forever, dispatching parsed events to `on_event`."""
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
    async with server:
        await server.serve_forever()
```

- [ ] **Step 5: Run test (expect PASS)**

Run: `uv run pytest tests/unit/test_listener.py -v`
Expected: 4 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/jarvis_cc/daemon tests/unit/test_listener.py
git commit -m "feat(daemon): unix-socket listener that yields normalized Events"
```

---

## Task 5: Dedup (`daemon/dedup.py`)

**Files:**
- Create: `src/jarvis_cc/daemon/dedup.py`
- Create: `tests/unit/test_dedup.py`

- [ ] **Step 1: Write test `tests/unit/test_dedup.py`**

```python
from jarvis_cc.daemon.dedup import DedupWindow
from jarvis_cc.types import Event


def _ev(t: float, tool: str = "Bash", cwd: str = "/x") -> Event:
    return Event(
        notification_type="permission_prompt",
        tool_name=tool,
        cwd=cwd,
        received_at=t,
    )


def test_first_event_is_not_duplicate():
    w = DedupWindow(window_seconds=10)
    assert w.is_duplicate(_ev(0.0)) is False


def test_same_key_within_window_is_duplicate():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0))
    assert w.is_duplicate(_ev(5.0)) is True


def test_same_key_outside_window_is_not_duplicate():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0))
    assert w.is_duplicate(_ev(10.1)) is False


def test_different_keys_are_independent():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0, tool="Bash"))
    assert w.is_duplicate(_ev(1.0, tool="Edit")) is False
    assert w.is_duplicate(_ev(1.0, tool="Bash", cwd="/y")) is False


def test_is_duplicate_updates_last_seen():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0))
    w.is_duplicate(_ev(5.0))  # dup → last_seen slid to 5.0; window now expires at 15.0
    # 14.9 is still inside the slid window → dup; this slides last_seen to 14.9
    assert w.is_duplicate(_ev(14.9)) is True
    # 25.1 - 14.9 = 10.2 > 10 → outside window from the freshly-slid anchor → not dup
    assert w.is_duplicate(_ev(25.1)) is False
```

- [ ] **Step 2: Run test (expect ImportError)**

Run: `uv run pytest tests/unit/test_dedup.py -v`

- [ ] **Step 3: Implement `src/jarvis_cc/daemon/dedup.py`**

```python
"""Sliding-window dedup keyed by Event.dedup_key()."""
from __future__ import annotations

from ..types import Event


class DedupWindow:
    """In-memory dedup: same dedup_key within `window_seconds` is suppressed.

    The "last seen" timestamp slides forward on each duplicate hit, so a stream
    of identical events within the window stays suppressed until silence.
    """

    def __init__(self, window_seconds: int) -> None:
        self._window = float(window_seconds)
        self._last_seen: dict[str, float] = {}

    def is_duplicate(self, event: Event) -> bool:
        key = event.dedup_key()
        now = event.received_at
        prev = self._last_seen.get(key)
        if prev is not None and (now - prev) <= self._window:
            self._last_seen[key] = now
            return True
        self._last_seen[key] = now
        return False
```

- [ ] **Step 4: Run test (expect PASS)**

Run: `uv run pytest tests/unit/test_dedup.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/daemon/dedup.py tests/unit/test_dedup.py
git commit -m "feat(daemon): sliding-window dedup keyed by (cwd, type, tool)"
```

---

## Task 6: Bounded queue + worker skeleton (`daemon/queue.py`)

**Files:**
- Create: `src/jarvis_cc/daemon/queue.py`
- Create: `tests/unit/test_queue.py`

- [ ] **Step 1: Write test `tests/unit/test_queue.py`**

```python
import asyncio

import pytest

from jarvis_cc.daemon.queue import BoundedEventQueue
from jarvis_cc.types import Event


def _ev(i: int) -> Event:
    return Event(notification_type="idle_prompt", tool_name=f"T{i}", cwd=f"/{i}")


@pytest.mark.asyncio
async def test_queue_preserves_order():
    q = BoundedEventQueue(maxsize=10)
    await q.put_or_drop(_ev(1))
    await q.put_or_drop(_ev(2))
    a = await q.get()
    b = await q.get()
    assert a.tool_name == "T1"
    assert b.tool_name == "T2"


@pytest.mark.asyncio
async def test_queue_drops_oldest_when_full():
    q = BoundedEventQueue(maxsize=2)
    for i in range(5):
        await q.put_or_drop(_ev(i))
    # Only last 2 survive
    a = await q.get()
    b = await q.get()
    assert a.tool_name == "T3"
    assert b.tool_name == "T4"
    assert q.dropped_count == 3
```

- [ ] **Step 2: Run test (expect ImportError)**

Run: `uv run pytest tests/unit/test_queue.py -v`

- [ ] **Step 3: Implement `src/jarvis_cc/daemon/queue.py`**

```python
"""Bounded FIFO event queue with drop-oldest semantics."""
from __future__ import annotations

import asyncio
from collections import deque

from loguru import logger

from ..types import Event


class BoundedEventQueue:
    """asyncio-compatible bounded FIFO. When full, drops oldest events."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._deque: deque[Event] = deque(maxlen=maxsize)
        self._cond = asyncio.Condition()
        self.dropped_count = 0

    async def put_or_drop(self, event: Event) -> None:
        async with self._cond:
            if len(self._deque) == self._maxsize:
                dropped = self._deque.popleft()
                self.dropped_count += 1
                logger.warning("Queue full, dropped event {}", dropped.dedup_key())
            self._deque.append(event)
            self._cond.notify()

    async def get(self) -> Event:
        async with self._cond:
            while not self._deque:
                await self._cond.wait()
            return self._deque.popleft()
```

- [ ] **Step 4: Run test (expect PASS)**

Run: `uv run pytest tests/unit/test_queue.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/daemon/queue.py tests/unit/test_queue.py
git commit -m "feat(daemon): bounded drop-oldest event queue"
```

---

## Task 7: Language detection (`phrase/language.py`)

**Files:**
- Create: `src/jarvis_cc/phrase/__init__.py`
- Create: `src/jarvis_cc/phrase/language.py`
- Create: `tests/unit/test_language.py`

- [ ] **Step 1: Write test `tests/unit/test_language.py`**

```python
from pathlib import Path

from jarvis_cc.phrase.language import detect_for


def test_detect_returns_zh_when_cwd_has_chinese_claude_md(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text(
        "本项目是一个使用 Valaxy 的中文博客。" * 10
    )
    assert detect_for(str(tmp_path)) == "zh"


def test_detect_returns_en_when_cwd_has_english_claude_md(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text(
        "This project is a personal blog built with Valaxy." * 10
    )
    assert detect_for(str(tmp_path)) == "en"


def test_detect_falls_back_to_readme_when_no_claude_md(tmp_path: Path):
    (tmp_path / "README.md").write_text("中文项目说明" * 10)
    assert detect_for(str(tmp_path)) == "zh"


def test_detect_defaults_to_zh_when_no_signal(tmp_path: Path):
    # Empty dir → assume the user (a Chinese speaker per spec) wants zh
    assert detect_for(str(tmp_path)) == "zh"


def test_detect_handles_none_cwd():
    assert detect_for(None) == "zh"
```

- [ ] **Step 2: Run test (expect ImportError)**

Run: `uv run pytest tests/unit/test_language.py -v`

- [ ] **Step 3: Write `src/jarvis_cc/phrase/__init__.py`**

```python
"""Phrase generation: language detection + LLM router + prompt + templates."""
```

- [ ] **Step 4: Implement `src/jarvis_cc/phrase/language.py`**

```python
"""Project-language detection: looks at CLAUDE.md, README.md in cwd."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from loguru import logger

from ..types import Lang

DetectorFactory.seed = 0  # deterministic

_SOURCES = ["CLAUDE.md", "AGENTS.md", "README.md", "README"]
_DEFAULT: Lang = "zh"
_SAMPLE_CHARS = 500


def _read_sample(cwd: Path) -> str | None:
    for name in _SOURCES:
        p = cwd / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")[:_SAMPLE_CHARS]
            except OSError:
                continue
    return None


@lru_cache(maxsize=64)
def detect_for(cwd: str | None) -> Lang:
    """Return 'zh' or 'en' for the project at `cwd`.

    Default 'zh' (user's primary language) when no signal can be extracted.
    """
    if not cwd:
        return _DEFAULT
    try:
        sample = _read_sample(Path(cwd))
    except OSError:
        return _DEFAULT
    if not sample or not sample.strip():
        return _DEFAULT
    try:
        code = detect(sample)
    except LangDetectException:
        return _DEFAULT
    if code.startswith("zh"):
        return "zh"
    if code.startswith("en"):
        return "en"
    logger.debug("Unmapped langdetect result {!r}, falling back to {}", code, _DEFAULT)
    return _DEFAULT
```

- [ ] **Step 5: Run test (expect PASS)**

Run: `uv run pytest tests/unit/test_language.py -v`
Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/jarvis_cc/phrase/__init__.py src/jarvis_cc/phrase/language.py tests/unit/test_language.py
git commit -m "feat(phrase): cwd-based zh/en language detection"
```

---

## Task 8: Phrase templates + prompt (`phrase/templates.py`, `phrase/prompt.py`)

**Files:**
- Create: `src/jarvis_cc/phrase/templates.py`
- Create: `src/jarvis_cc/phrase/prompt.py`
- Create: `tests/unit/test_templates.py`
- Create: `tests/unit/test_prompt.py`

- [ ] **Step 1: Write test `tests/unit/test_templates.py`**

```python
from jarvis_cc.phrase.templates import render_template
from jarvis_cc.types import Event


def test_permission_prompt_zh():
    ev = Event(notification_type="permission_prompt", tool_name="Bash")
    text = render_template(ev, lang="zh")
    assert "先生" in text
    assert "Bash" in text


def test_permission_prompt_en():
    ev = Event(notification_type="permission_prompt", tool_name="Edit")
    text = render_template(ev, lang="en")
    assert text.startswith("Sir")
    assert "Edit" in text


def test_idle_prompt_zh():
    ev = Event(notification_type="idle_prompt", tool_name=None)
    text = render_template(ev, lang="zh")
    assert "先生" in text


def test_elicitation_dialog_en():
    ev = Event(notification_type="elicitation_dialog", tool_name=None)
    text = render_template(ev, lang="en")
    assert "Sir" in text
```

- [ ] **Step 2: Implement `src/jarvis_cc/phrase/templates.py`**

```python
"""Fallback phrase templates when all LLM providers fail.

Tone tries to approximate Jarvis: polite, brief, with "Sir"/"先生".
"""
from __future__ import annotations

from ..types import Event, Lang

_ZH: dict[str, str] = {
    "permission_prompt": "先生，Claude 请求使用 {tool} 的权限。",
    "idle_prompt": "先生，Claude 正在等候您的指示。",
    "elicitation_dialog": "先生，有个对话框等您填写。",
}

_EN: dict[str, str] = {
    "permission_prompt": "Sir, Claude requests permission for {tool}.",
    "idle_prompt": "Sir, Claude awaits your guidance.",
    "elicitation_dialog": "Sir, a dialog awaits your input.",
}


def render_template(event: Event, lang: Lang) -> str:
    table = _ZH if lang == "zh" else _EN
    tmpl = table.get(event.notification_type, table["idle_prompt"])
    return tmpl.format(tool=event.tool_name or "something")
```

- [ ] **Step 3: Write test `tests/unit/test_prompt.py`**

```python
from jarvis_cc.phrase.prompt import build_messages
from jarvis_cc.types import Event


def test_build_messages_zh_includes_jarvis_system_and_event():
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "rm foo"},
        cwd="/x",
    )
    msgs = build_messages(ev, lang="zh", max_chars=30)
    assert msgs[0]["role"] == "system"
    assert "J.A.R.V.I.S" in msgs[0]["content"] or "管家" in msgs[0]["content"]
    assert "中文" in msgs[0]["content"]
    assert "30" in msgs[0]["content"]
    assert msgs[-1]["role"] == "user"
    assert "permission_prompt" in msgs[-1]["content"]
    assert "Bash" in msgs[-1]["content"]


def test_build_messages_en_swaps_language_clause():
    ev = Event(notification_type="idle_prompt", tool_name=None)
    msgs = build_messages(ev, lang="en", max_chars=30)
    assert "English" in msgs[0]["content"]
```

- [ ] **Step 4: Implement `src/jarvis_cc/phrase/prompt.py`**

```python
"""Jarvis-tone prompt builder shared across all LLM providers."""
from __future__ import annotations

import json

from ..types import Event, Lang

_SYSTEM_BASE = (
    "You are J.A.R.V.I.S., Tony Stark's polite British AI butler. "
    "Address the user as '{addr}'. "
    "Given a structured Claude Code event, reply with a single short sentence "
    "in {lang_name} (at most {max_chars} characters), notifying the user of "
    "what needs their decision. Be calm, courteous, with a hint of dry wit. "
    "Do NOT explain reasoning. Do NOT add quotes or labels. Output the sentence only."
)

_FEW_SHOT_ZH = [
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash"}'},
    {"role": "assistant", "content": "先生，Claude 想动用一下终端，请您过目。"},
    {"role": "user", "content": '{"notification_type":"idle_prompt","tool_name":null}'},
    {"role": "assistant", "content": "先生，Claude 静候您的吩咐。"},
]

_FEW_SHOT_EN = [
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash"}'},
    {"role": "assistant", "content": "Sir, Claude requests the shell, at your discretion."},
    {"role": "user", "content": '{"notification_type":"idle_prompt","tool_name":null}'},
    {"role": "assistant", "content": "Sir, Claude awaits your guidance."},
]


def build_messages(event: Event, lang: Lang, max_chars: int) -> list[dict[str, str]]:
    """Return the OpenAI-compatible chat messages for an Event."""
    if lang == "zh":
        sys = _SYSTEM_BASE.format(addr="先生", lang_name="中文", max_chars=max_chars)
        few_shot = _FEW_SHOT_ZH
    else:
        sys = _SYSTEM_BASE.format(addr="Sir", lang_name="English", max_chars=max_chars)
        few_shot = _FEW_SHOT_EN

    user_blob = json.dumps(
        {
            "notification_type": event.notification_type,
            "tool_name": event.tool_name,
            "tool_input": event.tool_input,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": sys}, *few_shot, {"role": "user", "content": user_blob}]
```

- [ ] **Step 5: Run tests**

Run:
```bash
uv run pytest tests/unit/test_templates.py tests/unit/test_prompt.py -v
```
Expected: 4 + 2 = 6 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/jarvis_cc/phrase/templates.py src/jarvis_cc/phrase/prompt.py \
        tests/unit/test_templates.py tests/unit/test_prompt.py
git commit -m "feat(phrase): jarvis tone prompt builder + fallback templates"
```

---

## Task 9: Phrase provider base + DeepSeek + Ollama

**Files:**
- Create: `src/jarvis_cc/phrase/providers/__init__.py`
- Create: `src/jarvis_cc/phrase/providers/base.py`
- Create: `src/jarvis_cc/phrase/providers/deepseek.py`
- Create: `src/jarvis_cc/phrase/providers/ollama.py`
- Create: `tests/unit/test_phrase_providers.py`

- [ ] **Step 1: Write test `tests/unit/test_phrase_providers.py`**

```python
import os

import httpx
import pytest
import respx

from jarvis_cc.config import DeepSeekConfig, OllamaConfig
from jarvis_cc.phrase.providers.deepseek import DeepSeekProvider
from jarvis_cc.phrase.providers.ollama import OllamaProvider
from jarvis_cc.types import Event


def _ev() -> Event:
    return Event(notification_type="permission_prompt", tool_name="Bash")


@pytest.mark.asyncio
async def test_deepseek_returns_assistant_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = DeepSeekConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "先生，Claude 请求许可。"}}
                ]
            },
        )
        p = DeepSeekProvider(cfg)
        out = await p.generate(_ev(), lang="zh", max_chars=30)
    assert out == "先生，Claude 请求许可。"


@pytest.mark.asyncio
async def test_deepseek_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = DeepSeekProvider(DeepSeekConfig())
    with pytest.raises(RuntimeError):
        await p.generate(_ev(), lang="zh", max_chars=30)


@pytest.mark.asyncio
async def test_deepseek_raises_on_http_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = DeepSeekConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(500)
        p = DeepSeekProvider(cfg)
        with pytest.raises(httpx.HTTPStatusError):
            await p.generate(_ev(), lang="zh", max_chars=30)


@pytest.mark.asyncio
async def test_ollama_returns_assistant_text():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200,
            json={"message": {"role": "assistant", "content": "先生，请过目。"}},
        )
        p = OllamaProvider(cfg)
        out = await p.generate(_ev(), lang="zh", max_chars=30)
    assert out == "先生，请过目。"


@pytest.mark.asyncio
async def test_ollama_raises_on_connection_error():
    cfg = OllamaConfig(base_url="http://127.0.0.1:1")  # nothing listens here
    p = OllamaProvider(cfg)
    with pytest.raises(httpx.HTTPError):
        await p.generate(_ev(), lang="zh", max_chars=30)
```

- [ ] **Step 2: Write `src/jarvis_cc/phrase/providers/__init__.py`**

```python
"""LLM provider implementations behind a shared ABC."""
```

- [ ] **Step 3: Implement `src/jarvis_cc/phrase/providers/base.py`**

```python
"""Abstract base for LLM phrase providers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ...types import Event, Lang


class PhraseProvider(ABC):
    """A provider returns a single Jarvis-tone sentence for an Event."""

    name: str

    @abstractmethod
    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str: ...

    async def healthcheck(self) -> bool:
        """Return True if this provider is likely to succeed right now."""
        return True
```

- [ ] **Step 4: Implement `src/jarvis_cc/phrase/providers/deepseek.py`**

```python
"""DeepSeek-Chat provider (OpenAI-compatible chat API)."""
from __future__ import annotations

import os

import httpx

from ...config import DeepSeekConfig
from ...types import Event, Lang
from ..prompt import build_messages
from .base import PhraseProvider


class DeepSeekProvider(PhraseProvider):
    name = "deepseek"

    def __init__(self, cfg: DeepSeekConfig) -> None:
        self.cfg = cfg

    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        messages = build_messages(event, lang, max_chars)
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 80,
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    async def healthcheck(self) -> bool:
        return bool(os.getenv(self.cfg.api_key_env))
```

- [ ] **Step 5: Implement `src/jarvis_cc/phrase/providers/ollama.py`**

```python
"""Ollama local LLM provider (uses /api/chat)."""
from __future__ import annotations

import httpx

from ...config import OllamaConfig
from ...types import Event, Lang
from ..prompt import build_messages
from .base import PhraseProvider


class OllamaProvider(PhraseProvider):
    name = "ollama"

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str:
        messages = build_messages(event, lang, max_chars)
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 80},
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["message"]["content"].strip()

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.cfg.base_url, timeout=1.0) as c:
                r = await c.get("/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_phrase_providers.py -v`
Expected: 5 PASSED.

- [ ] **Step 7: Commit**

```bash
git add src/jarvis_cc/phrase/providers tests/unit/test_phrase_providers.py
git commit -m "feat(phrase): DeepSeek + Ollama providers with shared ABC"
```

---

## Task 10: Anthropic + OpenAI providers

**Files:**
- Create: `src/jarvis_cc/phrase/providers/anthropic.py`
- Create: `src/jarvis_cc/phrase/providers/openai.py`
- Create: `tests/unit/test_phrase_providers_more.py`

- [ ] **Step 1: Write test `tests/unit/test_phrase_providers_more.py`**

```python
import pytest
import respx

from jarvis_cc.config import AnthropicConfig, OpenAIConfig
from jarvis_cc.phrase.providers.anthropic import AnthropicProvider
from jarvis_cc.phrase.providers.openai import OpenAIProvider
from jarvis_cc.types import Event


def _ev() -> Event:
    return Event(notification_type="permission_prompt", tool_name="Bash")


@pytest.mark.asyncio
async def test_anthropic_returns_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = AnthropicConfig()
    with respx.mock(base_url="https://api.anthropic.com") as router:
        router.post("/v1/messages").respond(
            200,
            json={
                "id": "m_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Sir, your shell awaits."}],
                "model": cfg.model,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        p = AnthropicProvider(cfg)
        out = await p.generate(_ev(), lang="en", max_chars=30)
    assert out == "Sir, your shell awaits."


@pytest.mark.asyncio
async def test_openai_returns_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    cfg = OpenAIConfig()
    with respx.mock(base_url="https://api.openai.com") as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [{"message": {"content": "Sir, at your discretion."}}],
            },
        )
        p = OpenAIProvider(cfg)
        out = await p.generate(_ev(), lang="en", max_chars=30)
    assert out == "Sir, at your discretion."
```

- [ ] **Step 2: Implement `src/jarvis_cc/phrase/providers/anthropic.py`**

```python
"""Anthropic Claude provider via raw HTTP (avoids SDK pinning issues)."""
from __future__ import annotations

import os

import httpx

from ...config import AnthropicConfig
from ...types import Event, Lang
from ..prompt import build_messages
from .base import PhraseProvider


class AnthropicProvider(PhraseProvider):
    name = "anthropic"

    def __init__(self, cfg: AnthropicConfig) -> None:
        self.cfg = cfg

    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        messages = build_messages(event, lang, max_chars)
        # Anthropic API splits "system" out of messages
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        async with httpx.AsyncClient(
            base_url="https://api.anthropic.com", timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "system": system_msg,
                    "messages": chat,
                    "max_tokens": 120,
                    "temperature": 0.7,
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["content"][0]["text"].strip()

    async def healthcheck(self) -> bool:
        return bool(os.getenv(self.cfg.api_key_env))
```

- [ ] **Step 3: Implement `src/jarvis_cc/phrase/providers/openai.py`**

```python
"""OpenAI chat-completions provider."""
from __future__ import annotations

import os

import httpx

from ...config import OpenAIConfig
from ...types import Event, Lang
from ..prompt import build_messages
from .base import PhraseProvider


class OpenAIProvider(PhraseProvider):
    name = "openai"

    def __init__(self, cfg: OpenAIConfig) -> None:
        self.cfg = cfg

    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        messages = build_messages(event, lang, max_chars)
        async with httpx.AsyncClient(
            base_url="https://api.openai.com", timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 80,
                },
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    async def healthcheck(self) -> bool:
        return bool(os.getenv(self.cfg.api_key_env))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_phrase_providers_more.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/phrase/providers/anthropic.py src/jarvis_cc/phrase/providers/openai.py tests/unit/test_phrase_providers_more.py
git commit -m "feat(phrase): Anthropic + OpenAI providers"
```

---

## Task 11: Phrase router (`phrase/router.py`)

**Files:**
- Create: `src/jarvis_cc/phrase/router.py`
- Create: `tests/unit/test_router.py`

- [ ] **Step 1: Write test `tests/unit/test_router.py`**

```python
from typing import Literal

import pytest

from jarvis_cc.config import Config
from jarvis_cc.phrase.providers.base import PhraseProvider
from jarvis_cc.phrase.router import PhraseRouter
from jarvis_cc.types import Event


class _Stub(PhraseProvider):
    def __init__(self, name: str, mode: Literal["ok", "fail"]) -> None:
        self.name = name
        self.mode = mode
        self.calls = 0

    async def generate(self, event, lang, max_chars):
        self.calls += 1
        if self.mode == "fail":
            raise RuntimeError(f"{self.name} down")
        return f"<{self.name}>"


def _ev() -> Event:
    return Event(notification_type="idle_prompt", tool_name=None)


@pytest.mark.asyncio
async def test_router_returns_primary_when_healthy():
    primary, fallback = _Stub("p", "ok"), _Stub("f", "ok")
    router = PhraseRouter(primary, fallback, Config())
    out = await router.phrase(_ev(), lang="zh")
    assert out == "<p>"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_router_falls_back_when_primary_fails():
    primary, fallback = _Stub("p", "fail"), _Stub("f", "ok")
    router = PhraseRouter(primary, fallback, Config())
    out = await router.phrase(_ev(), lang="zh")
    assert out == "<f>"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_router_returns_template_when_both_fail():
    primary, fallback = _Stub("p", "fail"), _Stub("f", "fail")
    router = PhraseRouter(primary, fallback, Config())
    out = await router.phrase(_ev(), lang="zh")
    assert "先生" in out


@pytest.mark.asyncio
async def test_router_template_when_no_fallback():
    primary = _Stub("p", "fail")
    router = PhraseRouter(primary, None, Config())
    out = await router.phrase(_ev(), lang="en")
    assert "Sir" in out
```

- [ ] **Step 2: Implement `src/jarvis_cc/phrase/router.py`**

```python
"""Provider routing: try primary, then fallback, then template."""
from __future__ import annotations

from loguru import logger

from ..config import Config
from ..types import Event, Lang
from .providers.base import PhraseProvider
from .templates import render_template


class PhraseRouter:
    """Owns the LLM provider chain. Always returns a string."""

    def __init__(
        self,
        primary: PhraseProvider | None,
        fallback: PhraseProvider | None,
        cfg: Config,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cfg = cfg

    async def phrase(self, event: Event, lang: Lang) -> str:
        max_chars = self.cfg.behavior.phrase_max_chars
        for provider in (self.primary, self.fallback):
            if provider is None:
                continue
            try:
                out = await provider.generate(event, lang, max_chars)
                if out and out.strip():
                    return out.strip()
            except Exception as exc:  # broad: any provider failure → next
                logger.warning(
                    "Phrase provider {} failed: {}", provider.name, exc
                )
        return render_template(event, lang)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_router.py -v`
Expected: 4 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/phrase/router.py tests/unit/test_router.py
git commit -m "feat(phrase): provider router with primary→fallback→template chain"
```

---

## Task 12: TTS provider base + macOS `say` (`tts/providers/say.py`)

**Files:**
- Create: `src/jarvis_cc/tts/__init__.py`
- Create: `src/jarvis_cc/tts/providers/__init__.py`
- Create: `src/jarvis_cc/tts/providers/base.py`
- Create: `src/jarvis_cc/tts/providers/say.py`
- Create: `tests/unit/test_tts_say.py`

- [ ] **Step 1: Write test `tests/unit/test_tts_say.py`**

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis_cc.tts.providers.say import SayProvider


@pytest.mark.asyncio
async def test_say_writes_wav_to_path(tmp_path: Path):
    p = SayProvider()
    # We don't actually want to spawn `say` in CI; mock subprocess.
    out_path = tmp_path / "out.aiff"
    with patch("jarvis_cc.tts.providers.say.asyncio.create_subprocess_exec") as mock_exec:
        async def _fake(*args, **kwargs):
            # Pretend `say` succeeded and wrote a file
            (out_path).write_bytes(b"FAKE-AUDIO")

            class _P:
                returncode = 0

                async def communicate(self):
                    return (b"", b"")

                async def wait(self):
                    return 0

            return _P()

        mock_exec.side_effect = _fake

        audio = await p.synthesize("hello sir", lang="en", out_path=out_path)
    assert audio == out_path
    assert audio.read_bytes() == b"FAKE-AUDIO"
```

- [ ] **Step 2: Write `src/jarvis_cc/tts/__init__.py`**

```python
"""TTS package: provider routing + engines."""
```

- [ ] **Step 3: Write `src/jarvis_cc/tts/providers/__init__.py`**

```python
"""TTS provider implementations behind a shared ABC."""
```

- [ ] **Step 4: Implement `src/jarvis_cc/tts/providers/base.py`**

```python
"""Abstract base for TTS providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...types import Lang


class TTSProvider(ABC):
    """Synthesize `text` in `lang` to an audio file at `out_path` and return it."""

    name: str

    @abstractmethod
    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path: ...

    async def healthcheck(self) -> bool:
        return True
```

- [ ] **Step 5: Implement `src/jarvis_cc/tts/providers/say.py`**

```python
"""macOS `say` provider: lowest-quality but always-available fallback."""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...types import Lang
from .base import TTSProvider

_VOICE_BY_LANG = {
    "en": "Daniel",   # British male
    "zh": "Tingting",  # Mandarin female
}


class SayProvider(TTSProvider):
    name = "say"

    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path:
        voice = _VOICE_BY_LANG.get(lang, "Daniel")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # `say` outputs AIFF, but afplay handles both .aiff and .wav.
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", voice, "-o", str(out_path), text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"say failed: {err.decode(errors='replace')}")
        return out_path
```

- [ ] **Step 6: Run test**

Run: `uv run pytest tests/unit/test_tts_say.py -v`
Expected: 1 PASSED.

- [ ] **Step 7: Commit**

```bash
git add src/jarvis_cc/tts tests/unit/test_tts_say.py
git commit -m "feat(tts): macOS `say` fallback provider with ABC"
```

---

## Task 13: TTS XTTS-v2 provider (`tts/providers/xtts.py`)

> **Implementation note:** XTTS-v2 model load is slow (~5-10s) and requires PyTorch + MPS. We **lazy-load** the model on first call (so daemon start stays fast enough to be useful in the foreground). We hide the heavy `TTS` import inside the class so unit tests can run without coqui-tts installed when XTTS provider isn't exercised.

**Files:**
- Create: `src/jarvis_cc/tts/providers/xtts.py`
- Create: `tests/unit/test_xtts.py`

- [ ] **Step 1: Write test `tests/unit/test_xtts.py`**

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cc.config import XTTSConfig
from jarvis_cc.tts.providers.xtts import XTTSProvider


@pytest.mark.asyncio
async def test_xtts_calls_underlying_engine(tmp_path: Path):
    ref = tmp_path / "ref_zh.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref),
        ref_audio_en=str(ref),
        device="cpu",
    )
    p = XTTSProvider(cfg)

    fake_tts = MagicMock()
    fake_tts.tts_to_file = MagicMock(return_value=None)

    with patch.object(p, "_load_model", return_value=fake_tts):
        out = tmp_path / "out.wav"
        result = await p.synthesize("hello", lang="zh", out_path=out)

    assert result == out
    fake_tts.tts_to_file.assert_called_once()
    kwargs = fake_tts.tts_to_file.call_args.kwargs
    assert kwargs["text"] == "hello"
    assert kwargs["language"] == "zh-cn"
    assert kwargs["speaker_wav"] == str(ref)
    assert kwargs["file_path"] == str(out)


@pytest.mark.asyncio
async def test_xtts_raises_if_ref_audio_missing(tmp_path: Path):
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "missing.wav"),
        ref_audio_en=str(tmp_path / "missing.wav"),
        device="cpu",
    )
    p = XTTSProvider(cfg)
    with pytest.raises(FileNotFoundError):
        await p.synthesize("hi", lang="zh", out_path=tmp_path / "o.wav")
```

- [ ] **Step 2: Implement `src/jarvis_cc/tts/providers/xtts.py`**

```python
"""XTTS-v2 zero-shot voice-clone TTS via coqui-tts.

Heavy imports (`torch`, `TTS`) are inside `_load_model` to keep the daemon
import-light when XTTS isn't actually used (eg. user opts into ElevenLabs).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from ...config import XTTSConfig
from ...types import Lang
from .base import TTSProvider

_LANG_CODE = {"zh": "zh-cn", "en": "en"}


class XTTSProvider(TTSProvider):
    name = "xtts"

    def __init__(self, cfg: XTTSConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None

    def _ref_audio_for(self, lang: Lang) -> Path:
        path = self.cfg.ref_audio_zh if lang == "zh" else self.cfg.ref_audio_en
        return Path(path)

    def _load_model(self) -> Any:
        """Lazy-load the XTTS-v2 model. Called at most once per provider lifetime."""
        if self._model is not None:
            return self._model
        logger.info("Loading XTTS-v2 model from {} on {}", self.cfg.model_dir, self.cfg.device)
        from TTS.api import TTS  # type: ignore

        self._model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to(self.cfg.device)
        return self._model

    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path:
        ref = self._ref_audio_for(lang)
        if not ref.is_file():
            raise FileNotFoundError(f"reference audio missing: {ref}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _run() -> None:
            model = self._load_model()
            model.tts_to_file(
                text=text,
                speaker_wav=str(ref),
                language=_LANG_CODE.get(lang, "en"),
                file_path=str(out_path),
            )

        await asyncio.to_thread(_run)
        return out_path

    async def healthcheck(self) -> bool:
        return self._ref_audio_for("zh").is_file() and self._ref_audio_for("en").is_file()
```

- [ ] **Step 3: Run test**

Run: `uv run pytest tests/unit/test_xtts.py -v`
Expected: 2 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/tts/providers/xtts.py tests/unit/test_xtts.py
git commit -m "feat(tts): XTTS-v2 zero-shot provider with lazy model load"
```

---

## Task 14: TTS ElevenLabs provider (`tts/providers/elevenlabs.py`)

**Files:**
- Create: `src/jarvis_cc/tts/providers/elevenlabs.py`
- Create: `tests/unit/test_elevenlabs.py`

- [ ] **Step 1: Write test `tests/unit/test_elevenlabs.py`**

```python
from pathlib import Path

import pytest
import respx

from jarvis_cc.config import ElevenLabsConfig
from jarvis_cc.tts.providers.elevenlabs import ElevenLabsProvider


@pytest.mark.asyncio
async def test_elevenlabs_writes_audio_to_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="vid")
    fake_bytes = b"BYTES-OF-MP3"
    with respx.mock(base_url="https://api.elevenlabs.io") as router:
        router.post("/v1/text-to-speech/vid").respond(200, content=fake_bytes)
        out = tmp_path / "out.mp3"
        result = await ElevenLabsProvider(cfg).synthesize("hi", lang="en", out_path=out)
    assert result == out
    assert out.read_bytes() == fake_bytes


@pytest.mark.asyncio
async def test_elevenlabs_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    cfg = ElevenLabsConfig(voice_id="vid")
    with pytest.raises(RuntimeError):
        await ElevenLabsProvider(cfg).synthesize(
            "hi", lang="en", out_path=tmp_path / "o.mp3"
        )


@pytest.mark.asyncio
async def test_elevenlabs_raises_when_voice_id_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    cfg = ElevenLabsConfig(voice_id="")
    with pytest.raises(RuntimeError):
        await ElevenLabsProvider(cfg).synthesize(
            "hi", lang="en", out_path=tmp_path / "o.mp3"
        )
```

- [ ] **Step 2: Implement `src/jarvis_cc/tts/providers/elevenlabs.py`**

```python
"""ElevenLabs cloud TTS provider."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from ...config import ElevenLabsConfig
from ...types import Lang
from .base import TTSProvider


class ElevenLabsProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(self, cfg: ElevenLabsConfig) -> None:
        self.cfg = cfg

    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        if not self.cfg.voice_id:
            raise RuntimeError("ElevenLabs voice_id is not configured")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(
            base_url="https://api.elevenlabs.io", timeout=15.0
        ) as client:
            r = await client.post(
                f"/v1/text-to-speech/{self.cfg.voice_id}",
                headers={
                    "xi-api-key": key,
                    "accept": "audio/mpeg",
                    "content-type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.cfg.model,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            )
            r.raise_for_status()
            out_path.write_bytes(r.content)
        return out_path

    async def healthcheck(self) -> bool:
        return bool(os.getenv(self.cfg.api_key_env)) and bool(self.cfg.voice_id)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_elevenlabs.py -v`
Expected: 3 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/tts/providers/elevenlabs.py tests/unit/test_elevenlabs.py
git commit -m "feat(tts): ElevenLabs cloud provider"
```

---

## Task 15: TTS engine routing (`tts/engine.py`)

**Files:**
- Create: `src/jarvis_cc/tts/engine.py`
- Create: `tests/unit/test_tts_engine.py`

- [ ] **Step 1: Write test `tests/unit/test_tts_engine.py`**

```python
from pathlib import Path
from typing import Literal

import pytest

from jarvis_cc.tts.engine import TTSEngine
from jarvis_cc.tts.providers.base import TTSProvider


class _StubTTS(TTSProvider):
    def __init__(self, name: str, mode: Literal["ok", "fail"]):
        self.name = name
        self.mode = mode
        self.calls = 0

    async def synthesize(self, text, lang, out_path):
        self.calls += 1
        if self.mode == "fail":
            raise RuntimeError(f"{self.name} down")
        out_path.write_bytes(b"AUDIO-" + self.name.encode())
        return out_path


@pytest.mark.asyncio
async def test_engine_uses_primary_when_ok(tmp_path: Path):
    primary, fallback = _StubTTS("p", "ok"), _StubTTS("f", "ok")
    eng = TTSEngine(primary, fallback)
    out = await eng.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")
    assert out.read_bytes() == b"AUDIO-p"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_engine_falls_back_to_secondary(tmp_path: Path):
    primary, fallback = _StubTTS("p", "fail"), _StubTTS("f", "ok")
    eng = TTSEngine(primary, fallback)
    out = await eng.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")
    assert out.read_bytes() == b"AUDIO-f"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_engine_raises_when_all_fail(tmp_path: Path):
    primary, fallback = _StubTTS("p", "fail"), _StubTTS("f", "fail")
    eng = TTSEngine(primary, fallback)
    with pytest.raises(RuntimeError):
        await eng.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")
```

- [ ] **Step 2: Implement `src/jarvis_cc/tts/engine.py`**

```python
"""TTS engine: chains primary → fallback providers."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from ..types import Lang
from .providers.base import TTSProvider


class TTSEngine:
    def __init__(
        self,
        primary: TTSProvider,
        fallback: TTSProvider | None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path:
        for provider in (self.primary, self.fallback):
            if provider is None:
                continue
            try:
                return await provider.synthesize(text, lang, out_path)
            except Exception as exc:
                logger.warning("TTS provider {} failed: {}", provider.name, exc)
        raise RuntimeError("All TTS providers failed")
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_tts_engine.py -v`
Expected: 3 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/tts/engine.py tests/unit/test_tts_engine.py
git commit -m "feat(tts): engine with primary→fallback routing"
```

---

## Task 16: Audio player (`player.py`)

**Files:**
- Create: `src/jarvis_cc/player.py`
- Create: `tests/unit/test_player.py`

- [ ] **Step 1: Write test `tests/unit/test_player.py`**

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis_cc.player import play


@pytest.mark.asyncio
async def test_play_invokes_afplay_with_path(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    calls = []

    async def _fake_exec(*args, **kwargs):
        calls.append(args)

        class _P:
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    with patch("jarvis_cc.player.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        await play(audio)

    assert calls[0] == ("afplay", str(audio))


@pytest.mark.asyncio
async def test_play_raises_when_afplay_fails(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    async def _fake_exec(*args, **kwargs):
        class _P:
            returncode = 1

            async def communicate(self):
                return (b"", b"missing file")

            async def wait(self):
                return 1

        return _P()

    with patch("jarvis_cc.player.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        with pytest.raises(RuntimeError):
            await play(audio)
```

- [ ] **Step 2: Implement `src/jarvis_cc/player.py`**

```python
"""Thin async wrapper around macOS `afplay`."""
from __future__ import annotations

import asyncio
from pathlib import Path


async def play(audio: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "afplay", str(audio),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"afplay failed: {err.decode(errors='replace')}")
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_player.py -v`
Expected: 2 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/player.py tests/unit/test_player.py
git commit -m "feat(player): afplay async wrapper"
```

---

## Task 17: Health HTTP endpoint (`daemon/health.py`)

**Files:**
- Create: `src/jarvis_cc/daemon/health.py`
- Create: `tests/unit/test_health.py`

- [ ] **Step 1: Write test `tests/unit/test_health.py`**

```python
import asyncio

import httpx
import pytest

from jarvis_cc.daemon.health import HealthServer


@pytest.mark.asyncio
async def test_health_endpoint_returns_status():
    state = {"queue_size": 3, "dropped": 1, "last_text": "Sir, hello."}
    server = HealthServer(host="127.0.0.1", port=0, state_getter=lambda: state)
    await server.start()
    try:
        async with httpx.AsyncClient(base_url=server.url) as c:
            r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["queue_size"] == 3
        assert body["dropped"] == 1
        assert body["last_text"] == "Sir, hello."
    finally:
        await server.stop()
```

- [ ] **Step 2: Implement `src/jarvis_cc/daemon/health.py`**

```python
"""Tiny HTTP /health server using stdlib asyncio.

We avoid a heavy web framework dep — this is a single-route status endpoint.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from loguru import logger


class HealthServer:
    def __init__(
        self,
        host: str,
        port: int,
        state_getter: Callable[[], dict[str, Any]],
    ) -> None:
        self.host = host
        self.requested_port = port
        self.actual_port = port
        self._state_getter = state_getter
        self._server: asyncio.base_events.Server | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.actual_port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, host=self.host, port=self.requested_port
        )
        self.actual_port = self._server.sockets[0].getsockname()[1]
        logger.info("Health server on {}", self.url)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            # Consume remaining headers
            while True:
                hl = await reader.readline()
                if hl in (b"\r\n", b"\n", b""):
                    break
            if not line.startswith(b"GET /health"):
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            body = json.dumps(self._state_getter()).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
```

- [ ] **Step 3: Run test**

Run: `uv run pytest tests/unit/test_health.py -v`
Expected: 1 PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/daemon/health.py tests/unit/test_health.py
git commit -m "feat(daemon): minimal asyncio /health endpoint"
```

---

## Task 18: Daemon main loop wiring (`daemon/main.py`)

**Files:**
- Create: `src/jarvis_cc/daemon/main.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_daemon_e2e.py`

- [ ] **Step 1: Write integration test `tests/integration/test_daemon_e2e.py`**

```python
"""End-to-end: socket → dedup → queue → router → engine → player.

Stubs the LLM router, TTS engine, and player so we exercise wiring only.
"""
import asyncio
import json
import socket
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from jarvis_cc.config import Config
from jarvis_cc.daemon.main import Daemon
from jarvis_cc.types import Event


@pytest.mark.asyncio
async def test_daemon_handles_event_end_to_end(tmp_path: Path, monkeypatch):
    cfg = Config()
    cfg.paths.socket = str(tmp_path / "j.sock")
    cfg.paths.log = str(tmp_path / "d.log")
    cfg.paths.missed_log = str(tmp_path / "m.log")

    phrased: list[str] = []

    async def fake_phrase(event: Event, lang):
        s = f"Sir, {event.tool_name}!"
        phrased.append(s)
        return s

    played: list[Path] = []

    async def fake_play(audio: Path) -> None:
        played.append(audio)

    async def fake_synth(text: str, lang, out_path: Path):
        out_path.write_bytes(b"FAKE")
        return out_path

    d = Daemon(cfg, health_port=0)
    d.router.phrase = fake_phrase  # type: ignore[assignment]
    d.tts.synthesize = fake_synth  # type: ignore[assignment]
    monkeypatch.setattr("jarvis_cc.daemon.main.play", fake_play)

    task = asyncio.create_task(d.run())
    # wait for socket
    for _ in range(100):
        if Path(cfg.paths.socket).exists():
            break
        await asyncio.sleep(0.02)

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(cfg.paths.socket)
    s.sendall(
        (
            json.dumps(
                {
                    "notification_type": "permission_prompt",
                    "tool_name": "Bash",
                    "cwd": str(tmp_path),
                }
            )
            + "\n"
        ).encode()
    )
    s.close()

    # give the daemon time to process
    for _ in range(200):
        if played:
            break
        await asyncio.sleep(0.02)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert phrased == ["Sir, Bash!"]
    assert len(played) == 1


@pytest.mark.asyncio
async def test_daemon_dedups_within_window(tmp_path: Path, monkeypatch):
    cfg = Config()
    cfg.behavior.dedup_window_seconds = 10
    cfg.paths.socket = str(tmp_path / "j.sock")

    phrased: list[str] = []

    async def fake_phrase(event, lang):
        phrased.append(event.tool_name or "?")
        return "ok"

    async def fake_synth(text, lang, out_path):
        out_path.write_bytes(b"x")
        return out_path

    async def fake_play(audio):
        pass

    d = Daemon(cfg, health_port=0)
    d.router.phrase = fake_phrase  # type: ignore[assignment]
    d.tts.synthesize = fake_synth  # type: ignore[assignment]
    monkeypatch.setattr("jarvis_cc.daemon.main.play", fake_play)

    task = asyncio.create_task(d.run())
    for _ in range(100):
        if Path(cfg.paths.socket).exists():
            break
        await asyncio.sleep(0.02)

    payload = {
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "cwd": str(tmp_path),
    }
    for _ in range(3):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(cfg.paths.socket)
        s.sendall((json.dumps(payload) + "\n").encode())
        s.close()
        await asyncio.sleep(0.05)

    # wait for processing
    await asyncio.sleep(0.4)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert phrased == ["Bash"], f"expected dedup to 1 call, got {phrased}"
```

- [ ] **Step 2: Implement `src/jarvis_cc/daemon/main.py`**

```python
"""Daemon entrypoint: assembles the full pipeline and runs forever."""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from loguru import logger

from ..config import DEFAULT_CONFIG_PATH, Config, load_config
from ..phrase.language import detect_for
from ..phrase.providers.anthropic import AnthropicProvider
from ..phrase.providers.base import PhraseProvider
from ..phrase.providers.deepseek import DeepSeekProvider
from ..phrase.providers.ollama import OllamaProvider
from ..phrase.providers.openai import OpenAIProvider
from ..phrase.router import PhraseRouter
from ..player import play
from ..tts.engine import TTSEngine
from ..tts.providers.base import TTSProvider
from ..tts.providers.elevenlabs import ElevenLabsProvider
from ..tts.providers.say import SayProvider
from ..tts.providers.xtts import XTTSProvider
from ..types import Event
from .dedup import DedupWindow
from .health import HealthServer
from .listener import serve_unix_socket
from .queue import BoundedEventQueue


def _make_phrase_provider(name: str, cfg: Config) -> PhraseProvider | None:
    return {
        "deepseek": lambda: DeepSeekProvider(cfg.llm.deepseek),
        "anthropic": lambda: AnthropicProvider(cfg.llm.anthropic),
        "openai": lambda: OpenAIProvider(cfg.llm.openai),
        "ollama": lambda: OllamaProvider(cfg.llm.ollama),
    }.get(name, lambda: None)()


def _make_tts_provider(name: str, cfg: Config) -> TTSProvider | None:
    return {
        "xtts": lambda: XTTSProvider(cfg.tts.xtts),
        "elevenlabs": lambda: ElevenLabsProvider(cfg.tts.elevenlabs),
        "say": lambda: SayProvider(),
    }.get(name, lambda: None)()


class Daemon:
    def __init__(self, cfg: Config, health_port: int = 9527) -> None:
        self.cfg = cfg
        self.queue = BoundedEventQueue(maxsize=cfg.behavior.queue_max_size)
        self.dedup = DedupWindow(window_seconds=cfg.behavior.dedup_window_seconds)
        self.router = PhraseRouter(
            primary=_make_phrase_provider(cfg.llm.provider, cfg),
            fallback=_make_phrase_provider(cfg.llm.fallback, cfg),
            cfg=cfg,
        )
        primary_tts = _make_tts_provider(cfg.tts.provider, cfg) or SayProvider()
        fallback_tts = _make_tts_provider(cfg.tts.fallback, cfg)
        self.tts = TTSEngine(primary=primary_tts, fallback=fallback_tts)
        self.health = HealthServer(
            host="127.0.0.1",
            port=health_port,
            state_getter=self._snapshot,
        )
        self._last_text: str | None = None

    def _snapshot(self) -> dict:
        return {
            "queue_size": self.queue._maxsize,
            "dropped": self.queue.dropped_count,
            "last_text": self._last_text,
        }

    async def _on_event(self, event: Event) -> None:
        if event.notification_type not in self.cfg.behavior.events:
            return
        if self.dedup.is_duplicate(event):
            logger.debug("Dedup: {}", event.dedup_key())
            return
        await self.queue.put_or_drop(event)

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                lang = (
                    detect_for(event.cwd)
                    if self.cfg.behavior.voice_language == "auto"
                    else self.cfg.behavior.voice_language  # type: ignore[assignment]
                )
                text = await self.router.phrase(event, lang=lang)
                self._last_text = text
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = Path(tmp.name)
                await self.tts.synthesize(text, lang, out_path)
                try:
                    await play(out_path)
                finally:
                    try:
                        out_path.unlink()
                    except OSError:
                        pass
            except Exception as exc:
                logger.exception("worker failed: {}", exc)

    async def run(self) -> None:
        await self.health.start()
        worker_task = asyncio.create_task(self._worker())
        try:
            await serve_unix_socket(Path(self.cfg.paths.socket), self._on_event)
        finally:
            worker_task.cancel()
            await self.health.stop()


def main() -> int:
    parser = argparse.ArgumentParser(prog="jarvis-cc-daemon")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--health-port", type=int, default=9527)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.add(cfg.paths.log, rotation="10 MB", retention="14 days", level="INFO")
    try:
        asyncio.run(Daemon(cfg, health_port=args.health_port).run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Create test pkg marker**

```bash
echo '"""Integration tests."""' > tests/integration/__init__.py
```

- [ ] **Step 4: Run integration tests**

Run: `uv run pytest tests/integration/test_daemon_e2e.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/daemon/main.py tests/integration
git commit -m "feat(daemon): main loop wiring listener → dedup → queue → router → tts → play"
```

---

## Task 19: Install CLI (`install.py`)

> This is the most operator-flavored task. We patch `~/.claude/settings.json` idempotently (preserving any other hooks the user already had) and write a launchd plist.

**Files:**
- Create: `src/jarvis_cc/install.py`
- Create: `scripts/com.jobin.jarvis-cc.plist` (template — checked-in reference)
- Create: `tests/unit/test_install.py`

- [ ] **Step 1: Write test `tests/unit/test_install.py`**

```python
import json
from pathlib import Path

from jarvis_cc.install import (
    PLIST_LABEL,
    merge_claude_settings,
    render_plist,
)


def test_merge_settings_into_empty():
    out = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    assert out["hooks"]["Notification"][0]["hooks"][0]["command"] == "jarvis-cc-hook"


def test_merge_settings_preserves_other_hooks():
    existing = {
        "hooks": {
            "PreToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "x"}]}]
        }
    }
    out = merge_claude_settings(existing, hook_command="jarvis-cc-hook")
    assert "PreToolUse" in out["hooks"]
    assert "Notification" in out["hooks"]


def test_merge_settings_is_idempotent():
    out1 = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    out2 = merge_claude_settings(out1, hook_command="jarvis-cc-hook")
    notif = out2["hooks"]["Notification"]
    # Should not duplicate our entry
    assert sum(1 for n in notif for h in n["hooks"] if h["command"] == "jarvis-cc-hook") == 1


def test_render_plist_contains_label_and_program(tmp_path: Path):
    plist = render_plist(
        label=PLIST_LABEL,
        program="/usr/local/bin/jarvis-cc-daemon",
        log_dir=str(tmp_path),
    )
    assert "<string>com.jobin.jarvis-cc</string>" in plist
    assert "<string>/usr/local/bin/jarvis-cc-daemon</string>" in plist
    assert "<key>KeepAlive</key>" in plist
    assert str(tmp_path) in plist
```

- [ ] **Step 2: Implement `src/jarvis_cc/install.py`**

```python
"""Operator CLI: install / uninstall / status / test.

Commands:
  jarvis-cc install     - write config, hook into ~/.claude/settings.json, install plist
  jarvis-cc uninstall   - reverse install steps (keeps user data)
  jarvis-cc status      - check daemon health
  jarvis-cc test        - send a synthetic event to the daemon
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx

from .config import DEFAULT_CONFIG_PATH, expanduser

PLIST_LABEL = "com.jobin.jarvis-cc"
PLIST_PATH = expanduser(f"~/Library/LaunchAgents/{PLIST_LABEL}.plist")
CLAUDE_SETTINGS_PATH = expanduser("~/.claude/settings.json")
JARVIS_DIR = expanduser("~/.jarvis-cc")


def merge_claude_settings(existing: dict, hook_command: str) -> dict:
    """Idempotently add our Notification hook entry without disturbing others."""
    out = copy.deepcopy(existing)
    hooks = out.setdefault("hooks", {})
    notification = hooks.setdefault("Notification", [])
    # Detect existing entry
    for matcher in notification:
        for hook in matcher.get("hooks", []):
            if hook.get("command") == hook_command:
                return out
    notification.append(
        {"matcher": "", "hooks": [{"type": "command", "command": hook_command}]}
    )
    return out


def remove_from_claude_settings(existing: dict, hook_command: str) -> dict:
    out = copy.deepcopy(existing)
    notification = out.get("hooks", {}).get("Notification", [])
    filtered = []
    for matcher in notification:
        hooks = [h for h in matcher.get("hooks", []) if h.get("command") != hook_command]
        if hooks:
            filtered.append({**matcher, "hooks": hooks})
    if "hooks" in out:
        out["hooks"]["Notification"] = filtered
    return out


def render_plist(label: str, program: str, log_dir: str) -> str:
    """Render a launchd plist for the daemon."""
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{label}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{program}</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{log_dir}/daemon.stdout.log</string>
            <key>StandardErrorPath</key>
            <string>{log_dir}/daemon.stderr.log</string>
            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
            </dict>
        </dict>
        </plist>
        """
    )


def cmd_install(args: argparse.Namespace) -> int:
    # 1. Make jarvis-cc dir tree
    base = Path(JARVIS_DIR)
    (base / "voices").mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)

    # 2. Write default config if not present
    cfg_path = Path(DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        cfg_path.write_text(_default_config_toml(), encoding="utf-8")
        print(f"  wrote {cfg_path}")
    else:
        print(f"  (kept existing {cfg_path})")

    # 3. Patch ~/.claude/settings.json
    settings_path = Path(CLAUDE_SETTINGS_PATH)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            print(f"!! could not parse {settings_path}, refusing to overwrite", file=sys.stderr)
            return 2
    merged = merge_claude_settings(existing, hook_command="jarvis-cc-hook")
    settings_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"  patched {settings_path}")

    # 4. Write plist
    program = shutil.which("jarvis-cc-daemon")
    if not program:
        print("!! jarvis-cc-daemon not on PATH — did you run `uv sync`?", file=sys.stderr)
        return 3
    plist_path = Path(PLIST_PATH)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(render_plist(PLIST_LABEL, program, str(base / "logs")))
    print(f"  wrote {plist_path}")

    # 5. Load plist
    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    rc = subprocess.run(["launchctl", "load", str(plist_path)], check=False).returncode
    if rc != 0:
        print(f"!! launchctl load returned {rc}", file=sys.stderr)
        return rc

    print(
        "\nDone. Next steps:\n"
        f"  1. Place reference audio at {base / 'voices/jarvis_zh.wav'} and "
        f"{base / 'voices/jarvis_en.wav'}\n"
        f"  2. Set environment variables: DEEPSEEK_API_KEY (required), "
        f"ELEVENLABS_API_KEY (optional)\n"
        f"  3. Restart Claude Code\n"
        f"  4. Run: jarvis-cc test\n"
    )
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    plist_path = Path(PLIST_PATH)
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        plist_path.unlink()
        print(f"  removed {plist_path}")

    settings_path = Path(CLAUDE_SETTINGS_PATH)
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
            pruned = remove_from_claude_settings(existing, hook_command="jarvis-cc-hook")
            settings_path.write_text(json.dumps(pruned, indent=2, ensure_ascii=False) + "\n")
            print(f"  cleaned {settings_path}")
        except json.JSONDecodeError:
            pass

    if args.purge:
        base = Path(JARVIS_DIR)
        if base.exists():
            shutil.rmtree(base)
            print(f"  purged {base}")
    else:
        print(f"  (kept {JARVIS_DIR}; pass --purge to remove)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        r = httpx.get("http://127.0.0.1:9527/health", timeout=1.0)
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        return 0
    except httpx.HTTPError as exc:
        print(f"daemon unreachable: {exc}", file=sys.stderr)
        return 1


def cmd_test(args: argparse.Namespace) -> int:
    import socket as _socket

    from .config import load_config

    cfg = load_config(DEFAULT_CONFIG_PATH)
    payload = {
        "notification_type": args.event,
        "tool_name": args.tool,
        "tool_input": {},
        "cwd": os.getcwd(),
        "session_id": "test",
    }
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.connect(cfg.paths.socket)
        s.sendall((json.dumps(payload) + "\n").encode())
        print(f"sent {args.event} ({args.tool}) to {cfg.paths.socket}")
        return 0
    except OSError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    finally:
        s.close()


def _default_config_toml() -> str:
    return textwrap.dedent(
        """\
        # jarvis-cc config.toml — auto-generated, edit freely
        [llm]
        provider = "deepseek"
        fallback = "ollama"

        [llm.deepseek]
        api_key_env = "DEEPSEEK_API_KEY"
        model = "deepseek-chat"

        [llm.ollama]
        base_url = "http://localhost:11434"
        model = "qwen2.5:7b"

        [tts]
        provider = "xtts"
        fallback = "say"

        [tts.xtts]
        model_dir = "~/.jarvis-cc/models/xtts-v2"
        ref_audio_zh = "~/.jarvis-cc/voices/jarvis_zh.wav"
        ref_audio_en = "~/.jarvis-cc/voices/jarvis_en.wav"
        device = "mps"

        [tts.elevenlabs]
        api_key_env = "ELEVENLABS_API_KEY"
        voice_id = ""
        model = "eleven_turbo_v2_5"

        [behavior]
        dedup_window_seconds = 10
        queue_max_size = 5
        voice_language = "auto"
        events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
        phrase_max_chars = 30
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="jarvis-cc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install").set_defaults(func=cmd_install)
    p_un = sub.add_parser("uninstall")
    p_un.add_argument("--purge", action="store_true", help="also remove ~/.jarvis-cc/")
    p_un.set_defaults(func=cmd_uninstall)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p_test = sub.add_parser("test")
    p_test.add_argument("--event", default="permission_prompt")
    p_test.add_argument("--tool", default="Bash")
    p_test.set_defaults(func=cmd_test)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write `scripts/com.jobin.jarvis-cc.plist` (reference template)**

```bash
mkdir -p scripts
cat > scripts/com.jobin.jarvis-cc.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!-- This is a reference template. The real plist is rendered by `jarvis-cc install`. -->
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jobin.jarvis-cc</string>
    <key>ProgramArguments</key>
    <array>
        <string>/REPLACED/AT/INSTALL/jarvis-cc-daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
PLIST
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_install.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/install.py scripts/com.jobin.jarvis-cc.plist tests/unit/test_install.py
git commit -m "feat(cli): install/uninstall/status/test with idempotent settings patch"
```

---

## Task 20: `__main__.py` + smoke test

**Files:**
- Create: `src/jarvis_cc/__main__.py`
- Create: `tests/integration/test_full_loop.py`

- [ ] **Step 1: Write `src/jarvis_cc/__main__.py`**

```python
"""Allow `python -m jarvis_cc <subcommand>` as an alternative to console scripts."""
from __future__ import annotations

import sys

from .install import main as install_main


def main() -> int:
    return install_main()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write smoke test `tests/integration/test_full_loop.py`**

```python
"""Smoke test: hook_client → real daemon (mocked LLM + TTS) → mocked player."""
import asyncio
import io
import json
import os
from pathlib import Path

import pytest

from jarvis_cc.config import Config
from jarvis_cc.daemon.main import Daemon
from jarvis_cc.hook_client import forward_event


@pytest.mark.asyncio
async def test_hook_to_daemon_smoke(tmp_path: Path, monkeypatch):
    cfg = Config()
    cfg.paths.socket = str(tmp_path / "j.sock")

    phrased: list[str] = []
    played: list[Path] = []

    async def fake_phrase(event, lang):
        s = f"Sir, {event.notification_type}"
        phrased.append(s)
        return s

    async def fake_synth(text, lang, out_path):
        out_path.write_bytes(b"WAV")
        return out_path

    async def fake_play(audio):
        played.append(audio)

    d = Daemon(cfg, health_port=0)
    d.router.phrase = fake_phrase  # type: ignore[assignment]
    d.tts.synthesize = fake_synth  # type: ignore[assignment]
    monkeypatch.setattr("jarvis_cc.daemon.main.play", fake_play)

    daemon_task = asyncio.create_task(d.run())
    for _ in range(100):
        if Path(cfg.paths.socket).exists():
            break
        await asyncio.sleep(0.02)
    assert Path(cfg.paths.socket).exists()

    # Use the real hook_client end (runs sync IO, so do it in a thread)
    payload = json.dumps(
        {
            "notification_type": "permission_prompt",
            "tool_name": "Bash",
            "cwd": str(tmp_path),
        }
    )
    ok = await asyncio.to_thread(forward_event, io.StringIO(payload), cfg.paths.socket)
    assert ok is True

    for _ in range(200):
        if played:
            break
        await asyncio.sleep(0.02)

    daemon_task.cancel()
    try:
        await daemon_task
    except asyncio.CancelledError:
        pass

    assert phrased and "permission_prompt" in phrased[0]
    assert played
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (~40 tests across 14 test files).

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/__main__.py tests/integration/test_full_loop.py
git commit -m "feat: __main__ entry + end-to-end smoke test"
```

---

## Task 21: Manual hardware smoke test

This step is not automatable but it's how we confirm the MVP actually works on real hardware. Do this after Task 20 commits.

- [ ] **Step 1: Place a reference audio file**

The user must record (or extract) a 10-30 second clean speech sample for each language and place at:
- `~/.jarvis-cc/voices/jarvis_en.wav`
- `~/.jarvis-cc/voices/jarvis_zh.wav`

For initial validation, any clean male voice clip works — quality of the Jarvis impression depends on this audio.

- [ ] **Step 2: Set API key**

```bash
export DEEPSEEK_API_KEY=sk-xxx-your-real-key
echo 'export DEEPSEEK_API_KEY=sk-xxx-your-real-key' >> ~/.zshrc
```

- [ ] **Step 3: Install**

```bash
cd ~/myself/jarvis-cc
uv sync
uv run jarvis-cc install
```

- [ ] **Step 4: Verify daemon is up**

```bash
uv run jarvis-cc status
```

Expected: JSON output with `queue_size`, `dropped`, `last_text` fields (last_text may be null).

- [ ] **Step 5: Send a test event**

```bash
uv run jarvis-cc test --event permission_prompt --tool Bash
```

Expected: You hear a Jarvis-voiced sentence within ~3 seconds.

- [ ] **Step 6: Verify Claude Code hook is wired**

Open a fresh Claude Code session in any project. Ask Claude to do something that needs Bash permission (eg `请运行 ls`). When the permission prompt appears in Claude Code, you should hear the Jarvis voice within 1-3 seconds.

- [ ] **Step 7: Tail logs to confirm**

```bash
tail -f ~/.jarvis-cc/logs/daemon.stdout.log
```

You should see info lines for each event processed.

- [ ] **Step 8: Merge to main + tag MVP**

```bash
git checkout main
git merge --no-ff feat/mvp
git tag v0.1.0-mvp
git log --oneline | head
```

---

## Test inventory (running totals)

| Task | Unit tests | Integration | Cumulative |
|---|---|---|---|
| 1  | 3 | 0 | 3 |
| 2  | 3 | 0 | 6 |
| 3  | 3 | 0 | 9 |
| 4  | 4 | 0 | 13 |
| 5  | 5 | 0 | 18 |
| 6  | 2 | 0 | 20 |
| 7  | 5 | 0 | 25 |
| 8  | 6 | 0 | 31 |
| 9  | 5 | 0 | 36 |
| 10 | 2 | 0 | 38 |
| 11 | 4 | 0 | 42 |
| 12 | 1 | 0 | 43 |
| 13 | 2 | 0 | 45 |
| 14 | 3 | 0 | 48 |
| 15 | 3 | 0 | 51 |
| 16 | 2 | 0 | 53 |
| 17 | 1 | 0 | 54 |
| 18 | 0 | 2 | 56 |
| 19 | 4 | 0 | 60 |
| 20 | 0 | 1 | 61 |

Final smoke test in Task 21 is manual.

---

## Notes for the implementer

1. **Run order strict.** Tasks have an implicit dependency chain (e.g., `daemon.main` imports everything). Don't skip ahead.
2. **MPS device caveats.** XTTS-v2 on Apple Silicon MPS occasionally produces NaN output on some PyTorch versions. If you see warnings during Task 21 hardware test, fall back to `device = "cpu"` in `~/.jarvis-cc/config.toml`. CPU on M1 Pro is ~3-5s per sentence, still acceptable.
3. **Don't commit reference audio.** `.gitignore` covers `*.wav` and `*.mp3` but be careful when staging — never `git add ~/.jarvis-cc/voices/`.
4. **DeepSeek base URL evolves.** Spec uses `https://api.deepseek.com`. If 404s, double-check current docs and adjust `DeepSeekConfig.base_url` (config-driven, no code change).
5. **Tests use respx.** Each HTTP-touching provider test should isolate `respx.mock(base_url=...)` so tests run offline.
6. **`coqui-tts` install footprint.** First `uv sync` will pull ~2GB of PyTorch + transformers. Expect 5-10 min on first run. Subsequent invocations are fast.
7. **No `git push` in any task.** All commits stay on local `feat/mvp` until Task 21 step 8.
