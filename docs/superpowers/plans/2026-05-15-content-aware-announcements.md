# Content-Aware Announcements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Jarvis's phrase generation so the spoken sentence names the salient artefact from `tool_input` (Bash command, file basename, URL, grep pattern) instead of issuing a generic alert.

**Architecture:** Insert two new helpers inside `src/jarvis_cc/phrase/` — `extract.py` (per-tool summary) and `redact.py` (secret/path scrubber) — and have `PhraseRouter` call them before building messages. Change `PhraseProvider.generate` from `(event, lang, max_chars)` to `(messages)` so providers become dumb HTTP adapters. Length control moves from runtime truncation into prompt-level soft target + hard cap.

**Tech Stack:** Python 3.11+, dataclasses, `httpx`, `pytest`/`pytest-asyncio`, `loguru`. No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-05-15-content-aware-announcements-design.md`

---

## File Map

**New files:**
- `src/jarvis_cc/phrase/extract.py` — per-tool extractor (~40 LOC)
- `src/jarvis_cc/phrase/redact.py` — secret/path scrubber (~30 LOC)
- `tests/unit/test_phrase_extract.py`
- `tests/unit/test_phrase_redact.py`
- `tests/unit/test_phrase_router_content_aware.py`

**Modified files:**
- `src/jarvis_cc/config.py` — add `PrivacyConfig`, two new `BehaviorConfig` fields
- `src/jarvis_cc/phrase/prompt.py` — new signature, new system prompt, expanded few-shot
- `src/jarvis_cc/phrase/router.py` — wire extract → redact → build_messages, drop post-truncation
- `src/jarvis_cc/phrase/providers/base.py` — change `generate` signature
- `src/jarvis_cc/phrase/providers/deepseek.py` — adapt to new signature
- `src/jarvis_cc/phrase/providers/openai.py` — adapt to new signature
- `src/jarvis_cc/phrase/providers/anthropic.py` — adapt to new signature
- `src/jarvis_cc/phrase/providers/ollama.py` — adapt to new signature
- `src/jarvis_cc/install.py` — write new keys in default `config.toml`
- `tests/unit/test_config.py` — assert new field defaults
- `tests/unit/test_prompt.py` — update for new signature
- `tests/unit/test_router.py` — update stub to new provider signature
- `tests/unit/test_phrase_providers.py` / `test_phrase_providers_more.py` — update for new provider signature

**Untouched:** hook_client, daemon/\*, tts/\*, player, types.py, templates.py, the `say --text` path, `paths.socket`/launchd plist plumbing.

---

## Task 1: Config — privacy block + length-budget fields

**Files:**
- Modify: `src/jarvis_cc/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Add a failing test for the new defaults**

Append to `tests/unit/test_config.py`:

```python
def test_load_config_defaults_phrase_budget(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.behavior.phrase_target_chars == 70
    assert cfg.behavior.phrase_hard_cap == 120
    assert cfg.behavior.privacy.cloud_redaction is True
    # legacy field kept silently for back-compat
    assert hasattr(cfg.behavior, "phrase_max_chars")


def test_load_config_reads_privacy_override(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(
        """
[behavior]
phrase_target_chars = 90
phrase_hard_cap = 160

[behavior.privacy]
cloud_redaction = false
"""
    )
    cfg = load_config(p)
    assert cfg.behavior.phrase_target_chars == 90
    assert cfg.behavior.phrase_hard_cap == 160
    assert cfg.behavior.privacy.cloud_redaction is False
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /Users/jiabinbin/myself/jarvis-cc
uv run pytest tests/unit/test_config.py::test_load_config_defaults_phrase_budget -v
```
Expected: FAIL — `BehaviorConfig` has no `phrase_target_chars`.

- [ ] **Step 3: Add `PrivacyConfig` and extend `BehaviorConfig`**

Edit `src/jarvis_cc/config.py`. Add new dataclass above `BehaviorConfig`:

```python
@dataclass
class PrivacyConfig:
    cloud_redaction: bool = True
```

Replace the `BehaviorConfig` block with:

```python
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
    # DEPRECATED: kept so old config.toml files don't error on load. Not read
    # at runtime; replaced by phrase_target_chars + phrase_hard_cap below.
    phrase_max_chars: int = 30
    phrase_target_chars: int = 70
    phrase_hard_cap: int = 120
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
```

- [ ] **Step 4: Run both new tests**

```bash
uv run pytest tests/unit/test_config.py -v
```
Expected: all 5 tests PASS (3 pre-existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/config.py tests/unit/test_config.py
git commit -m "feat(config): add privacy.cloud_redaction + phrase length budget fields"
```

---

## Task 2: `phrase/redact.py` — secret + path scrubber

**Files:**
- Create: `src/jarvis_cc/phrase/redact.py`
- Test: `tests/unit/test_phrase_redact.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_phrase_redact.py`:

```python
import os

from jarvis_cc.phrase.redact import scrub


def test_scrub_replaces_home_path(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/jobin")
    # Force module-level _HOME refresh via import-time computation
    import importlib
    import jarvis_cc.phrase.redact as r
    importlib.reload(r)
    assert r.scrub("rm -rf /Users/jobin/tmp/x") == "rm -rf ~/tmp/x"


def test_scrub_redacts_openai_key():
    out = scrub("curl -H 'auth: sk-abcdef1234567890ABCDEF' x")
    assert "sk-abcdef" not in out
    assert "<REDACTED>" in out


def test_scrub_redacts_eleven_key():
    out = scrub("ELEVENLABS=sk_1234567890abcdefABCDEF")
    assert "<REDACTED>" in out


def test_scrub_redacts_github_pat():
    out = scrub("token ghp_abcdefghijklmnopqrstuvwxyz123456")
    assert "<REDACTED>" in out


def test_scrub_redacts_aws_key():
    out = scrub("AKIA1234567890ABCDEF foo")
    assert "<REDACTED>" in out


def test_scrub_redacts_slack_token():
    out = scrub("xoxb-12345-67890-abcdefgABCDEFG")
    assert "<REDACTED>" in out


def test_scrub_redacts_long_hex_token():
    out = scrub("commit deadbeef1234567890cafebabe1234567890aabbcc11")
    assert "<REDACTED>" in out
    assert "deadbeef" not in out


def test_scrub_truncates_to_200_chars():
    long = "x" * 500
    assert len(scrub(long)) == 200


def test_scrub_disabled_only_truncates():
    long = "/Users/jobin/" + "x" * 300
    out = scrub(long, enabled=False)
    assert len(out) == 200
    assert out.startswith("/Users/jobin/")  # HOME NOT replaced


def test_scrub_empty_string_is_empty():
    assert scrub("") == ""
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
uv run pytest tests/unit/test_phrase_redact.py -v
```
Expected: FAIL — `jarvis_cc.phrase.redact` does not exist.

- [ ] **Step 3: Implement `phrase/redact.py`**

Create `src/jarvis_cc/phrase/redact.py`:

```python
"""Light redaction applied to extracted summary before sending to LLM.

Substitutes the user's home directory and common secret-shaped tokens
with placeholders, then truncates to a hard length cap. NOT a security
guarantee — best-effort defence-in-depth so cloud LLM providers don't
see paths or accidentally-pasted keys in Jarvis's prompt.
"""
from __future__ import annotations

import os
import re

_HOME = os.path.expanduser("~")
_MAX_OUT = 200

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk_[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),
]


def scrub(text: str, *, enabled: bool = True) -> str:
    """Return a possibly-redacted, length-capped version of `text`.

    Length cap (200 chars) always applies; pattern substitutions only
    when `enabled` is True.
    """
    if not text:
        return text
    if not enabled:
        return text[:_MAX_OUT]
    out = text.replace(_HOME, "~")
    for p in _PATTERNS:
        out = p.sub("<REDACTED>", out)
    return out[:_MAX_OUT]
```

- [ ] **Step 4: Run all redact tests**

```bash
uv run pytest tests/unit/test_phrase_redact.py -v
```
Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/phrase/redact.py tests/unit/test_phrase_redact.py
git commit -m "feat(phrase): redact.py — HOME path + secret-shaped token scrubber"
```

---

## Task 3: `phrase/extract.py` — per-tool summary

**Files:**
- Create: `src/jarvis_cc/phrase/extract.py`
- Test: `tests/unit/test_phrase_extract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_phrase_extract.py`:

```python
from jarvis_cc.phrase.extract import extract


def test_extract_empty_input_returns_empty():
    assert extract("Bash", {}) == ""
    assert extract(None, {}) == ""
    assert extract("Bash", None) == ""  # type: ignore[arg-type]


def test_extract_bash_returns_command_truncated():
    out = extract("Bash", {"command": "rm -rf /tmp/foo"})
    assert out == "rm -rf /tmp/foo"


def test_extract_bash_truncates_long_command():
    long_cmd = "echo " + "x" * 500
    out = extract("Bash", {"command": long_cmd})
    assert len(out) <= 200


def test_extract_write_uses_basename():
    out = extract("Write", {"file_path": "/Users/jobin/proj/config.toml", "content": "..."})
    assert out == "write config.toml"


def test_extract_edit_uses_basename():
    out = extract("Edit", {"file_path": "/a/b/foo.py", "old_string": "x", "new_string": "y"})
    assert out == "edit foo.py"


def test_extract_multiedit_uses_basename():
    out = extract("MultiEdit", {"file_path": "/a/b/foo.py"})
    assert out == "edit foo.py"


def test_extract_read_uses_basename():
    assert extract("Read", {"file_path": "/a/b/c.md"}) == "read c.md"


def test_extract_grep_quotes_pattern():
    out = extract("Grep", {"pattern": "def main"})
    assert out == "grep 'def main'"


def test_extract_grep_truncates_long_pattern():
    out = extract("Grep", {"pattern": "x" * 200})
    assert len(out) <= 200


def test_extract_glob_quotes_pattern():
    out = extract("Glob", {"pattern": "**/*.py"})
    assert out == "glob '**/*.py'"


def test_extract_webfetch_includes_url():
    out = extract("WebFetch", {"url": "https://example.com/secret", "prompt": "x"})
    assert out.startswith("fetch https://example.com")


def test_extract_unknown_tool_dumps_json():
    out = extract("SomeNewTool", {"foo": "bar", "n": 1})
    assert '"foo"' in out
    assert '"bar"' in out
    assert len(out) <= 200


def test_extract_write_without_file_path_falls_back():
    assert extract("Write", {"content": "x"}) == "write"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
uv run pytest tests/unit/test_phrase_extract.py -v
```
Expected: FAIL — `jarvis_cc.phrase.extract` does not exist.

- [ ] **Step 3: Implement `phrase/extract.py`**

Create `src/jarvis_cc/phrase/extract.py`:

```python
"""Per-tool extractor: turns raw tool_input into a short summary string.

The summary is what the LLM-side prompt sees instead of the full
tool_input dict. Keeps the prompt small, predictable, and free of
nuisance fields (Write/Edit content blobs, etc.).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

_MAX_RAW = 200


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip("/")) or path


def _bash(ti: dict[str, Any]) -> str:
    cmd = str(ti.get("command", "")).strip()
    return cmd[:_MAX_RAW]


def _write(ti: dict[str, Any]) -> str:
    fp = str(ti.get("file_path", ""))
    return f"write {_basename(fp)}" if fp else "write"


def _edit(ti: dict[str, Any]) -> str:
    fp = str(ti.get("file_path", ""))
    return f"edit {_basename(fp)}" if fp else "edit"


def _read(ti: dict[str, Any]) -> str:
    fp = str(ti.get("file_path", ""))
    return f"read {_basename(fp)}" if fp else "read"


def _grep(ti: dict[str, Any]) -> str:
    pat = str(ti.get("pattern", ""))[:80]
    return f"grep '{pat}'" if pat else "grep"


def _glob(ti: dict[str, Any]) -> str:
    pat = str(ti.get("pattern", ""))[:80]
    return f"glob '{pat}'" if pat else "glob"


def _webfetch(ti: dict[str, Any]) -> str:
    url = str(ti.get("url", ""))[:120]
    return f"fetch {url}" if url else "fetch"


_EXTRACTORS: dict[str, Callable[[dict[str, Any]], str]] = {
    "Bash": _bash,
    "Write": _write,
    "Edit": _edit,
    "MultiEdit": _edit,
    "Read": _read,
    "Grep": _grep,
    "Glob": _glob,
    "WebFetch": _webfetch,
    "WebSearch": _webfetch,
}


def extract(tool_name: str | None, tool_input: dict[str, Any] | None) -> str:
    """Return a one-line summary; '' when nothing useful is present."""
    if not tool_input:
        return ""
    if tool_name and tool_name in _EXTRACTORS:
        return _EXTRACTORS[tool_name](tool_input).strip()
    return json.dumps(tool_input, ensure_ascii=False)[:_MAX_RAW]
```

- [ ] **Step 4: Run all extract tests**

```bash
uv run pytest tests/unit/test_phrase_extract.py -v
```
Expected: 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/phrase/extract.py tests/unit/test_phrase_extract.py
git commit -m "feat(phrase): extract.py — per-tool tool_input → summary normaliser"
```

---

## Task 4: Rewrite `prompt.py` with new signature

**Files:**
- Modify: `src/jarvis_cc/phrase/prompt.py`
- Test: `tests/unit/test_prompt.py`

- [ ] **Step 1: Replace test_prompt.py with new-signature tests**

Replace the entire contents of `tests/unit/test_prompt.py`:

```python
from jarvis_cc.phrase.prompt import build_messages
from jarvis_cc.types import Event


def _ev(**kw) -> Event:
    return Event(
        notification_type=kw.get("notification_type", "permission_prompt"),
        tool_name=kw.get("tool_name", "Bash"),
        tool_input=kw.get("tool_input", {}),
    )


def test_build_messages_zh_system_includes_target_and_cap():
    ev = _ev()
    msgs = build_messages(ev, lang="zh", summary="rm -rf /tmp/x",
                          target_chars=70, hard_cap=120)
    assert msgs[0]["role"] == "system"
    assert "J.A.R.V.I.S" in msgs[0]["content"] or "管家" in msgs[0]["content"]
    assert "中文" in msgs[0]["content"]
    assert "70" in msgs[0]["content"]
    assert "120" in msgs[0]["content"]


def test_build_messages_en_swaps_language_clause():
    msgs = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120)
    assert "English" in msgs[0]["content"]


def test_build_messages_user_blob_contains_summary_not_raw_tool_input():
    ev = _ev(tool_input={"command": "rm -rf /Users/jobin/tmp", "extra": "huge"})
    msgs = build_messages(ev, lang="en", summary="rm -rf ~/tmp",
                          target_chars=70, hard_cap=120)
    last = msgs[-1]["content"]
    assert "rm -rf ~/tmp" in last
    # The full raw tool_input should NOT leak into the prompt
    assert "huge" not in last
    assert "/Users/jobin" not in last


def test_build_messages_includes_tool_name_in_user_blob():
    msgs = build_messages(_ev(tool_name="Write"), lang="en",
                          summary="write c.toml", target_chars=70, hard_cap=120)
    assert "Write" in msgs[-1]["content"]


def test_build_messages_few_shot_present_for_both_langs():
    msgs_zh = build_messages(_ev(), lang="zh", summary="",
                             target_chars=70, hard_cap=120)
    msgs_en = build_messages(_ev(), lang="en", summary="",
                             target_chars=70, hard_cap=120)
    # system + few-shot pairs + final user → at least 5 messages each
    assert len(msgs_zh) >= 5
    assert len(msgs_en) >= 5


def test_build_messages_empty_summary_still_valid():
    msgs = build_messages(_ev(notification_type="idle_prompt", tool_name=None),
                          lang="zh", summary="", target_chars=70, hard_cap=120)
    assert msgs[-1]["role"] == "user"
    last = msgs[-1]["content"]
    assert '"summary": ""' in last or '"summary":""' in last
```

- [ ] **Step 2: Run new prompt tests — they should fail**

```bash
uv run pytest tests/unit/test_prompt.py -v
```
Expected: FAIL — signature mismatch (`max_chars` vs `summary`/`target_chars`/`hard_cap`).

- [ ] **Step 3: Rewrite `prompt.py`**

Replace the entire contents of `src/jarvis_cc/phrase/prompt.py`:

```python
"""Jarvis-tone prompt builder shared across all LLM providers.

Inputs: an Event, a redacted-and-extracted `summary` string, language, and
soft/hard length budget. Output: an OpenAI-compatible chat messages list
ready to pass to any provider.
"""
from __future__ import annotations

import json

from ..types import Event, Lang

_SYSTEM_BASE = (
    "You are J.A.R.V.I.S., Tony Stark's polite British AI butler. "
    "Address the user as '{addr}'. Given a Claude Code event, reply with ONE "
    "short sentence in {lang_name} that ALERTS the user AND names the salient "
    "thing they need to decide on. Aim for roughly {target_chars} characters; "
    "you may go up to {hard_cap} if needed to keep the key detail. Be calm, "
    "courteous, with a hint of dry wit. If a 'summary' field is provided, weave "
    "its content into your sentence (quote a file name, the command verb, or "
    "the pattern — whatever is most actionable). Do NOT explain. Do NOT add "
    "quotes or labels around your output."
)

_FEW_SHOT_ZH = [
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm -rf ~/tmp/xyz"}'},
    {"role": "assistant", "content": "先生，他打算 rm -rf 一个临时目录，烦请定夺。"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Write","summary":"write config.toml"}'},
    {"role": "assistant", "content": "先生，他想覆写 config.toml，是否放行？"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"WebFetch","summary":"fetch https://example.com"}'},
    {"role": "assistant", "content": "先生，他欲访问 example.com，请您过目。"},
    {"role": "user",
     "content": '{"notification_type":"idle_prompt","tool_name":null,"summary":""}'},
    {"role": "assistant", "content": "先生，Claude 静候您的吩咐。"},
]

_FEW_SHOT_EN = [
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm -rf ~/tmp/xyz"}'},
    {"role": "assistant", "content": "Sir, he intends `rm -rf ~/tmp/xyz` — your verdict?"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Write","summary":"write config.toml"}'},
    {"role": "assistant", "content": "Sir, Claude wishes to overwrite `config.toml` — shall I permit?"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"WebFetch","summary":"fetch https://example.com"}'},
    {"role": "assistant", "content": "Sir, he wishes to reach example.com — please attend."},
    {"role": "user",
     "content": '{"notification_type":"idle_prompt","tool_name":null,"summary":""}'},
    {"role": "assistant", "content": "Sir, Claude awaits your guidance."},
]


def build_messages(
    event: Event,
    lang: Lang,
    summary: str,
    target_chars: int,
    hard_cap: int,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for an Event.

    `summary` is the already-extracted-and-redacted one-line digest of
    `event.tool_input`. The raw `tool_input` is NOT passed to the LLM.
    """
    if lang == "zh":
        sys = _SYSTEM_BASE.format(
            addr="先生", lang_name="中文",
            target_chars=target_chars, hard_cap=hard_cap,
        )
        few_shot = _FEW_SHOT_ZH
    else:
        sys = _SYSTEM_BASE.format(
            addr="Sir", lang_name="English",
            target_chars=target_chars, hard_cap=hard_cap,
        )
        few_shot = _FEW_SHOT_EN

    user_blob = json.dumps(
        {
            "notification_type": event.notification_type,
            "tool_name": event.tool_name,
            "summary": summary,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": sys}, *few_shot, {"role": "user", "content": user_blob}]
```

- [ ] **Step 4: Run prompt tests**

```bash
uv run pytest tests/unit/test_prompt.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jarvis_cc/phrase/prompt.py tests/unit/test_prompt.py
git commit -m "feat(phrase): prompt.py uses summary + soft target/hard cap"
```

After this commit, `pytest` overall will be RED because the providers and the router still call `build_messages(event, lang, max_chars)`. The next tasks fix that. Don't stop here.

---

## Task 5: Change `PhraseProvider.generate` signature

**Files:**
- Modify: `src/jarvis_cc/phrase/providers/base.py`

- [ ] **Step 1: Update the abstract base**

Replace the contents of `src/jarvis_cc/phrase/providers/base.py`:

```python
"""Abstract base for LLM phrase providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PhraseProvider(ABC):
    """A provider returns a single Jarvis-tone sentence given pre-built
    OpenAI-compatible chat `messages`. The router is responsible for
    constructing `messages` (extract + redact + build_messages); providers
    are dumb HTTP adapters.
    """

    name: str

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str: ...

    async def healthcheck(self) -> bool:
        return True
```

- [ ] **Step 2: Run base import to confirm syntax**

```bash
uv run python -c "from jarvis_cc.phrase.providers.base import PhraseProvider; print('ok')"
```
Expected: `ok`.

(Don't commit yet — concrete provider subclasses are temporarily broken; commit together in Task 6.)

---

## Task 6: Adapt each provider implementation

**Files:**
- Modify: `src/jarvis_cc/phrase/providers/deepseek.py`
- Modify: `src/jarvis_cc/phrase/providers/openai.py`
- Modify: `src/jarvis_cc/phrase/providers/anthropic.py`
- Modify: `src/jarvis_cc/phrase/providers/ollama.py`

- [ ] **Step 1: Rewrite `deepseek.py`**

Replace the contents of `src/jarvis_cc/phrase/providers/deepseek.py`:

```python
"""DeepSeek-Chat provider (OpenAI-compatible chat API)."""
from __future__ import annotations

import os

import httpx

from ...config import DeepSeekConfig
from .base import PhraseProvider


class DeepSeekProvider(PhraseProvider):
    name = "deepseek"

    def __init__(self, cfg: DeepSeekConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
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

- [ ] **Step 2: Rewrite `openai.py`**

Replace the contents of `src/jarvis_cc/phrase/providers/openai.py`:

```python
"""OpenAI chat-completions provider."""
from __future__ import annotations

import os

import httpx

from ...config import OpenAIConfig
from .base import PhraseProvider


class OpenAIProvider(PhraseProvider):
    name = "openai"

    def __init__(self, cfg: OpenAIConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
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

- [ ] **Step 3: Rewrite `anthropic.py`**

Replace the contents of `src/jarvis_cc/phrase/providers/anthropic.py`:

```python
"""Anthropic Claude provider via raw HTTP (avoids SDK pinning issues)."""
from __future__ import annotations

import os

import httpx

from ...config import AnthropicConfig
from .base import PhraseProvider


class AnthropicProvider(PhraseProvider):
    name = "anthropic"

    def __init__(self, cfg: AnthropicConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
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

- [ ] **Step 4: Rewrite `ollama.py`**

Replace the contents of `src/jarvis_cc/phrase/providers/ollama.py`:

```python
"""Ollama local LLM provider (uses /api/chat)."""
from __future__ import annotations

import httpx

from ...config import OllamaConfig
from .base import PhraseProvider


class OllamaProvider(PhraseProvider):
    name = "ollama"

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    async def generate(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.7, "num_predict": 200},
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

- [ ] **Step 5: Update provider tests for new signature**

Edit `tests/unit/test_phrase_providers.py` and `tests/unit/test_phrase_providers_more.py` — every call site that invokes `provider.generate(event, lang, max_chars)` must become `provider.generate(messages)`, where `messages` is a small canned list. Quick way to find and fix:

```bash
uv run grep -n "generate(" tests/unit/test_phrase_providers*.py
```

For each match, replace `provider.generate(event, lang, 30)` with:

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm /tmp/x"}'},
]
out = await provider.generate(messages)
```

(Use any non-empty system+user pair; the providers don't inspect content.) Remove any `from jarvis_cc.phrase.prompt import build_messages` calls in tests if they pass `max_chars=30`.

- [ ] **Step 6: Run provider tests**

```bash
uv run pytest tests/unit/test_phrase_providers.py tests/unit/test_phrase_providers_more.py -v
```
Expected: all PASS. If a test still references the old signature, fix it inline using the recipe above.

- [ ] **Step 7: Commit**

```bash
git add src/jarvis_cc/phrase/providers/ tests/unit/test_phrase_providers*.py
git commit -m "refactor(phrase): provider.generate takes pre-built messages"
```

After this commit, the router still passes the old shape. Test suite remains RED until Task 7 lands.

---

## Task 7: Wire `PhraseRouter` end-to-end + drop post-truncation

**Files:**
- Modify: `src/jarvis_cc/phrase/router.py`
- Test: `tests/unit/test_router.py` (existing — update)
- Test: `tests/unit/test_phrase_router_content_aware.py` (new)

- [ ] **Step 1: Update existing router tests for new stub signature**

Replace the contents of `tests/unit/test_router.py`:

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
        self.last_messages: list[dict[str, str]] | None = None

    async def generate(self, messages):
        self.calls += 1
        self.last_messages = messages
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

- [ ] **Step 2: Write content-aware behaviour tests**

Create `tests/unit/test_phrase_router_content_aware.py`:

```python
import pytest

from jarvis_cc.config import Config
from jarvis_cc.phrase.providers.base import PhraseProvider
from jarvis_cc.phrase.router import PhraseRouter
from jarvis_cc.types import Event


class _CapturingStub(PhraseProvider):
    name = "cap"

    def __init__(self, output: str = "<ok>") -> None:
        self.output = output
        self.last_messages: list[dict[str, str]] | None = None

    async def generate(self, messages):
        self.last_messages = messages
        return self.output


@pytest.mark.asyncio
async def test_router_passes_extracted_summary_into_user_blob():
    stub = _CapturingStub()
    router = PhraseRouter(stub, None, Config())
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "rm -rf /tmp/xyz"},
    )
    await router.phrase(ev, lang="en")
    assert stub.last_messages is not None
    user_blob = stub.last_messages[-1]["content"]
    assert "rm -rf /tmp/xyz" in user_blob
    # Raw key "tool_input" must NOT be in the prompt
    assert "tool_input" not in user_blob


@pytest.mark.asyncio
async def test_router_redacts_home_path_when_enabled(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/jobin")
    # Reload redact module so _HOME picks up the patched env
    import importlib, jarvis_cc.phrase.redact as r
    importlib.reload(r)

    stub = _CapturingStub()
    cfg = Config()
    cfg.behavior.privacy.cloud_redaction = True
    router = PhraseRouter(stub, None, cfg)
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "rm -rf /Users/jobin/tmp/x"},
    )
    await router.phrase(ev, lang="en")
    blob = stub.last_messages[-1]["content"]
    assert "rm -rf ~/tmp/x" in blob
    assert "/Users/jobin" not in blob


@pytest.mark.asyncio
async def test_router_skips_redaction_when_disabled(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/jobin")
    import importlib, jarvis_cc.phrase.redact as r
    importlib.reload(r)

    stub = _CapturingStub()
    cfg = Config()
    cfg.behavior.privacy.cloud_redaction = False
    router = PhraseRouter(stub, None, cfg)
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "ls /Users/jobin/tmp"},
    )
    await router.phrase(ev, lang="en")
    blob = stub.last_messages[-1]["content"]
    assert "/Users/jobin/tmp" in blob


@pytest.mark.asyncio
async def test_router_does_not_post_truncate_long_llm_output():
    long_output = "Sir, " + "x" * 200  # ~205 chars, well beyond old 30 cap
    stub = _CapturingStub(output=long_output)
    router = PhraseRouter(stub, None, Config())
    out = await router.phrase(
        Event(notification_type="idle_prompt", tool_name=None),
        lang="en",
    )
    assert out == long_output  # unchanged — no router-side truncation


@pytest.mark.asyncio
async def test_router_empty_tool_input_passes_empty_summary():
    stub = _CapturingStub()
    router = PhraseRouter(stub, None, Config())
    ev = Event(notification_type="idle_prompt", tool_name=None, tool_input={})
    await router.phrase(ev, lang="en")
    blob = stub.last_messages[-1]["content"]
    assert '"summary": ""' in blob or '"summary":""' in blob
```

- [ ] **Step 3: Run router tests — they should fail**

```bash
uv run pytest tests/unit/test_router.py tests/unit/test_phrase_router_content_aware.py -v
```
Expected: FAIL — router still passes `(event, lang, max_chars)` and post-truncates.

- [ ] **Step 4: Rewrite `phrase/router.py`**

Replace the contents of `src/jarvis_cc/phrase/router.py`:

```python
"""Provider routing: try primary, then fallback, then template.

Owns the extract → redact → build_messages pipeline so providers stay
dumb HTTP adapters that just take pre-built messages and return a string.
"""
from __future__ import annotations

from loguru import logger

from ..config import Config
from ..types import Event, Lang
from . import extract, redact
from .prompt import build_messages
from .providers.base import PhraseProvider
from .templates import render_template


class PhraseRouter:
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
        summary = extract.extract(event.tool_name, event.tool_input)
        summary = redact.scrub(
            summary,
            enabled=self.cfg.behavior.privacy.cloud_redaction,
        )
        messages = build_messages(
            event, lang, summary,
            target_chars=self.cfg.behavior.phrase_target_chars,
            hard_cap=self.cfg.behavior.phrase_hard_cap,
        )
        for provider in (self.primary, self.fallback):
            if provider is None:
                continue
            try:
                out = await provider.generate(messages)
                if out and out.strip():
                    return out.strip()
            except Exception as exc:
                logger.warning(
                    "Phrase provider {} failed: {}", provider.name, exc
                )
        return render_template(event, lang)
```

- [ ] **Step 5: Run the router tests**

```bash
uv run pytest tests/unit/test_router.py tests/unit/test_phrase_router_content_aware.py -v
```
Expected: 9 tests PASS (4 existing-style + 5 new).

- [ ] **Step 6: Run the full unit + integration suite**

```bash
uv run pytest -v
```
Expected: all 63 previously-existing tests still PASS, plus the new ~30 tests added in Tasks 1–7. Total ~93 PASS, 0 FAIL.

If any test fails, fix it in place. Do not skip and do not commit a red suite.

- [ ] **Step 7: Commit**

```bash
git add src/jarvis_cc/phrase/router.py tests/unit/test_router.py tests/unit/test_phrase_router_content_aware.py
git commit -m "feat(phrase): router wires extract → redact → build_messages end-to-end"
```

---

## Task 8: Default config + README deprecation note

**Files:**
- Modify: `src/jarvis_cc/install.py` (the `_default_config_toml` function around line 285)
- Modify: `README.md`

- [ ] **Step 1: Update the default config TOML emitted by `jarvis-cc install`**

Edit `src/jarvis_cc/install.py`. Replace the `_default_config_toml` function body's `[behavior]` section. Find:

```python
        [behavior]
        dedup_window_seconds = 10
        queue_max_size = 5
        voice_language = "auto"
        events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
        phrase_max_chars = 30
        """
```

Replace with:

```python
        [behavior]
        dedup_window_seconds = 10
        queue_max_size = 5
        voice_language = "auto"
        events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
        # phrase_max_chars is deprecated and ignored; use the budget below.
        phrase_target_chars = 70
        phrase_hard_cap = 120

        [behavior.privacy]
        cloud_redaction = true
        """
```

- [ ] **Step 2: Confirm install tests still pass**

```bash
uv run pytest tests/unit/test_install.py -v
```
Expected: all PASS. (`test_install.py` does not assert on TOML body contents — verify before relying on this.)

- [ ] **Step 3: Add a README note about deprecation + new keys**

Edit `README.md`. In the `[behavior]` config example near the top of the `## Configuration` section, replace `phrase_max_chars = 30` with the new keys. Find:

```toml
[behavior]
dedup_window_seconds = 10
queue_max_size = 5
voice_language = "auto"        # auto | zh | en
events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
phrase_max_chars = 30
```

Replace with:

```toml
[behavior]
dedup_window_seconds = 10
queue_max_size = 5
voice_language = "auto"        # auto | zh | en
events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
phrase_target_chars = 70       # LLM aims for this length
phrase_hard_cap = 120          # LLM is told not to exceed this; no post-truncation

[behavior.privacy]
cloud_redaction = true         # scrub HOME path + secret-shaped tokens before send
```

Also add this sentence to the **Troubleshooting** section as the last bullet:

```markdown
**Jarvis says the wrong thing about my command.** Content-awareness pipes `tool_input` (e.g. the actual Bash command, the file basename) into the LLM prompt. If the line still feels generic, check `tail ~/.jarvis-cc/logs/daemon.stderr.log` for whether the provider call succeeded — when LLMs error out, the daemon falls back to the generic template. The `phrase_max_chars` key in older configs is silently ignored; set `phrase_target_chars` / `phrase_hard_cap` instead.
```

- [ ] **Step 4: Commit**

```bash
git add src/jarvis_cc/install.py README.md
git commit -m "docs(config): default new install with phrase budget + privacy keys"
```

---

## Task 9: End-to-end smoke test

**Files:** none modified — verification only.

- [ ] **Step 1: Full test suite green**

```bash
uv run pytest -v
```
Expected: all tests PASS, 0 FAIL, 0 ERROR.

- [ ] **Step 2: Lint clean**

```bash
uv run ruff check src/ tests/
```
Expected: `All checks passed!`. If ruff reports new issues introduced by this work, fix them before continuing.

- [ ] **Step 3: Reload daemon so it picks up the new code**

```bash
launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis-cc.plist
launchctl load   ~/Library/LaunchAgents/com.jobin.jarvis-cc.plist
sleep 1
uv run jarvis-cc status
```
Expected: status JSON returns; daemon healthy.

- [ ] **Step 4: Fire a synthetic Bash permission prompt and listen**

```bash
# Send an event with a real-looking command in tool_input
python -c "
import json, socket, os
payload = {
    'notification_type': 'permission_prompt',
    'tool_name': 'Bash',
    'tool_input': {'command': 'rm -rf /tmp/jarvis-smoke-test'},
    'cwd': os.getcwd(),
    'session_id': 'smoke',
}
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(os.path.expanduser('~/.jarvis-cc/jarvis.sock'))
s.sendall((json.dumps(payload) + '\n').encode())
s.close()
print('sent')
"
```
Expected (audio): Jarvis names the command — e.g. *"Sir, he intends `rm -rf /tmp/jarvis-smoke-test`..."* — instead of a generic line. Confirm by ear.

- [ ] **Step 5: Verify daemon log shows summary going through**

```bash
tail -n 30 ~/.jarvis-cc/logs/daemon.stderr.log
```
Expected: a recent log line showing the phrase produced; no provider error traces tied to the smoke event.

- [ ] **Step 6: Re-listen with redaction (HOME path)**

```bash
python -c "
import json, socket, os
payload = {
    'notification_type': 'permission_prompt',
    'tool_name': 'Write',
    'tool_input': {'file_path': os.path.expanduser('~/proj/config.toml'), 'content': '...'},
    'cwd': os.getcwd(),
    'session_id': 'smoke-write',
}
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(os.path.expanduser('~/.jarvis-cc/jarvis.sock'))
s.sendall((json.dumps(payload) + '\n').encode())
s.close()
print('sent')
"
```
Expected (audio): the file basename `config.toml` is named; HOME path does NOT appear in the daemon log's prompt blob. Open the stderr log to verify — `grep config.toml ~/.jarvis-cc/logs/daemon.stderr.log` should show the file name appearing in the LLM message, but the absolute `~/proj/...` shape (with `~` prefix) rather than `/Users/...`.

- [ ] **Step 7: Final commit (only if any tweaks were needed during smoke)**

```bash
git status
# if any files changed:
git add -A
git commit -m "fix: smoke-test follow-ups for content-aware announcements"
```

---

## Self-Review Notes (filled in by plan author)

**Spec coverage:**
- ✅ Per-tool extractor — Task 3.
- ✅ Redactor (HOME + 5 secret patterns + 200-char cap + `enabled=False` truncation-only path) — Task 2.
- ✅ Prompt rewrite with summary field, target_chars/hard_cap, expanded few-shot — Task 4.
- ✅ Router drops post-truncation, drives the pipeline — Task 7.
- ✅ Config: new `phrase_target_chars`/`phrase_hard_cap`/`PrivacyConfig`, legacy `phrase_max_chars` kept — Task 1.
- ✅ `install.py` writes new keys; legacy users unaffected — Task 8.
- ✅ All 4 fallback-matrix rows verified — empty tool_input (extract returns "", Task 3 test); unknown tool (JSON-dump fallback, Task 3 test); provider error (Task 7 existing tests still cover); `say --text` untouched (router not on that path, no test needed, but the spec asserts and code path proves it).
- ✅ Test files: extract, redact, router-content-aware are new; prompt, router (existing), config (existing) updated.

**Placeholder scan:** no "TBD", no "implement later", every code step has full source. Test bodies are complete, not "similar to above".

**Type consistency:**
- `extract.extract(tool_name: str | None, tool_input: dict | None) -> str` consistent across Task 3, Task 7.
- `redact.scrub(text: str, *, enabled: bool = True) -> str` consistent across Task 2, Task 7.
- `build_messages(event, lang, summary, target_chars, hard_cap)` consistent across Tasks 4, 7.
- `PhraseProvider.generate(messages: list[dict[str, str]]) -> str` consistent across Tasks 5, 6, 7.
- Config field names (`phrase_target_chars`, `phrase_hard_cap`, `privacy.cloud_redaction`) consistent across Tasks 1, 7, 8.

**Scope check:** single coherent feature, ~9 tasks, all in `phrase/` package + thin config/install touches. Implementable in one focused session.
