# Content-Aware Announcements — Design

**Date:** 2026-05-15
**Status:** Approved, pending implementation plan
**Scope:** Upgrade Jarvis phrase generation so the spoken sentence names the
salient thing the user must decide on (file, command, URL) instead of a generic
"Sir, please attend." Constrained to data already on the `Event` object —
specifically `tool_input` — with cloud-friendly redaction and no hard truncation.

## Motivation

Today `phrase/router.py` calls each LLM provider with `tool_input` serialised
into the user blob (see `phrase/prompt.py:42`), but the system prompt does
nothing to direct the model toward those fields and `phrase_max_chars = 30`
makes content callouts physically impossible. The result is uniform, low-
information sentences like *"Sir, Claude requests a tool"* even when the
underlying request is *"run `rm -rf /tmp/xyz`"*. Users away from the screen
cannot distinguish a benign file write from a destructive shell command.

## Goals

1. The spoken sentence names a concrete artefact (file basename, command verb,
   URL host, grep pattern) whenever `tool_input` carries one.
2. No regression for events without `tool_input` (idle, elicitation with empty
   input) — those keep their current generic phrasing.
3. Predictable token cost: large `tool_input` payloads (e.g. a Write tool with
   a 200 KB body) cannot blow up the LLM call.
4. Cloud LLM providers see redacted content by default (HOME path, secret-
   shaped tokens), behind a single config toggle.
5. Existing `say --text` literal-bypass path is untouched.

## Non-goals

- Reading `raw_message` or recent CC transcript context (deferred; see "Future
  work").
- Intelligent muting / priority classification (separate spec).
- New CC hook events (`Stop`, `SubagentStop`, `PreToolUse`, etc.).
- Removing the deprecated `phrase_max_chars` config key — kept for back-compat.

## Architecture

The change is confined to `src/jarvis_cc/phrase/`. Two new modules are inserted
between the router and the provider call; three existing files are lightly
modified. Everything outside `phrase/` (hook client, daemon, TTS, player,
installer, Event schema) is untouched.

```
listener ──Event──► phrase/router.py
                     │
                     ▼
      ┌─── extract.summary(tool_name, tool_input)  ── (new) phrase/extract.py
      │      per-tool normalised summary string
      │
      ▼
   redact.scrub(summary, enabled=cfg.behavior.privacy.cloud_redaction)
      │                                              ── (new) phrase/redact.py
      ▼
   prompt.build_messages(event, summary, lang,
                         target_chars, hard_cap)     ── prompt.py (rewritten)
      │
      ▼
   provider.generate(messages) → str (no post-truncation)
```

## Components

### `phrase/extract.py` (new, ~40 LOC)

Per-tool dispatcher. Returns a short string summarising the most salient field
in `tool_input`. Empty string when `tool_input` is empty or None (caller treats
this as "no content awareness possible — let the LLM produce its generic line").

Mappings (initial set):

| Tool      | Extracted summary                          |
|-----------|--------------------------------------------|
| Bash      | `tool_input["command"]` truncated to 200ch |
| Write     | `f"write {basename(file_path)}"`           |
| Edit / MultiEdit | `f"edit {basename(file_path)}"`     |
| Read      | `f"read {basename(file_path)}"`            |
| Grep      | `f"grep '{pattern[:80]}'"`                 |
| Glob      | `f"glob '{pattern[:80]}'"`                 |
| WebFetch / WebSearch | `f"fetch {url[:120]}"`          |
| *unknown* | `json.dumps(tool_input)[:200]`             |

The `_MAX_RAW = 200` constant caps any single extracted field before it ever
reaches the redactor.

### `phrase/redact.py` (new, ~30 LOC)

Single function `scrub(text: str, *, enabled: bool) -> str`. When `enabled`:

1. Replace `os.path.expanduser("~")` with literal `"~"`.
2. Substitute the following secret-shaped patterns with `<REDACTED>`:
   - `sk-[A-Za-z0-9_-]{16,}` (OpenAI / Anthropic)
   - `sk_[A-Za-z0-9_-]{16,}` (ElevenLabs etc.)
   - `ghp_[A-Za-z0-9]{20,}` (GitHub PAT)
   - `AKIA[0-9A-Z]{16}` (AWS access key id)
   - `xox[baprs]-[A-Za-z0-9-]{10,}` (Slack)
   - `(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])` (hex hash/token)
3. Truncate to `_MAX_OUT = 200` characters.

When `enabled = False`, only the truncation runs — a final guardrail against
unbounded prompts even if the user opts out of redaction.

### `phrase/prompt.py` (rewritten)

System prompt:

```text
You are J.A.R.V.I.S., Tony Stark's polite British AI butler.
Address the user as '{addr}'. Given a Claude Code event, reply with ONE short
sentence in {lang_name} that ALERTS the user AND names the salient thing they
need to decide on. Aim for roughly {target_chars} characters; you may go up to
{hard_cap} if needed to keep the key detail. Be calm, courteous, with a hint
of dry wit. If a 'summary' field is provided, weave its content into your
sentence (quote a file name, the command verb, or the pattern — whatever is
most actionable). Do NOT explain. Do NOT add quotes or labels around your
output.
```

User blob (replaces today's tool_input passthrough):

```json
{"notification_type": "permission_prompt",
 "tool_name": "Bash",
 "summary": "rm -rf /tmp/xyz"}
```

Few-shot grows to ~6 pairs per language, covering: Bash command, Write file,
Edit file, WebFetch URL, idle (empty summary), unknown-tool JSON blob. Existing
2-pair few-shot is replaced.

### `phrase/router.py` (modified)

- Calls `extract.extract(event.tool_name, event.tool_input)`.
- Calls `redact.scrub(summary, enabled=cfg.behavior.privacy.cloud_redaction)`.
- Passes the redacted summary into `prompt.build_messages` along with
  `target_chars` and `hard_cap` from config.
- **Removes** the post-generation `out[:max_chars]` behaviour. The LLM is
  trusted; the prompt carries the soft target.
- Template fallback (`render_template`) when all providers fail is unchanged.

### `config.py` (modified)

New fields on `BehaviorConfig`:

| Field                 | Default | Purpose                                  |
|-----------------------|---------|------------------------------------------|
| `phrase_target_chars` | 70      | Soft target injected into system prompt  |
| `phrase_hard_cap`     | 120     | Upper bound mentioned in system prompt   |
| `privacy.cloud_redaction` | true | Toggles `redact.scrub` patterns         |

The legacy `phrase_max_chars` field is **kept** in the TOML schema for
back-compat: existing user configs do not break, but the value is no longer
consulted at runtime. README marks it deprecated.

`install.py` writes the new keys into a fresh default config, but does NOT
overwrite an existing one. Code reads via `getattr(behavior, "phrase_target_chars", 70)`
etc. so partial configs work.

## Data flow examples

**Bash with `rm -rf`:**

- Event: `{notification_type: "permission_prompt", tool_name: "Bash", tool_input: {command: "rm -rf /Users/jobin/tmp/xyz"}}`
- Extract → `"rm -rf /Users/jobin/tmp/xyz"`
- Redact → `"rm -rf ~/tmp/xyz"`
- Prompt summary → `"rm -rf ~/tmp/xyz"`
- LLM (target 70) → *"Sir, he intends `rm -rf ~/tmp/xyz` — your verdict?"*

**Write to a new file:**

- Event: `tool_input: {file_path: "/Users/jobin/proj/config.toml", content: "..."}`
- Extract → `"write config.toml"`
- Redact → `"write config.toml"` (no secret patterns)
- LLM → *"Sir, Claude wishes to overwrite `config.toml` — shall I permit?"*

**Idle:**

- Event: `tool_name=None, tool_input={}`
- Extract → `""`
- Redact → `""`
- LLM (seeing empty summary in few-shot) → *"Sir, Claude awaits your guidance."*

**Unknown tool:**

- `tool_name="NewMcpServer__weird_tool"`, `tool_input={"foo": "bar"}`
- Extract → `'{"foo": "bar"}'`
- LLM still gets *something* to chew on.

## Fallback matrix

| Condition                              | Behaviour                              |
|----------------------------------------|----------------------------------------|
| `tool_input` empty / None              | Empty summary → idle-style LLM line    |
| Unknown tool name                      | JSON-dump-truncate-200 as summary      |
| Primary LLM provider errors            | Existing fallback provider (unchanged) |
| All providers fail                     | `render_template` (unchanged)          |
| `Event.text != None` (say --text path) | Bypasses extract / redact / prompt entirely — verbatim TTS (unchanged) |

## Testing

New test files:

- `tests/test_phrase_extract.py` — one case per mapped tool, one unknown-tool
  case, one empty-input case, one over-length truncation case.
- `tests/test_phrase_redact.py` — HOME replacement, each of the 5 secret
  patterns, 200-character truncation, `enabled=False` truncation-only path.
- `tests/test_phrase_prompt.py` — `build_messages` no longer contains raw
  `tool_input`; `summary` appears in user blob; `phrase_target_chars` and
  `phrase_hard_cap` are interpolated into system message.
- `tests/test_router_content_aware.py` — router calls extract → redact →
  prompt in order; verifies no post-truncation occurs on a 90-char LLM output.

Existing 63 tests must continue to pass.

## Out of scope (future specs)

- `raw_message` ingestion — the CC notification body often duplicates
  `tool_input` info, so deferred until we see real cases where it adds signal.
- Recent-conversation context reading from `~/.claude/projects/.../session.jsonl`
  — high coupling to CC internals; revisit if user demands it.
- Smart prioritisation / quiet hours — independent feature.
- Removal of `phrase_max_chars` — next major version.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| LLM ignores soft target and produces 250-char sentence | Prompt mentions a hard cap; in practice GPT-4 / DeepSeek / Claude all respect explicit numbered targets. If observed, escalate by adding `max_tokens` provider-side. |
| Redactor misses a novel secret format | Patterns are best-effort, not a security guarantee. Documented. Users with strict requirements can disable cloud providers or set `cloud_redaction = false` after carefully reading what they're sending. |
| Unknown tool dumps sensitive input | The 200-char truncation + secret-pattern scrub apply uniformly. Worst case: user sees a Jarvis sentence that quotes obscure JSON, which is no worse than today (since `tool_input` was already in the prompt). |
| User upgrades and existing config lacks new keys | `getattr(..., default)` everywhere; install.py does not overwrite. |

## Open questions

None — all clarifying questions answered during brainstorming on 2026-05-15.
