# Idle-prompt variety: LLM improvises a fresh line each time

**Date:** 2026-06-12
**Status:** Approved

## Problem

Every idle prompt speaks the identical line "Sir, Claude awaits your
guidance." Two compounding causes:

1. The idle_prompt LLM request is byte-identical every time, and the
   few-shot exemplar's answer for it is exactly that sentence — so the model
   (qwen3:8b, temperature already 0.7) parrots the exemplar.
2. The no-LLM fallback template is the same sentence again.

The user wants a freshly improvised line every time — explicitly NOT a
pre-written pool of output lines.

## Decision

Make every idle request's input different so the output varies naturally;
the sentence itself is always LLM-generated.

- **`flavor`**: each idle request carries a randomly chosen improvisation
  angle (chess move, theatre cue, situation report, …) from a small internal
  hint list. Hints are directions, not output lines.
- **`avoid`**: the previously spoken idle line is threaded into the request
  and the system prompt forbids reusing its phrasing.
- The few-shot idle exemplar is rewritten to include `flavor`/`avoid` and an
  answer that demonstrably follows the flavor, teaching the model the schema.

## Changes

### `src/jarvis/phrase/prompt.py`

- `_IDLE_FLAVORS`: ~12 short English angle hints; `_pick_flavor()` =
  `random.choice` (monkeypatch-friendly).
- `build_messages(..., avoid: str | None = None)`: for idle_prompt events the
  user blob gains `flavor` (always) and `avoid` (only when given), and the
  system prompt gains an idle-specific clause: improvise fresh, follow the
  flavor, never reuse the avoid phrasing. Non-idle events are untouched.
- EN/ZH few-shot idle exemplars updated to the new schema.

### `src/jarvis/phrase/router.py`

- `PhraseRouter` remembers `_last_idle_line`; passes it as `avoid` on the
  next idle request and updates it with whatever line was produced
  (LLM or template — the template line is also worth avoiding next time).

### Unchanged

- Fallback template (only heard when all LLM providers fail).
- All other event types' prompts.
- Daemon: loads code at start, so a `launchctl kickstart -k` restart applies it.

## Testing

`tests/unit/test_phrase_idle_variety.py`:

- idle user blob contains a `flavor` from `_IDLE_FLAVORS`; `avoid` present
  exactly when supplied; system prompt carries the idle clause.
- non-idle blobs gain no `flavor`/`avoid` keys.
- router threads call N's output into call N+1's `avoid` (capturing stub).
