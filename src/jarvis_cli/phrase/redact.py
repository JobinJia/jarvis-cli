"""Light redaction applied to extracted summary before sending to LLM.

Substitutes the user's home directory and common secret-shaped tokens
with placeholders, then truncates to a hard length cap. NOT a security
guarantee — best-effort defence-in-depth so cloud LLM providers don't
see paths or accidentally-pasted keys in Jarvis's prompt.
"""
from __future__ import annotations

import os
import re

_MAX_OUT = 200

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk_[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),
]

# Tokens that are legitimate context but unspeakable: TTS spells them out
# character by character, so a 36-char UUID becomes ~26 seconds of alphabet
# soup (the "cicada buzz" incident, 2026-08-28 — a tool_failure line carried
# a raw session id straight into XTTS). Shortened to a 4-char stub the voice
# can say in under a second, keeping just enough identity to correlate with
# logs. Unlike the secret patterns above, these apply regardless of the
# privacy toggle — they protect the listener's ears, not their keys.
_UNSPEAKABLE = [
    # UUID — its hyphens split it into sub-12-char segments, so the run
    # pattern below can't catch it whole.
    re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    # Digit-containing alnum runs of 12+ (Figma keys like
    # hp6VI6CyREYbtiukMR6BmN, epoch-ms timestamps, hex hashes, base64-ish
    # ids): read letter-by-letter at ~5 cps, they crawl — the 2026-08-29
    # "half a beat slow" report. The digit requirement spares long ordinary
    # words; a digit-free hex run slips through, but real hashes without a
    # single digit are vanishingly rare.
    re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{12,}(?![A-Za-z0-9])"),
]


def _stub(m: re.Match[str]) -> str:
    return m.group(0)[:4]


def speakable(text: str) -> str:
    """Shorten unspeakable ID tokens (UUIDs, long alnum id runs) to a
    4-char stub.

    Applied both to summaries headed for the phrase LLM and, as a second
    gate, to final lines headed for TTS — the LLM can also invent or echo
    an ID from elsewhere in its context.
    """
    if not text:
        return text
    out = text
    for p in _UNSPEAKABLE:
        out = p.sub(_stub, out)
    return out


def scrub(text: str, *, enabled: bool = True) -> str:
    """Return a possibly-redacted, length-capped version of `text`.

    Length cap (200 chars) and unspeakable-ID shortening always apply;
    secret-pattern substitutions only when `enabled` is True.
    """
    if not text:
        return text
    out = text
    if enabled:
        home = os.path.expanduser("~")
        if home and home != "/":
            out = out.replace(home, "~")
        for p in _PATTERNS:
            out = p.sub("<REDACTED>", out)
    return speakable(out)[:_MAX_OUT]
