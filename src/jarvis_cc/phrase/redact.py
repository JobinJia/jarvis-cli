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


def scrub(text: str, *, enabled: bool = True) -> str:
    """Return a possibly-redacted, length-capped version of `text`.

    Length cap (200 chars) always applies; pattern substitutions only
    when `enabled` is True.
    """
    if not text:
        return text
    if not enabled:
        return text[:_MAX_OUT]
    home = os.path.expanduser("~")
    out = text
    if home and home != "/":
        out = out.replace(home, "~")
    for p in _PATTERNS:
        out = p.sub("<REDACTED>", out)
    return out[:_MAX_OUT]
