"""Shared text utilities for the retrieval pipeline."""
from __future__ import annotations

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEP = re.compile(r"[-_:/.]+")
_TOKEN = re.compile(r"[a-z0-9]{2,}|[一-鿿]+")


def deslug(text: str) -> str:
    """Split a slug/identifier into space-separated words.

    ``deploy-to-vercel`` -> ``deploy to vercel``
    """
    return _SEP.sub(" ", _CAMEL.sub(" ", text)).strip()


def lexical_tokens(text: str) -> set[str]:
    """Extract ASCII words (>=2 chars), CJK runs, and CJK bigrams.

    CJK text has no whitespace word boundaries, so a query like
    "帮我看下之前的会话" is one contiguous run and would never intersect
    a keyword token "会话".  Extracting overlapping bigrams from long CJK
    runs recovers those matches without affecting ASCII tokens.
    """
    tokens: set[str] = set()
    for m in _TOKEN.findall((text or "").lower()):
        tokens.add(m)
        if len(m) > 2 and not m.isascii():
            for i in range(len(m) - 1):
                tokens.add(m[i : i + 2])
    return tokens
