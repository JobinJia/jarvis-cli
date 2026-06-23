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


def whole_word_tokens(text: str) -> set[str]:
    """Whole-word tokens only — ASCII words and complete CJK runs, NO bigrams.

    The gate uses this to tell a *substantive* lexical hit from an incidental
    one. A shared proper noun ("vercel", "memex") or an explicitly-listed
    keyword is a whole word here; the CJK bigrams ``lexical_tokens`` also emits
    (so "更新文档" can overlap a keyword "文档") are not. Bigram-only overlaps
    still shape ranking, but must never on their own exempt a match from the
    semantic floor — that is what let common words like "更新"/"项目" drag in
    unrelated skills.
    """
    return {m for m in _TOKEN.findall((text or "").lower())}


def is_vague_query(
    text: str, *, min_cjk_chars: int = 6, min_ascii_words: int = 3
) -> bool:
    """True when a prompt is too short/generic for command-style body injection.

    CJK chars and ASCII words are counted on separate scales — one CJK char
    carries far more meaning than one ASCII letter. "更新一下" (4 CJK chars,
    0 words) is vague; "deploy to vercel now" (4 words) is not. Callers still
    allow a body when the match carries a whole-word lexical hit (a named
    target), so this only gates the *thin* prompts, not every short one.
    """
    t = (text or "").strip()
    cjk = sum(1 for ch in t if "一" <= ch <= "鿿")
    ascii_words = len(re.findall(r"[A-Za-z0-9]{2,}", t))
    return cjk < min_cjk_chars and ascii_words < min_ascii_words
