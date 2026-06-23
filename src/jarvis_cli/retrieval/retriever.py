"""Hybrid cosine + lexical retriever, generic over record types.

Records must provide:
  ``.name``         (str)
  ``.description``  (str)
  ``.keywords``     (list[str])

Pure cross-lingual embeddings miss matches that hinge on a shared proper
noun (a Chinese prompt naming "vercel"/"memex" against an English
description). A small additive lexical boost recovers exactly those cases
without letting lexical noise outvote semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .embedder import Embedder
from .index import Index
from .text import deslug, lexical_tokens, whole_word_tokens

_NAME_HIT_BOOST = 0.22
_DESC_HIT_STEP = 0.05
_DESC_HIT_CAP = 3

# A lexical match must be a whole-word hit OR clear this cosine on its own to
# pass the gate; a bigram-only overlap on a common CJK word does not qualify.
_COSINE_SOLO_FLOOR = 0.50


@dataclass
class Match:
    record: Any
    score: float  # hybrid score (cosine + lexical boost)
    cosine: float = 0.0
    # True when a whole-word lexical hit (proper noun / explicit keyword), not
    # just a CJK bigram overlap, backed the boost. Gates the semantic-floor
    # exemption so common-word overlaps can't drag in unrelated records.
    whole_word: bool = False


class Retriever:
    def __init__(self, embedder: Embedder, index: Index) -> None:
        self._embedder = embedder
        self._index = index
        self._name_tokens: list[set[str]] = []
        self._desc_tokens: list[set[str]] = []
        # Whole-word counterparts (no bigrams) for the gate's signal test.
        self._name_whole: list[set[str]] = []
        self._desc_whole: list[set[str]] = []
        for r in index.records:
            name_ds = deslug(r.name)
            desc_kw = r.description + " " + " ".join(r.keywords)
            self._name_tokens.append(lexical_tokens(name_ds))
            self._desc_tokens.append(lexical_tokens(desc_kw))
            self._name_whole.append(whole_word_tokens(name_ds))
            self._desc_whole.append(whole_word_tokens(desc_kw))

    @property
    def size(self) -> int:
        return len(self._index.records)

    def _lexical_boost(self, i: int, qtok: set[str]) -> float:
        if not qtok:
            return 0.0
        boost = 0.0
        if qtok & self._name_tokens[i]:
            boost += _NAME_HIT_BOOST
        desc_hits = len(qtok & self._desc_tokens[i])
        boost += _DESC_HIT_STEP * min(desc_hits, _DESC_HIT_CAP)
        return boost

    def query(self, text: str, *, k: int = 5) -> list[Match]:
        """Top-*k* records by hybrid score, highest first."""
        text = (text or "").strip()
        if not text or self._index.vectors.size == 0:
            return []
        qv = self._embedder.embed_one(text)
        cosine = self._index.vectors @ qv
        qtok = lexical_tokens(text)
        qwhole = whole_word_tokens(text)
        hybrid = cosine + np.array(
            [self._lexical_boost(i, qtok) for i in range(len(cosine))],
            dtype=np.float32,
        )
        k = min(k, len(self._index.records))
        top = np.argpartition(hybrid, -k)[-k:]
        top = top[np.argsort(hybrid[top])[::-1]]
        return [
            Match(
                self._index.records[i],
                float(hybrid[i]),
                float(cosine[i]),
                whole_word=bool(
                    qwhole & self._name_whole[i]
                    or qwhole & self._desc_whole[i]
                ),
            )
            for i in top
        ]


def has_lexical_signal(m: Match) -> bool:
    """True when a whole-word lexical hit (not a bigram-only overlap) backed the
    boost — the only lexical evidence allowed to exempt a match from the
    semantic floor."""
    return m.whole_word


def gate_matches(
    matches: list[Match],
    *,
    med_threshold: float,
    cosine_solo_floor: float = _COSINE_SOLO_FLOOR,
) -> list[Match]:
    """Structural gate, shared by skills and MCP so the two never drift apart.

    Keep matches that clear *med_threshold* AND carry a whole-word lexical
    signal OR a strong-enough standalone cosine. Pure semantic near-misses and
    common-word bigram overlaps are dropped regardless of hybrid score.
    """
    return [
        m
        for m in matches
        if m.score >= med_threshold
        and (has_lexical_signal(m) or m.cosine >= cosine_solo_floor)
    ]
