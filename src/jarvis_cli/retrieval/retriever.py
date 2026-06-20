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
from .text import deslug, lexical_tokens

_NAME_HIT_BOOST = 0.22
_DESC_HIT_STEP = 0.05
_DESC_HIT_CAP = 3


@dataclass
class Match:
    record: Any
    score: float  # hybrid score (cosine + lexical boost)
    cosine: float = 0.0


class Retriever:
    def __init__(self, embedder: Embedder, index: Index) -> None:
        self._embedder = embedder
        self._index = index
        self._name_tokens: list[set[str]] = []
        self._desc_tokens: list[set[str]] = []
        for r in index.records:
            self._name_tokens.append(lexical_tokens(deslug(r.name)))
            self._desc_tokens.append(
                lexical_tokens(r.description + " " + " ".join(r.keywords))
            )

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
        hybrid = cosine + np.array(
            [self._lexical_boost(i, qtok) for i in range(len(cosine))],
            dtype=np.float32,
        )
        k = min(k, len(self._index.records))
        top = np.argpartition(hybrid, -k)[-k:]
        top = top[np.argsort(hybrid[top])[::-1]]
        return [
            Match(self._index.records[i], float(hybrid[i]), float(cosine[i]))
            for i in top
        ]
