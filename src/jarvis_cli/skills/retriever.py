"""Query the skill index: embed the prompt, rank by a hybrid of cosine
similarity and a lexical-overlap boost.

Pure cross-lingual embeddings miss skills whose match hinges on a shared proper
noun (a Chinese prompt naming "vercel"/"vue"/"git" against an English
description). A small additive lexical boost — strong when the query token hits
the skill *name*, weak for description hits — recovers exactly those cases
without letting lexical noise outvote semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import SkillRecord, deslug, lexical_tokens
from .embedder import Embedder
from .index import SkillIndex

# Additive boosts on top of cosine (which is ~[-0.1, 0.7] here).
_NAME_HIT_BOOST = 0.22  # query shares a token with the (de-slugged) skill name
_DESC_HIT_STEP = 0.05   # per query token also found in description/keywords
_DESC_HIT_CAP = 3


@dataclass
class Match:
    record: SkillRecord
    score: float  # hybrid score (cosine + lexical boost)
    cosine: float = 0.0


class SkillRetriever:
    def __init__(self, embedder: Embedder, index: SkillIndex) -> None:
        self._embedder = embedder
        self._index = index
        # Precompute per-skill lexical token sets once (cheap, ~tens of skills).
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
        """Top-`k` skills by hybrid score, highest first. Vectors are
        L2-normalized, so the dot product is the cosine."""
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
