"""Build, persist, and load an embedding index for any record type.

Records must provide:
  ``.text_for_embedding() -> str``
  ``.key() -> str``
  ``.content_hash``  (str attribute)

Serialization is pluggable: callers pass ``to_dict`` / ``from_dict`` functions
so each domain (skills, MCP) controls its own on-disk format.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .embedder import Embedder

_CATALOG_NAME = "catalog.json"
_VECTORS_NAME = "vectors.npy"


@dataclass
class Index:
    model_name: str
    records: list[Any]
    vectors: np.ndarray  # shape (len(records), dim), L2-normalized

    def signature(self) -> dict[str, str]:
        """key -> content_hash, for cheap staleness checks."""
        return {r.key(): r.content_hash for r in self.records}


def build_index(records: list[Any], embedder: Embedder) -> Index:
    texts = [r.text_for_embedding() for r in records]
    vectors = embedder.embed(texts) if texts else np.empty((0, 0), dtype=np.float32)
    logger.info("retrieval: built index for {} records", len(records))
    return Index(model_name=embedder.model_name, records=records, vectors=vectors)


def save_index(
    index: Index,
    index_dir: Path,
    to_dict: Callable[[Any], dict],
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": index.model_name,
        "records": [to_dict(r) for r in index.records],
    }
    (index_dir / _CATALOG_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(index_dir / _VECTORS_NAME, index.vectors)


def load_index(
    index_dir: Path,
    from_dict: Callable[[dict], Any],
) -> Index | None:
    index_dir = Path(index_dir)
    cat = index_dir / _CATALOG_NAME
    vec = index_dir / _VECTORS_NAME
    if not cat.exists() or not vec.exists():
        return None
    try:
        payload = json.loads(cat.read_text(encoding="utf-8"))
        vectors = np.load(vec)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("retrieval: failed to load index ({}); will rebuild", exc)
        return None
    records = [from_dict(d) for d in payload.get("records", [])]
    if len(records) != vectors.shape[0]:
        logger.warning("retrieval: index row mismatch; will rebuild")
        return None
    return Index(
        model_name=payload.get("model_name", ""), records=records, vectors=vectors
    )


def is_stale(index: Index, fresh: list[Any], model_name: str) -> bool:
    """True if the index no longer matches the on-disk records or the model."""
    if index.model_name != model_name:
        return True
    return index.signature() != {r.key(): r.content_hash for r in fresh}


def ensure_index(
    index_dir: Path,
    embedder: Embedder,
    fresh: list[Any],
    to_dict: Callable[[Any], dict],
    from_dict: Callable[[dict], Any],
) -> Index:
    """Load a current index from disk, or (re)build and persist it when the
    on-disk records or the model changed."""
    existing = load_index(index_dir, from_dict)
    if existing is not None and not is_stale(existing, fresh, embedder.model_name):
        logger.debug("retrieval: index up to date ({} records)", len(existing.records))
        return existing
    rebuilt = build_index(fresh, embedder)
    save_index(rebuilt, index_dir, to_dict)
    return rebuilt
