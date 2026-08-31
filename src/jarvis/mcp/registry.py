"""MCP server registry: load, validate, and search server descriptions.

The registry is a hand-curated JSON file listing all *known* MCP servers
with their capabilities and connection config.  Only a subset may be
connected at any time — the intent router matches user prompts against
descriptions and injects connection instructions for the best match.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ..retrieval.text import deslug


@dataclass
class McpServerRecord:
    """One known MCP server.  ``connect`` holds the ``add_server`` params
    (type, url, command, args, env) ready to be spread into the MCP router
    tool call."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    connect: dict = field(default_factory=dict)
    content_hash: str = ""

    def text_for_embedding(self) -> str:
        words = deslug(self.name)
        parts = [words, words, self.description]
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return "\n".join(p for p in parts if p).strip()

    def key(self) -> str:
        return self.name


def _compute_hash(r: McpServerRecord) -> str:
    sig = f"{r.name}\0{r.description}\0{','.join(r.keywords)}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def record_to_dict(r: McpServerRecord) -> dict:
    return {
        "name": r.name,
        "description": r.description,
        "keywords": r.keywords,
        "connect": r.connect,
        "content_hash": r.content_hash,
    }


def record_from_dict(d: dict) -> McpServerRecord:
    return McpServerRecord(
        name=d["name"],
        description=d.get("description", ""),
        keywords=list(d.get("keywords", [])),
        connect=dict(d.get("connect", {})),
        content_hash=d.get("content_hash", ""),
    )


def load_registry(path: str | Path) -> list[McpServerRecord]:
    """Load the server registry JSON and return validated records."""
    p = Path(path)
    if not p.exists():
        logger.debug("mcp: registry not found at {}", p)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("mcp: failed to read registry {} ({})", p, exc)
        return []

    servers = data if isinstance(data, list) else data.get("servers", [])
    records: list[McpServerRecord] = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        rec = McpServerRecord(
            name=name,
            description=str(entry.get("description") or "").strip(),
            keywords=[str(k).strip() for k in (entry.get("keywords") or [])],
            connect=dict(entry.get("connect") or {}),
        )
        rec.content_hash = _compute_hash(rec)
        records.append(rec)

    logger.debug("mcp: loaded {} servers from registry", len(records))
    return records


def save_registry(records: list[McpServerRecord], path: str | Path) -> None:
    """Write the registry back to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "servers": [
            {
                "name": r.name,
                "description": r.description,
                "keywords": r.keywords,
                "connect": r.connect,
            }
            for r in records
        ]
    }
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
