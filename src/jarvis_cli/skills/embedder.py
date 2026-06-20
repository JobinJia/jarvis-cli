"""Backward-compatibility re-export.

The embedding model now lives in the shared retrieval package; domain
modules (skills, mcp) import from there.  This shim keeps existing
``from jarvis_cli.skills.embedder import ...`` paths working.
"""
from ..retrieval.embedder import DEFAULT_MODEL, Embedder, EmbedderUnavailable

__all__ = ["DEFAULT_MODEL", "Embedder", "EmbedderUnavailable"]
