"""MCP intent verification — re-exports the generic verifier.

All logic lives in ``retrieval.verifier``; this module keeps the existing
import path (``from jarvis.mcp.verifier import verify_candidates``)
working. MCP callers pass ``noun="tool server"`` (the default).
"""
from ..retrieval.verifier import (
    CONFIRMED,
    NONE,
    UNCLEAR,
    VerifyResult,
    verify_candidates,
)

__all__ = ["verify_candidates", "VerifyResult", "CONFIRMED", "NONE", "UNCLEAR"]
