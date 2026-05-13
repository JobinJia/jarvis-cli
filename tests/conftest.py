"""Test infrastructure for jarvis-cc.

Pytest's default `tmp_path` on macOS resolves to a long `/private/var/folders/...`
prefix, which exceeds AF_UNIX's 108-byte path limit and breaks unix-socket tests.
We override the tmp_path_factory base to a short prefix under `/tmp/`.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _short_tmp_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Force a short tmp root so AF_UNIX socket paths fit in 108 bytes."""
    # Resolve symlinks (e.g. /tmp -> /private/tmp on macOS) so pytest's
    # internal `_ensure_relative_to_basetemp` check, which compares against
    # the resolved parent, succeeds.
    short = Path(tempfile.mkdtemp(prefix="jarvis-cc-", dir="/tmp")).resolve()
    # Repoint the factory at our short directory
    tmp_path_factory._basetemp = short  # type: ignore[attr-defined]
    return short
