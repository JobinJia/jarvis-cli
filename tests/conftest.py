"""Test infrastructure for jarvis-cli.

Pytest's default tmp_path lives under /private/var/folders/... on macOS,
which exceeds AF_UNIX's 104-byte sun_path limit (Darwin) and breaks unix-
socket tests like the hook_client and listener integration tests.

We work around this by pinning pytest's temp root to a short path under
/tmp via the supported `PYTEST_DEBUG_TEMPROOT` env var. We honor user
overrides (`--basetemp` flag, pre-set `PYTEST_DEBUG_TEMPROOT`) and clean
up any directories we create.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_OWNED_TMP_ROOTS: list[Path] = []


def _cleanup_owned() -> None:
    for path in _OWNED_TMP_ROOTS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_owned)


def pytest_configure(config: pytest.Config) -> None:
    """Pin pytest's temp root under /tmp when not overridden by the user.

    This runs before TempPathFactory is constructed, so we can use the
    supported `PYTEST_DEBUG_TEMPROOT` env var instead of reaching into
    private `_basetemp` attributes.
    """
    if config.option.basetemp:
        return  # user passed --basetemp
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return  # user pre-set the env var
    short_root = Path(tempfile.mkdtemp(prefix="jarvis-cli-", dir="/tmp")).resolve()
    _OWNED_TMP_ROOTS.append(short_root)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(short_root)
