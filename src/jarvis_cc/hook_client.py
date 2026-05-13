"""Thin client invoked by Claude Code Notification hook.

Reads JSON payload from stdin, writes a single NDJSON line over the
configured Unix socket, and exits. Must never raise to stdout — Claude
Code reads stdout for hook decisions.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import IO

from .config import DEFAULT_CONFIG_PATH, load_config


def forward_event(stream: IO[str], sock_path: str | Path) -> bool:
    """Forward an NDJSON event from `stream` to the unix socket at `sock_path`.

    Returns True if successfully sent. Returns False on any failure
    (invalid JSON, socket missing, write error) — never raises.
    """
    sock_path = Path(sock_path)
    try:
        raw = stream.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False

    payload["_received_at"] = time.time()
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(str(sock_path))
        s.sendall(line)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def main() -> int:
    """Entry point registered as `jarvis-cc-hook` console_script."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    forward_event(sys.stdin, cfg.paths.socket)
    return 0  # Always 0; failures are silent to Claude Code


if __name__ == "__main__":
    raise SystemExit(main())
