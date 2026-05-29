"""Sliding-window dedup keyed by Event.dedup_key()."""
from __future__ import annotations

from ..types import Event


class DedupWindow:
    """In-memory dedup: same dedup_key within `window_seconds` is suppressed.

    The "last seen" timestamp slides forward on each duplicate hit, so a stream
    of identical events within the window stays suppressed until silence.
    """

    def __init__(self, window_seconds: int) -> None:
        self._window = float(window_seconds)
        self._last_seen: dict[str, float] = {}

    def is_duplicate(self, event: Event) -> bool:
        key = event.dedup_key()
        now = event.received_at
        prev = self._last_seen.get(key)
        if prev is not None and (now - prev) <= self._window:
            self._last_seen[key] = now
            return True
        self._last_seen[key] = now
        return False
