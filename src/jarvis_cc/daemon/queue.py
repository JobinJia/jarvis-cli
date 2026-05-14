"""Bounded FIFO event queue with drop-oldest semantics."""
from __future__ import annotations

import asyncio
from collections import deque

from loguru import logger

from ..types import Event


class BoundedEventQueue:
    """asyncio-compatible bounded FIFO. When full, drops oldest events."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._deque: deque[Event] = deque(maxlen=maxsize)
        self._cond = asyncio.Condition()
        self.dropped_count = 0

    @property
    def size(self) -> int:
        """Current number of queued events."""
        return len(self._deque)

    @property
    def maxsize(self) -> int:
        """Configured maximum capacity."""
        return self._maxsize

    async def put_or_drop(self, event: Event) -> None:
        async with self._cond:
            if len(self._deque) == self._maxsize:
                dropped = self._deque.popleft()
                self.dropped_count += 1
                logger.warning("Queue full, dropped event {}", dropped.dedup_key())
            self._deque.append(event)
            self._cond.notify()

    async def get(self) -> Event:
        async with self._cond:
            while not self._deque:
                await self._cond.wait()
            return self._deque.popleft()
