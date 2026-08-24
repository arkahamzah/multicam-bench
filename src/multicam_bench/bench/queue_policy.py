"""Bounded, drop-oldest queue (CLAUDE.md hard rule 3: "No unbounded queues. `maxsize`
is always set, policy is drop-oldest, drops are always counted as a metric.")

Used inside each per-camera reader process to decouple frame capture from frame
processing: if processing falls behind capture, the oldest unconsumed frame is
evicted rather than letting the queue grow, and every eviction is counted. This is
what turns "we dropped it" into a number distinct from "the decoder skipped it"
(an embedded-index gap, see bench/lag.py) per THREATS-TO-VALIDITY.md T6.
"""

from __future__ import annotations

import queue


class BoundedDropOldestQueue[T]:
    """A `queue.Queue(maxsize=1)`-style buffer that evicts the oldest item on overflow
    instead of blocking the producer or raising.
    """

    def __init__(self, maxsize: int = 1) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._q: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self.drops = 0

    def put(self, item: T) -> bool:
        """Enqueue `item`. Returns True if an existing item was dropped to make room."""
        dropped = False
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                dropped = True
            except queue.Empty:
                pass
            self._q.put_nowait(item)
        if dropped:
            self.drops += 1
        return dropped

    def get(self, timeout: float | None = None) -> T:
        """Block for the next item. Raises `queue.Empty` on timeout."""
        return self._q.get(timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()
