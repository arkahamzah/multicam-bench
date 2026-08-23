"""ingest_lag math (see THREATS-TO-VALIDITY.md §1.2).

    elapsed_frames(t) = unwrap(embedded_index)          # NOT the received-frame count
    lag                = t_recv − (t_anchor + elapsed_frames / fps_source)

`t_anchor` is set after the warm-up window, never on the first frame. Elapsed frame
count must come from the embedded index — deriving it from how many frames were
*received* under-reports lag exactly when frames are being dropped, which is
precisely the condition this metric exists to detect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    """Reference point set after warm-up: (embedded frame index, wall-clock time)."""

    frame_index: int
    t_wall: float


class LoopUnwrapper:
    """Reconstructs a monotonic frame index across loop boundaries.

    The publisher repeats a `period`-frame source (`ffmpeg -stream_loop -1`), so the
    embedded index resets to near zero every `period` frames. A backward jump larger
    than half the period is treated as a loop restart and a full period is added to
    every subsequent index; a small backward jump (marker decode noise) is not.
    """

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = period
        self._loops = 0
        self._last_raw: int | None = None

    def unwrap(self, raw_index: int) -> int:
        if self._last_raw is not None and raw_index < self._last_raw - self.period // 2:
            self._loops += 1
        self._last_raw = raw_index
        return self._loops * self.period + raw_index


def ingest_lag_s(anchor: Anchor, frame_index: int, t_recv: float, fps_source: float) -> float:
    """lag = t_recv - (t_anchor + elapsed_frames / fps_source), elapsed relative to anchor."""
    elapsed = frame_index - anchor.frame_index
    return t_recv - (anchor.t_wall + elapsed / fps_source)
