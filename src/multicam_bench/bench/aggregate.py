"""Median/IQR aggregation (CLAUDE.md hard rule 4: "No `mean` as a headline metric.
p50/p95/p99 or nothing.") and loop-boundary exclusion (THREATS-TO-VALIDITY.md T4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MedianIQR:
    median: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def median_iqr(values: Sequence[float]) -> MedianIQR:
    """Median and interquartile range (linear-interpolated percentiles) of `values`."""
    if not values:
        raise ValueError("values must not be empty")
    q1, median, q3 = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return MedianIQR(median=float(median), q1=float(q1), q3=float(q3))


def exclude_near_loop_boundary(
    frame_indices: Sequence[int], period_frames: int, margin: int = 5
) -> tuple[list[int], int]:
    """Drop frames within `margin` of a loop boundary; return (kept, excluded_count).

    `frame_indices` are the unwrapped (monotonic) indices produced by
    `bench.lag.LoopUnwrapper` — a boundary is any multiple of `period_frames`, since
    that is where the source's `-stream_loop -1` restarts and timestamp/GOP
    discontinuities can produce lag spikes unrelated to system load.
    """
    if period_frames <= 0:
        raise ValueError("period_frames must be positive")
    if margin < 0:
        raise ValueError("margin must not be negative")

    kept: list[int] = []
    excluded = 0
    for idx in frame_indices:
        offset = idx % period_frames
        near_boundary = offset < margin or offset >= period_frames - margin
        if near_boundary:
            excluded += 1
        else:
            kept.append(idx)
    return kept, excluded
