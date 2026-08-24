"""Saturation criterion (THREATS-TO-VALIDITY.md §1.1, pre-registered before data
collection): a configuration is saturated when, over a 30 s sustained window after
warm-up, either holds:

    ingest_lag p95 > thresholds.lag_p95_ms, or
    drop_rate    > thresholds.drop_rate_pct

`N_max` is the largest swept N where no camera is saturated. Both thresholds are
read from configs/thresholds.yaml, never hardcoded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from multicam_bench.config import Thresholds


@dataclass(frozen=True)
class SaturationResult:
    saturated: bool
    first_saturated_window_start_s: float | None
    reason: str | None


def percentile_ms(values_ms: Sequence[float], pct: float) -> float:
    if not values_ms:
        raise ValueError("values_ms must not be empty")
    return float(np.percentile(np.asarray(values_ms, dtype=float), pct))


def is_window_saturated(
    lag_p95_ms: float, drop_rate_pct: float, thresholds: Thresholds
) -> bool:
    """The pre-registered criterion applied to one already-aggregated window."""
    return lag_p95_ms > thresholds.lag_p95_ms or drop_rate_pct > thresholds.drop_rate_pct


def detect_saturation(
    lag_samples_ms: Sequence[tuple[float, float]],
    drop_events_t_s: Sequence[float],
    expected_fps: float,
    thresholds: Thresholds,
    window_s: float = 30.0,
) -> SaturationResult:
    """Slide non-overlapping `window_s`-wide windows across the measurement period.

    `lag_samples_ms` is `(t_s, lag_ms)` for every measured frame; `drop_events_t_s`
    is the timestamp of every dropped frame (queue-policy drops and embedded-index
    gaps alike — both are real frames the consumer never saw). `t_s` is seconds
    since the measurement anchor (post warm-up), so a window starting at 0 is
    already a full 30 s+ past warm-up by construction.

    Returns the first window (in time order) where the criterion holds, or
    `saturated=False` if none does.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if not lag_samples_ms:
        raise ValueError("lag_samples_ms must not be empty")
    if expected_fps <= 0:
        raise ValueError("expected_fps must be positive")

    end_t = max(t for t, _ in lag_samples_ms)
    expected_per_window = expected_fps * window_s

    window_start = 0.0
    while window_start < end_t:
        window_end = window_start + window_s
        window_lags = [lag for t, lag in lag_samples_ms if window_start <= t < window_end]
        window_drops = sum(1 for t in drop_events_t_s if window_start <= t < window_end)

        if window_lags:
            lag_p95 = percentile_ms(window_lags, 95)
            drop_rate = (window_drops / expected_per_window) * 100.0
            if is_window_saturated(lag_p95, drop_rate, thresholds):
                reason = (
                    f"lag_p95={lag_p95:.1f}ms drop_rate={drop_rate:.2f}% "
                    f"in window [{window_start:.0f}s, {window_end:.0f}s)"
                )
                return SaturationResult(
                    saturated=True,
                    first_saturated_window_start_s=window_start,
                    reason=reason,
                )
        window_start += window_s

    return SaturationResult(saturated=False, first_saturated_window_start_s=None, reason=None)


def compute_n_max(saturated_by_n: dict[int, bool]) -> int | None:
    """N_max = largest swept N where no camera was saturated.

    Returns None if every tested N was saturated, or no N was tested.
    """
    not_saturated = [n for n, saturated in saturated_by_n.items() if not saturated]
    return max(not_saturated) if not_saturated else None
