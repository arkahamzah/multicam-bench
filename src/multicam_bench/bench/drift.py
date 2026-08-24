"""Publisher pacing-drift measurement (THREATS-TO-VALIDITY.md T1).

`ffmpeg -re` paces frames at the source fps, but under CPU contention from N sibling
publisher/reader processes it may fail to keep pace. That failure looks identical to
consumer saturation in ingest_lag, so it must be measured and, when it exceeds
`publisher_drift_reject_ms` (configs/thresholds.yaml), the data point is rejected
rather than reported as a saturation result.

Drift is measured from the embedded index timeline, not from a frame-received count
(same rationale as bench/lag.py): the expected elapsed time between the first and
last sample is `(last_index - first_index) / fps_source`; drift is how far the
observed wall-clock elapsed time strays from that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftResult:
    drift_ms: float
    rejected: bool
    reason: str | None


def measure_publisher_drift_ms(
    frame_indices: Sequence[int], timestamps_s: Sequence[float], fps_source: float
) -> float:
    """drift_ms = (observed elapsed - expected elapsed) * 1000, over the first/last
    sample in the given (already anchor-relative or absolute) sequences.

    Positive means the publisher fell behind its own nominal pace (frames arrived
    later than the fps would predict); negative means it ran ahead.
    """
    if len(frame_indices) != len(timestamps_s):
        raise ValueError("frame_indices and timestamps_s must be the same length")
    if len(frame_indices) < 2:
        raise ValueError("need at least two samples to measure drift")
    if fps_source <= 0:
        raise ValueError("fps_source must be positive")

    expected_elapsed_s = (frame_indices[-1] - frame_indices[0]) / fps_source
    observed_elapsed_s = timestamps_s[-1] - timestamps_s[0]
    return (observed_elapsed_s - expected_elapsed_s) * 1000.0


def evaluate_publisher_drift(
    frame_indices: Sequence[int],
    timestamps_s: Sequence[float],
    fps_source: float,
    reject_threshold_ms: float,
) -> DriftResult:
    """Measure drift and decide whether the data point must be rejected."""
    drift_ms = measure_publisher_drift_ms(frame_indices, timestamps_s, fps_source)
    if abs(drift_ms) > reject_threshold_ms:
        reason = (
            f"publisher drift {drift_ms:.1f}ms exceeds reject threshold "
            f"{reject_threshold_ms:.1f}ms"
        )
        return DriftResult(drift_ms=drift_ms, rejected=True, reason=reason)
    return DriftResult(drift_ms=drift_ms, rejected=False, reason=None)
