from __future__ import annotations

import pytest

from multicam_bench.bench.drift import evaluate_publisher_drift, measure_publisher_drift_ms


def test_perfect_pacing_has_zero_drift() -> None:
    # 30 frames at 30fps should take exactly 1s of wall clock.
    frame_indices = [0, 30]
    timestamps_s = [100.0, 101.0]
    drift = measure_publisher_drift_ms(frame_indices, timestamps_s, fps_source=30.0)
    assert drift == pytest.approx(0.0)


def test_publisher_running_behind_is_positive_drift() -> None:
    # Expected 1s elapsed for 30 frames at 30fps, but 1.1s actually passed.
    frame_indices = [0, 30]
    timestamps_s = [100.0, 101.1]
    drift = measure_publisher_drift_ms(frame_indices, timestamps_s, fps_source=30.0)
    assert drift == pytest.approx(100.0)


def test_publisher_running_ahead_is_negative_drift() -> None:
    frame_indices = [0, 30]
    timestamps_s = [100.0, 100.9]
    drift = measure_publisher_drift_ms(frame_indices, timestamps_s, fps_source=30.0)
    assert drift == pytest.approx(-100.0)


def test_drift_uses_only_first_and_last_sample() -> None:
    # Jitter in the middle of the window must not affect the measured drift.
    frame_indices = [0, 10, 20, 30]
    timestamps_s = [100.0, 100.5, 100.2, 101.0]
    drift = measure_publisher_drift_ms(frame_indices, timestamps_s, fps_source=30.0)
    assert drift == pytest.approx(0.0)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="length"):
        measure_publisher_drift_ms([0, 1], [0.0], fps_source=30.0)


def test_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="two samples"):
        measure_publisher_drift_ms([0], [0.0], fps_source=30.0)


def test_non_positive_fps_rejected() -> None:
    with pytest.raises(ValueError, match="fps_source"):
        measure_publisher_drift_ms([0, 30], [0.0, 1.0], fps_source=0.0)


def test_evaluate_within_threshold_is_not_rejected() -> None:
    # 30ms drift, 50ms threshold (configs/thresholds.yaml default).
    frame_indices = [0, 30]
    timestamps_s = [100.0, 101.03]
    result = evaluate_publisher_drift(
        frame_indices, timestamps_s, fps_source=30.0, reject_threshold_ms=50.0
    )
    assert result.rejected is False
    assert result.reason is None
    assert result.drift_ms == pytest.approx(30.0)


def test_evaluate_over_threshold_is_rejected_with_reason() -> None:
    frame_indices = [0, 30]
    timestamps_s = [100.0, 101.2]
    result = evaluate_publisher_drift(
        frame_indices, timestamps_s, fps_source=30.0, reject_threshold_ms=50.0
    )
    assert result.rejected is True
    assert result.reason is not None
    assert "200.0ms" in result.reason
    assert "50.0ms" in result.reason


def test_evaluate_negative_drift_beyond_threshold_is_also_rejected() -> None:
    # Running far ahead of pace is just as much a pacing failure as running behind.
    frame_indices = [0, 30]
    timestamps_s = [100.0, 100.8]
    result = evaluate_publisher_drift(
        frame_indices, timestamps_s, fps_source=30.0, reject_threshold_ms=50.0
    )
    assert result.rejected is True
    assert result.drift_ms == pytest.approx(-200.0)
