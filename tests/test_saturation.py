from __future__ import annotations

import pytest

from multicam_bench.bench.saturation import (
    SaturationResult,
    compute_n_max,
    detect_saturation,
    is_window_saturated,
    percentile_ms,
)
from multicam_bench.config import Thresholds

THRESHOLDS = Thresholds(
    lag_p95_ms=200.0,
    drop_rate_pct=1.0,
    publisher_drift_reject_ms=50.0,
    warmup_s=20.0,
    measure_s=90.0,
)


def test_percentile_ms_basic() -> None:
    assert percentile_ms([100.0, 200.0, 300.0], 50) == pytest.approx(200.0)


def test_percentile_ms_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile_ms([], 95)


def test_is_window_saturated_by_lag() -> None:
    assert is_window_saturated(lag_p95_ms=201.0, drop_rate_pct=0.0, thresholds=THRESHOLDS)


def test_is_window_saturated_by_drop_rate() -> None:
    assert is_window_saturated(lag_p95_ms=0.0, drop_rate_pct=1.1, thresholds=THRESHOLDS)


def test_is_window_saturated_at_exact_threshold_is_not_saturated() -> None:
    # "exceeds" is strict per THREATS-TO-VALIDITY.md §1.1 — equal to threshold is fine.
    assert not is_window_saturated(lag_p95_ms=200.0, drop_rate_pct=1.0, thresholds=THRESHOLDS)


def test_is_window_saturated_below_both_thresholds() -> None:
    assert not is_window_saturated(lag_p95_ms=50.0, drop_rate_pct=0.1, thresholds=THRESHOLDS)


def test_detect_saturation_healthy_run_not_saturated() -> None:
    # 60 samples across 60s, flat 10ms lag, no drops.
    lag_samples_ms = [(float(t), 10.0) for t in range(60)]
    result = detect_saturation(
        lag_samples_ms, drop_events_t_s=[], expected_fps=1.0, thresholds=THRESHOLDS
    )
    assert result == SaturationResult(
        saturated=False, first_saturated_window_start_s=None, reason=None
    )


def test_detect_saturation_flags_lag_spike_window() -> None:
    # First 30s window healthy, second 30s window has lag spikes over 200ms.
    healthy = [(float(t), 10.0) for t in range(30)]
    spiking = [(float(t), 250.0) for t in range(30, 60)]
    result = detect_saturation(
        healthy + spiking, drop_events_t_s=[], expected_fps=1.0, thresholds=THRESHOLDS
    )
    assert result.saturated is True
    assert result.first_saturated_window_start_s == pytest.approx(30.0)
    assert result.reason is not None


def test_detect_saturation_flags_high_drop_rate_window() -> None:
    lag_samples_ms = [(float(t), 10.0) for t in range(30)]
    # expected_fps=1 over a 30s window => 30 expected frames; 1 drop is > 1% already
    # at drop_rate_pct=1.0, so use enough drops to be unambiguous.
    drop_events = [5.0, 10.0, 15.0]
    result = detect_saturation(
        lag_samples_ms, drop_events_t_s=drop_events, expected_fps=1.0, thresholds=THRESHOLDS
    )
    assert result.saturated is True


def test_detect_saturation_reports_first_saturated_window_not_last() -> None:
    spiking_first = [(float(t), 250.0) for t in range(0, 30)]
    healthy_second = [(float(t), 10.0) for t in range(30, 60)]
    result = detect_saturation(
        spiking_first + healthy_second,
        drop_events_t_s=[],
        expected_fps=1.0,
        thresholds=THRESHOLDS,
    )
    assert result.saturated is True
    assert result.first_saturated_window_start_s == pytest.approx(0.0)


def test_detect_saturation_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        detect_saturation([], [], expected_fps=30.0, thresholds=THRESHOLDS)


def test_detect_saturation_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_s"):
        detect_saturation(
            [(0.0, 10.0)], [], expected_fps=30.0, thresholds=THRESHOLDS, window_s=0
        )


def test_compute_n_max_picks_largest_unsaturated() -> None:
    assert compute_n_max({1: False, 2: False, 4: False, 8: True, 16: True}) == 4


def test_compute_n_max_all_saturated_returns_none() -> None:
    assert compute_n_max({1: True, 2: True}) is None


def test_compute_n_max_non_monotonic_still_picks_largest_false() -> None:
    # A noisy result where a smaller N saturated but a larger one didn't is possible;
    # N_max is defined as the largest unsaturated N, not the largest contiguous run.
    assert compute_n_max({1: False, 2: True, 4: False}) == 4


def test_compute_n_max_empty_input() -> None:
    assert compute_n_max({}) is None
