from __future__ import annotations

import pytest

from multicam_bench.bench.aggregate import MedianIQR, exclude_near_loop_boundary, median_iqr


def test_median_iqr_basic() -> None:
    result = median_iqr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert result.median == pytest.approx(4.5)
    assert result.q1 == pytest.approx(2.75)
    assert result.q3 == pytest.approx(6.25)


def test_median_iqr_property() -> None:
    result = MedianIQR(median=5.0, q1=2.0, q3=8.0)
    assert result.iqr == pytest.approx(6.0)


def test_median_iqr_single_value_has_zero_iqr() -> None:
    result = median_iqr([42.0])
    assert result.median == pytest.approx(42.0)
    assert result.iqr == pytest.approx(0.0)


def test_median_iqr_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        median_iqr([])


def test_median_iqr_not_swayed_by_a_single_outlier() -> None:
    # CLAUDE.md hard rule 4: a mean would move a lot here; the median should not.
    baseline = median_iqr([10.0, 10.0, 10.0, 10.0, 10.0]).median
    with_outlier = median_iqr([10.0, 10.0, 10.0, 10.0, 10000.0]).median
    assert with_outlier == pytest.approx(baseline)


def test_exclude_near_loop_boundary_keeps_interior_frames() -> None:
    # period=100, margin=5: frames 5..94 are interior, 0..4 and 95..99 are boundary.
    frame_indices = list(range(10, 20))  # well inside the interior
    kept, excluded = exclude_near_loop_boundary(frame_indices, period_frames=100, margin=5)
    assert kept == frame_indices
    assert excluded == 0


def test_exclude_near_loop_boundary_drops_frames_at_start() -> None:
    frame_indices = [0, 1, 2, 3, 4, 5, 6]
    kept, excluded = exclude_near_loop_boundary(frame_indices, period_frames=100, margin=5)
    assert kept == [5, 6]
    assert excluded == 5


def test_exclude_near_loop_boundary_drops_frames_at_end() -> None:
    frame_indices = [93, 94, 95, 96, 97, 98, 99]
    kept, excluded = exclude_near_loop_boundary(frame_indices, period_frames=100, margin=5)
    assert kept == [93, 94]
    assert excluded == 5


def test_exclude_near_loop_boundary_handles_multiple_loops() -> None:
    # Boundary at every multiple of period, not just the first.
    frame_indices = [98, 99, 100, 101, 198, 199, 200, 201]
    kept, excluded = exclude_near_loop_boundary(frame_indices, period_frames=100, margin=5)
    assert kept == []
    assert excluded == 8


def test_exclude_near_loop_boundary_zero_margin_excludes_nothing() -> None:
    frame_indices = [0, 1, 50, 99]
    kept, excluded = exclude_near_loop_boundary(frame_indices, period_frames=100, margin=0)
    assert kept == frame_indices
    assert excluded == 0


def test_exclude_near_loop_boundary_rejects_non_positive_period() -> None:
    with pytest.raises(ValueError, match="period_frames"):
        exclude_near_loop_boundary([1, 2, 3], period_frames=0)


def test_exclude_near_loop_boundary_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        exclude_near_loop_boundary([1, 2, 3], period_frames=100, margin=-1)
