from __future__ import annotations

import pytest

from multicam_bench.pipeline.counting import (
    CountingLine,
    Detection,
    associate_nearest,
    count_crossings,
    count_retention,
    intersecting_frame_indices,
    side_of_line,
)

# Horizontal line across the middle of a normalised [0,1] frame.
LINE = CountingLine(a=(0.0, 0.5), b=(1.0, 0.5))


def _car(cx: float, cy: float) -> Detection:
    return Detection(cx=cx, cy=cy, class_name="car", score=0.9)


def test_side_of_line_above_and_below() -> None:
    above = side_of_line((0.5, 0.7), LINE)
    below = side_of_line((0.5, 0.3), LINE)
    assert above > 0
    assert below < 0
    assert (above > 0) != (below > 0)


def test_side_of_line_on_the_line_is_zero() -> None:
    assert side_of_line((0.5, 0.5), LINE) == pytest.approx(0.0)


def test_associate_nearest_matches_closest_pairs_greedily() -> None:
    prev = [_car(0.0, 0.0), _car(5.0, 5.0)]
    curr = [_car(0.1, 0.1), _car(5.1, 5.1)]
    pairs = associate_nearest(prev, curr, max_distance=1.0)
    assert set(pairs) == {(0, 0), (1, 1)}


def test_associate_nearest_excludes_different_classes() -> None:
    prev = [Detection(cx=0.0, cy=0.0, class_name="car", score=1.0)]
    curr = [Detection(cx=0.0, cy=0.0, class_name="truck", score=1.0)]
    assert associate_nearest(prev, curr, max_distance=1.0) == []


def test_associate_nearest_excludes_beyond_max_distance() -> None:
    prev = [_car(0.0, 0.0)]
    curr = [_car(10.0, 10.0)]
    assert associate_nearest(prev, curr, max_distance=1.0) == []


def test_associate_nearest_rejects_negative_max_distance() -> None:
    with pytest.raises(ValueError, match="max_distance"):
        associate_nearest([], [], max_distance=-1.0)


def test_associate_nearest_each_detection_used_at_most_once() -> None:
    # Two prev detections both close to one curr detection: only the nearer wins.
    prev = [_car(0.0, 0.0), _car(0.2, 0.0)]
    curr = [_car(0.05, 0.0)]
    pairs = associate_nearest(prev, curr, max_distance=1.0)
    assert pairs == [(0, 0)]


def test_count_crossings_single_object_crossing_the_line() -> None:
    frames = [[_car(0.5, 0.3)], [_car(0.5, 0.7)]]
    assert count_crossings(frames, LINE, max_match_distance=1.0) == 1


def test_count_crossings_object_staying_on_one_side_does_not_count() -> None:
    frames = [[_car(0.5, 0.3)], [_car(0.5, 0.35)], [_car(0.5, 0.4)]]
    assert count_crossings(frames, LINE, max_match_distance=1.0) == 0


def test_count_crossings_multiple_objects() -> None:
    frames = [
        [_car(0.2, 0.3), _car(0.8, 0.7)],
        [_car(0.2, 0.7), _car(0.8, 0.3)],  # both cross, in opposite directions
    ]
    assert count_crossings(frames, LINE, max_match_distance=1.0) == 2


def test_count_crossings_unassociated_detection_does_not_count() -> None:
    # No association possible (too far) — must not be mistaken for a crossing.
    frames = [[_car(0.1, 0.3)], [_car(0.9, 0.7)]]
    assert count_crossings(frames, LINE, max_match_distance=0.1) == 0


def test_count_crossings_empty_frames_is_zero() -> None:
    assert count_crossings([], LINE, max_match_distance=1.0) == 0


def test_count_crossings_single_frame_is_zero() -> None:
    assert count_crossings([[_car(0.5, 0.5)]], LINE, max_match_distance=1.0) == 0


def test_count_retention_exact_match_is_one() -> None:
    assert count_retention(golden_count=10, candidate_count=10) == pytest.approx(1.0)


def test_count_retention_partial_undercount() -> None:
    assert count_retention(golden_count=10, candidate_count=8) == pytest.approx(0.8)


def test_count_retention_overcount_symmetric() -> None:
    assert count_retention(golden_count=10, candidate_count=15) == pytest.approx(0.5)


def test_count_retention_clamped_at_zero_for_large_error() -> None:
    assert count_retention(golden_count=10, candidate_count=100) == pytest.approx(0.0)


def test_count_retention_both_zero_is_perfect_retention() -> None:
    assert count_retention(golden_count=0, candidate_count=0) == pytest.approx(1.0)


def test_count_retention_golden_zero_candidate_nonzero_is_zero() -> None:
    assert count_retention(golden_count=0, candidate_count=5) == pytest.approx(0.0)


def test_count_retention_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="negative"):
        count_retention(golden_count=-1, candidate_count=5)


def test_intersecting_frame_indices() -> None:
    assert intersecting_frame_indices([1, 2, 3], [2, 3, 4]) == {2, 3}


def test_intersecting_frame_indices_disjoint_is_empty() -> None:
    assert intersecting_frame_indices([1, 2], [3, 4]) == set()
