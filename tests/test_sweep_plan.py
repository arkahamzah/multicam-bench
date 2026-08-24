from __future__ import annotations

import random
from pathlib import Path

import pytest

from multicam_bench.bench.sweep_plan import (
    SweepPoint,
    build_sweep_plan,
    camera_dir,
    point_dir,
    randomize_order,
)


def test_randomize_order_returns_one_permutation_per_repetition() -> None:
    orders = randomize_order([1, 2, 4, 8], repetitions=3, rng=random.Random(0))
    assert len(orders) == 3
    for order in orders:
        assert sorted(order) == [1, 2, 4, 8]


def test_randomize_order_is_deterministic_for_a_given_seed() -> None:
    orders_a = randomize_order([1, 2, 4, 8, 12, 16], repetitions=3, rng=random.Random(42))
    orders_b = randomize_order([1, 2, 4, 8, 12, 16], repetitions=3, rng=random.Random(42))
    assert orders_a == orders_b


def test_randomize_order_differs_across_repetitions_with_high_probability() -> None:
    # THREATS-TO-VALIDITY.md T2: order must be randomised per repetition, not fixed.
    orders = randomize_order([1, 2, 4, 8, 12, 16], repetitions=3, rng=random.Random(7))
    assert len({tuple(o) for o in orders}) > 1


def test_randomize_order_rejects_fewer_than_one_repetition() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        randomize_order([1, 2], repetitions=0, rng=random.Random(0))


def test_randomize_order_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        randomize_order([], repetitions=3, rng=random.Random(0))


def test_build_sweep_plan_covers_every_n_baseline_and_measured_per_repetition() -> None:
    n_values = [1, 2, 4]
    repetitions = 3
    plan = build_sweep_plan(n_values, repetitions, rng=random.Random(1))

    assert len(plan) == len(n_values) * repetitions * 2  # baseline + measured

    for rep in range(repetitions):
        rep_points = [p for p in plan if p.repetition == rep]
        rep_ns = {p.n_streams for p in rep_points}
        assert rep_ns == set(n_values)
        for n in n_values:
            modes = {p.publisher_only for p in rep_points if p.n_streams == n}
            assert modes == {True, False}


def test_build_sweep_plan_pairs_baseline_immediately_before_measured() -> None:
    plan = build_sweep_plan([1, 2], repetitions=1, rng=random.Random(3))
    for i in range(0, len(plan), 2):
        baseline, measured = plan[i], plan[i + 1]
        assert baseline.publisher_only is True
        assert measured.publisher_only is False
        assert baseline.n_streams == measured.n_streams
        assert baseline.repetition == measured.repetition


def test_sweep_point_dir_name_encodes_n_repetition_and_mode() -> None:
    baseline = SweepPoint(n_streams=8, repetition=1, order_index=0, publisher_only=True)
    measured = SweepPoint(n_streams=8, repetition=1, order_index=0, publisher_only=False)
    assert baseline.dir_name() == "N8_rep1_baseline"
    assert measured.dir_name() == "N8_rep1_measured"


def test_point_dir_and_camera_dir_layout() -> None:
    sweep_dir = Path("runs/20260824-000000")
    point = SweepPoint(n_streams=4, repetition=0, order_index=2, publisher_only=False)
    p_dir = point_dir(sweep_dir, point)
    assert p_dir == sweep_dir / "points" / "N4_rep0_measured"
    assert camera_dir(p_dir, 2) == p_dir / "cam2"
