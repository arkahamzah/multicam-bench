from __future__ import annotations

import random
from pathlib import Path

import pytest

from multicam_bench.bench.sweep_plan import (
    SweepConfigPoint,
    SweepPoint,
    build_sweep_plan,
    camera_dir,
    cross_product_configs,
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


def test_randomize_order_works_over_arbitrary_config_type() -> None:
    configs = cross_product_configs([1, 2], ["libx264", "libx265"], ["ffmpeg-cpu"])
    orders = randomize_order(configs, repetitions=2, rng=random.Random(1))
    assert len(orders) == 2
    for order in orders:
        assert set(order) == set(configs)


def test_cross_product_configs_covers_every_combination() -> None:
    configs = cross_product_configs([1, 2], ["libx264", "libx265"], ["ffmpeg-cpu", "qsv"])
    assert len(configs) == 2 * 2 * 2
    assert SweepConfigPoint(1, "libx264", "qsv") in configs
    assert SweepConfigPoint(2, "libx265", "ffmpeg-cpu") in configs


def test_build_sweep_plan_covers_every_config_baseline_and_measured_per_repetition() -> None:
    configs = cross_product_configs([1, 2, 4], ["libx264"], ["ffmpeg-cpu"])
    repetitions = 3
    plan = build_sweep_plan(configs, repetitions, rng=random.Random(1))

    assert len(plan) == len(configs) * repetitions * 2  # baseline + measured

    for rep in range(repetitions):
        rep_points = [p for p in plan if p.repetition == rep]
        rep_configs = {(p.n_streams, p.codec, p.backend) for p in rep_points}
        assert rep_configs == {(c.n_streams, c.codec, c.backend) for c in configs}
        for cfg in configs:
            modes = {
                p.publisher_only
                for p in rep_points
                if (p.n_streams, p.codec, p.backend) == (cfg.n_streams, cfg.codec, cfg.backend)
            }
            assert modes == {True, False}


def test_build_sweep_plan_pairs_baseline_immediately_before_measured() -> None:
    configs = cross_product_configs([1, 2], ["libx264"], ["ffmpeg-cpu"])
    plan = build_sweep_plan(configs, repetitions=1, rng=random.Random(3))
    for i in range(0, len(plan), 2):
        baseline, measured = plan[i], plan[i + 1]
        assert baseline.publisher_only is True
        assert measured.publisher_only is False
        assert baseline.n_streams == measured.n_streams
        assert baseline.codec == measured.codec
        assert baseline.backend == measured.backend
        assert baseline.repetition == measured.repetition


def test_build_sweep_plan_randomises_full_cross_product_order_per_repetition() -> None:
    # T2: order must vary across the WHOLE (N, codec, backend) space, not just N.
    configs = cross_product_configs([1, 2, 4], ["libx264", "libx265"], ["ffmpeg-cpu", "qsv"])
    plan = build_sweep_plan(configs, repetitions=3, rng=random.Random(9))
    order_sequences = set()
    for rep in range(3):
        seq = tuple(
            (p.n_streams, p.codec, p.backend)
            for p in plan
            if p.repetition == rep and p.publisher_only
        )
        order_sequences.add(seq)
    assert len(order_sequences) > 1


def test_sweep_point_dir_name_encodes_n_codec_backend_repetition_and_mode() -> None:
    baseline = SweepPoint(
        n_streams=8, repetition=1, order_index=0, publisher_only=True,
        codec="libx264", backend="qsv",
    )
    measured = SweepPoint(
        n_streams=8, repetition=1, order_index=0, publisher_only=False,
        codec="libx264", backend="qsv",
    )
    assert baseline.dir_name() == "N8_libx264_qsv_rep1_baseline"
    assert measured.dir_name() == "N8_libx264_qsv_rep1_measured"


def test_point_dir_and_camera_dir_layout() -> None:
    sweep_dir = Path("runs/20260824-000000")
    point = SweepPoint(
        n_streams=4, repetition=0, order_index=2, publisher_only=False,
        codec="libx265", backend="d3d11va",
    )
    p_dir = point_dir(sweep_dir, point)
    assert p_dir == sweep_dir / "points" / "N4_libx265_d3d11va_rep0_measured"
    assert camera_dir(p_dir, 2) == p_dir / "cam2"
