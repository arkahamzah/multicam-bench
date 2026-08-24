from __future__ import annotations

import random

import pytest

from multicam_bench.bench.decode_only_sweep import (
    DecodeOnlyPoint,
    build_decode_only_plan,
    cv2_backend_to_ffmpeg_hwaccel,
)


def test_cv2_backend_to_ffmpeg_hwaccel_mapping() -> None:
    assert cv2_backend_to_ffmpeg_hwaccel("ffmpeg-cpu") == "none"
    assert cv2_backend_to_ffmpeg_hwaccel("d3d11va") == "d3d11va"
    assert cv2_backend_to_ffmpeg_hwaccel("qsv") == "qsv"
    assert cv2_backend_to_ffmpeg_hwaccel("cuda") == "cuda"
    assert cv2_backend_to_ffmpeg_hwaccel("vaapi") == "vaapi"


def test_cv2_backend_to_ffmpeg_hwaccel_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        cv2_backend_to_ffmpeg_hwaccel("made-up")


def test_build_decode_only_plan_covers_full_cross_product_per_repetition() -> None:
    plan = build_decode_only_plan(
        n_values=[1, 2],
        codecs=["libx264", "libx265"],
        hwaccels=["none", "cuda"],
        repetitions=3,
        rng=random.Random(1),
    )
    assert len(plan) == 2 * 2 * 2 * 3

    for rep in range(3):
        rep_points = {(p.n_streams, p.codec, p.hwaccel) for p in plan if p.repetition == rep}
        assert rep_points == {
            (n, c, h) for n in [1, 2] for c in ["libx264", "libx265"] for h in ["none", "cuda"]
        }


def test_build_decode_only_plan_has_no_baseline_publisher_concept() -> None:
    # Unlike the full-pipeline sweep, every point here IS a measured point —
    # there's no publisher to baseline, so no `publisher_only` field exists.
    plan = build_decode_only_plan(
        n_values=[1], codecs=["libx264"], hwaccels=["none"], repetitions=1,
        rng=random.Random(0),
    )
    assert len(plan) == 1
    assert not hasattr(plan[0], "publisher_only")


def test_build_decode_only_plan_randomises_order_per_repetition() -> None:
    plan = build_decode_only_plan(
        n_values=[1, 2, 4],
        codecs=["libx264", "libx265"],
        hwaccels=["none", "cuda", "qsv"],
        repetitions=4,
        rng=random.Random(5),
    )
    order_sequences = set()
    for rep in range(4):
        seq = tuple((p.n_streams, p.codec, p.hwaccel) for p in plan if p.repetition == rep)
        order_sequences.add(seq)
    assert len(order_sequences) > 1


def test_decode_only_point_dir_name_encodes_full_config() -> None:
    point = DecodeOnlyPoint(n_streams=8, codec="libx265", hwaccel="qsv", repetition=2)
    assert point.dir_name() == "N8_libx265_qsv_rep2"
