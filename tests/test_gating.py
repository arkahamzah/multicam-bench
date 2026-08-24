from __future__ import annotations

import pytest

from multicam_bench.pipeline.gating import detect_frame_indices


def test_equal_fps_selects_every_frame() -> None:
    indices = detect_frame_indices(decode_fps=30.0, detect_fps=30.0, total_frames=5)
    assert indices == [0, 1, 2, 3, 4]


def test_half_rate_selects_every_other_frame() -> None:
    indices = detect_frame_indices(decode_fps=30.0, detect_fps=15.0, total_frames=6)
    assert indices == [0, 2, 4]


def test_low_detect_fps_sparsely_samples() -> None:
    # 30fps decode, 5fps detect -> every 6th frame.
    indices = detect_frame_indices(decode_fps=30.0, detect_fps=5.0, total_frames=30)
    assert indices == [0, 6, 12, 18, 24]


def test_always_includes_frame_zero() -> None:
    indices = detect_frame_indices(decode_fps=30.0, detect_fps=1.0, total_frames=1)
    assert indices == [0]


def test_zero_total_frames_returns_empty() -> None:
    assert detect_frame_indices(decode_fps=30.0, detect_fps=5.0, total_frames=0) == []


def test_detect_fps_cannot_exceed_decode_fps() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        detect_frame_indices(decode_fps=5.0, detect_fps=30.0, total_frames=10)


def test_rejects_non_positive_decode_fps() -> None:
    with pytest.raises(ValueError, match="decode_fps"):
        detect_frame_indices(decode_fps=0.0, detect_fps=1.0, total_frames=10)


def test_rejects_non_positive_detect_fps() -> None:
    with pytest.raises(ValueError, match="detect_fps"):
        detect_frame_indices(decode_fps=30.0, detect_fps=0.0, total_frames=10)


def test_rejects_negative_total_frames() -> None:
    with pytest.raises(ValueError, match="total_frames"):
        detect_frame_indices(decode_fps=30.0, detect_fps=5.0, total_frames=-1)


def test_sampled_rate_approaches_detect_fps_over_a_long_window() -> None:
    total_frames = 300  # 10s at 30fps
    indices = detect_frame_indices(decode_fps=30.0, detect_fps=10.0, total_frames=total_frames)
    achieved_fps = len(indices) / (total_frames / 30.0)
    assert achieved_fps == pytest.approx(10.0, rel=0.05)
