from __future__ import annotations

import pytest

from multicam_bench.pipeline.golden import DetectionConfig, select_golden_config


def _config(pixel_rate: float, detect_fps: float, strength: int = 1) -> DetectionConfig:
    return DetectionConfig(
        resolution_name="360p",
        pixel_rate=pixel_rate,
        detect_fps=detect_fps,
        detector_name="rtdetr_v2",
        detector_strength=strength,
    )


def test_select_golden_picks_highest_pixel_rate() -> None:
    low = _config(pixel_rate=100.0, detect_fps=5.0)
    high = _config(pixel_rate=1000.0, detect_fps=5.0)
    assert select_golden_config([low, high]) is high


def test_select_golden_uses_detect_fps_as_tiebreaker() -> None:
    low_fps = _config(pixel_rate=1000.0, detect_fps=5.0)
    high_fps = _config(pixel_rate=1000.0, detect_fps=25.0)
    assert select_golden_config([low_fps, high_fps]) is high_fps


def test_select_golden_uses_detector_strength_as_final_tiebreaker() -> None:
    weak = _config(pixel_rate=1000.0, detect_fps=25.0, strength=1)
    strong = _config(pixel_rate=1000.0, detect_fps=25.0, strength=2)
    assert select_golden_config([weak, strong]) is strong


def test_select_golden_pixel_rate_dominates_detect_fps() -> None:
    # Higher pixel_rate wins even if its detect_fps is lower — pixel_rate is
    # compared first, per PRIOR-ART.md §4.1 ("max resolution" before "max fps").
    high_pixel_low_fps = _config(pixel_rate=1000.0, detect_fps=1.0)
    low_pixel_high_fps = _config(pixel_rate=100.0, detect_fps=25.0)
    assert select_golden_config([high_pixel_low_fps, low_pixel_high_fps]) is high_pixel_low_fps


def test_select_golden_single_config_is_itself() -> None:
    only = _config(pixel_rate=500.0, detect_fps=10.0)
    assert select_golden_config([only]) is only


def test_select_golden_rejects_empty_configs() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_golden_config([])
