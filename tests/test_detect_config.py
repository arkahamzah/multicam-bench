from __future__ import annotations

from pathlib import Path

import pytest

from multicam_bench.pipeline.detect_config import load_detect_config

REPO_ROOT = Path(__file__).parents[1]

VALID_TEXT = """\
enabled: false
detector: PekingU/rtdetr_v2_r18vd
detect_fps_values: [1, 2, 5, 10, 25]
counting_line:
  a: [0.0, 0.5]
  b: [1.0, 0.5]
max_match_distance_px: 80
device: cpu
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "detect.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_detect_config(tmp_path: Path) -> None:
    config = load_detect_config(_write(tmp_path, VALID_TEXT))

    assert config.enabled is False
    assert config.detector_name == "PekingU/rtdetr_v2_r18vd"
    assert config.detect_fps_values == [1.0, 2.0, 5.0, 10.0, 25.0]
    assert config.counting_line.a == (0.0, 0.5)
    assert config.counting_line.b == (1.0, 0.5)
    assert config.max_match_distance_px == 80.0
    assert config.device == "cpu"


def test_load_detect_config_rejects_empty_detect_fps_values(tmp_path: Path) -> None:
    text = VALID_TEXT.replace(
        "detect_fps_values: [1, 2, 5, 10, 25]", "detect_fps_values: []"
    )
    with pytest.raises(ValueError, match="detect_fps_values"):
        load_detect_config(_write(tmp_path, text))


def test_load_detect_config_rejects_malformed_counting_line_point(tmp_path: Path) -> None:
    text = VALID_TEXT.replace("a: [0.0, 0.5]", "a: [0.0]")
    with pytest.raises(ValueError, match=r"\[x, y\]"):
        load_detect_config(_write(tmp_path, text))


def test_repo_detect_config_is_valid_and_disabled_by_default() -> None:
    config = load_detect_config(REPO_ROOT / "configs" / "detect.yaml")
    assert config.enabled is False
    assert config.detect_fps_values == [1.0, 2.0, 5.0, 10.0, 25.0]
