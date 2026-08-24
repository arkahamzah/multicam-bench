"""Loads `configs/detect.yaml`: whether the v0.5 detection stage is enabled, which
detector to use, the detect-fps axis, counting line, and association distance.
Default is disabled — the stage never runs unless a config explicitly turns it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multicam_bench.pipeline.counting import CountingLine, Point
from multicam_bench.yaml_io import load_yaml_mapping, require


@dataclass(frozen=True)
class DetectConfig:
    enabled: bool
    detector_name: str
    detect_fps_values: list[float]
    counting_line: CountingLine
    max_match_distance_px: float
    device: str


def _point(values: Any) -> Point:
    if len(values) != 2:
        raise ValueError("counting line points must be [x, y]")
    return (float(values[0]), float(values[1]))


def load_detect_config(path: Path) -> DetectConfig:
    data = load_yaml_mapping(path)
    context = str(path)

    line_data = require(data, "counting_line", context)
    line = CountingLine(
        a=_point(require(line_data, "a", f"{context}: counting_line")),
        b=_point(require(line_data, "b", f"{context}: counting_line")),
    )

    detect_fps_values = [float(v) for v in require(data, "detect_fps_values", context)]
    if not detect_fps_values:
        raise ValueError("detect_fps_values must not be empty")

    return DetectConfig(
        enabled=bool(require(data, "enabled", context)),
        detector_name=str(require(data, "detector", context)),
        detect_fps_values=detect_fps_values,
        counting_line=line,
        max_match_distance_px=float(require(data, "max_match_distance_px", context)),
        device=str(require(data, "device", context)),
    )
