"""Loads `configs/detect.yaml`: whether the v0.5 detection stage is enabled, which
detector to use, the detect-fps axis, counting line, and association distance.
Default is disabled — the stage never runs unless a config explicitly turns it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from multicam_bench.pipeline.counting import CountingLine, Point


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
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    line_data = data["counting_line"]
    line = CountingLine(a=_point(line_data["a"]), b=_point(line_data["b"]))

    detect_fps_values = [float(v) for v in data["detect_fps_values"]]
    if not detect_fps_values:
        raise ValueError("detect_fps_values must not be empty")

    return DetectConfig(
        enabled=bool(data["enabled"]),
        detector_name=str(data["detector"]),
        detect_fps_values=detect_fps_values,
        counting_line=line,
        max_match_distance_px=float(data["max_match_distance_px"]),
        device=str(data["device"]),
    )
