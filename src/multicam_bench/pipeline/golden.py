"""Golden-configuration selection (PRIOR-ART.md §4.1, following VideoStorm and
Chameleon's method): the most expensive configuration in a sweep — max resolution,
max detect fps, strongest detector — serves as pseudo-ground-truth. No hand
labelling required; every cheaper configuration is scored by its retention against
this one (see `pipeline/counting.py::count_retention`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    resolution_name: str
    pixel_rate: float  # width * height * decode_fps
    detect_fps: float
    detector_name: str
    detector_strength: int  # higher = stronger/more expensive; caller-assigned


def select_golden_config(configs: Sequence[DetectionConfig]) -> DetectionConfig:
    """The single most expensive configuration: max pixel_rate, then max
    detect_fps, then max detector_strength as tie-breakers.
    """
    if not configs:
        raise ValueError("configs must not be empty")
    return max(configs, key=lambda c: (c.pixel_rate, c.detect_fps, c.detector_strength))
