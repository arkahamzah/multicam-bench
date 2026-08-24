"""Loads `configs/sweep.yaml`: the N sweep, repetition count, cooldown, and default
test content (resolution/codec/fps/duration) for the v0.2 sweep orchestrator.
Mirrors `config.py`'s contract for thresholds.yaml — read only, never written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# THREATS-TO-VALIDITY.md T3: minimum repetitions for a variance estimate.
MIN_REPETITIONS = 3


@dataclass(frozen=True)
class ContentSpec:
    resolution: str
    codec: str
    fps: int
    duration_s: float


@dataclass(frozen=True)
class SweepConfig:
    n_values: list[int]
    repetitions: int
    cooldown_s: float
    content: ContentSpec
    rtsp_base_url: str
    mediamtx_config: Path


def load_sweep_config(path: Path) -> SweepConfig:
    """Parse a sweep YAML file into a `SweepConfig` instance."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    repetitions = int(data["repetitions"])
    if repetitions < MIN_REPETITIONS:
        raise ValueError(
            f"repetitions must be >= {MIN_REPETITIONS} (THREATS-TO-VALIDITY.md T3), "
            f"got {repetitions}"
        )

    content_data = data["content"]
    content = ContentSpec(
        resolution=str(content_data["resolution"]),
        codec=str(content_data["codec"]),
        fps=int(content_data["fps"]),
        duration_s=float(content_data["duration_s"]),
    )

    return SweepConfig(
        n_values=[int(n) for n in data["n_values"]],
        repetitions=repetitions,
        cooldown_s=float(data["cooldown_s"]),
        content=content,
        rtsp_base_url=str(data["rtsp_base_url"]),
        mediamtx_config=Path(data["mediamtx_config"]),
    )
