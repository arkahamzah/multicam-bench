"""Loads `configs/sweep.yaml`: the N sweep, codec axis, backend axis, repetition
count, cooldown, and test content for the v0.2/v0.4 sweep orchestrator. Mirrors
`config.py`'s contract for thresholds.yaml — read only, never written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# THREATS-TO-VALIDITY.md T3: minimum repetitions for a variance estimate.
MIN_REPETITIONS = 3


@dataclass(frozen=True)
class SweepConfig:
    n_values: list[int]
    codecs: list[str]
    backends: list[str]
    repetitions: int
    cooldown_s: float
    resolution: str
    fps: int
    duration_s: float
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

    n_values = [int(n) for n in data["n_values"]]
    codecs = [str(c) for c in data["codecs"]]
    backends = [str(b) for b in data["backends"]]
    if not n_values:
        raise ValueError("n_values must not be empty")
    if not codecs:
        raise ValueError("codecs must not be empty")
    if not backends:
        raise ValueError("backends must not be empty")

    return SweepConfig(
        n_values=n_values,
        codecs=codecs,
        backends=backends,
        repetitions=repetitions,
        cooldown_s=float(data["cooldown_s"]),
        resolution=str(data["resolution"]),
        fps=int(data["fps"]),
        duration_s=float(data["duration_s"]),
        rtsp_base_url=str(data["rtsp_base_url"]),
        mediamtx_config=Path(data["mediamtx_config"]),
    )
