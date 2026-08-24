"""Loads `configs/sweep.yaml`: the N sweep, codec axis, backend axis, repetition
count, cooldown, and test content for the v0.2/v0.4 sweep orchestrator. Mirrors
`config.py`'s contract for thresholds.yaml — read only, never written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multicam_bench.yaml_io import load_yaml_mapping, require

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
    data = load_yaml_mapping(path)
    context = str(path)

    repetitions = int(require(data, "repetitions", context))
    if repetitions < MIN_REPETITIONS:
        raise ValueError(
            f"repetitions must be >= {MIN_REPETITIONS} (THREATS-TO-VALIDITY.md T3), "
            f"got {repetitions}"
        )

    n_values = [int(n) for n in require(data, "n_values", context)]
    codecs = [str(c) for c in require(data, "codecs", context)]
    backends = [str(b) for b in require(data, "backends", context)]
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
        cooldown_s=float(require(data, "cooldown_s", context)),
        resolution=str(require(data, "resolution", context)),
        fps=int(require(data, "fps", context)),
        duration_s=float(require(data, "duration_s", context)),
        rtsp_base_url=str(require(data, "rtsp_base_url", context)),
        mediamtx_config=Path(require(data, "mediamtx_config", context)),
    )
