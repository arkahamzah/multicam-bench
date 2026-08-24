"""Loads `configs/thresholds.yaml` — the pre-registered saturation and protocol
thresholds from THREATS-TO-VALIDITY.md. Values are read, never hardcoded, and this
module never writes to that file.

Every required key is validated explicitly via `yaml_io.require` and raises a
clear `ValueError` naming the exact missing key if the file is malformed. This
must never silently fall back to a default: a silently-wrong threshold would
invalidate every saturation/rejection decision downstream with no visible
symptom. See `yaml_io.py` for why the file is read with `utf-8-sig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multicam_bench.yaml_io import load_yaml_mapping, require


@dataclass(frozen=True)
class Thresholds:
    lag_p95_ms: float
    drop_rate_pct: float
    publisher_drift_reject_ms: float
    warmup_s: float
    measure_s: float


def load_thresholds(path: Path) -> Thresholds:
    """Parse a thresholds YAML file into a `Thresholds` instance."""
    raw = load_yaml_mapping(path)

    saturation = require(raw, "saturation", str(path))
    if not isinstance(saturation, dict):
        raise ValueError(
            f"{path}: 'saturation' must be a mapping, got {type(saturation).__name__}"
        )

    return Thresholds(
        lag_p95_ms=float(require(saturation, "lag_p95_ms", f"{path}: saturation")),
        drop_rate_pct=float(require(saturation, "drop_rate_pct", f"{path}: saturation")),
        publisher_drift_reject_ms=float(
            require(raw, "publisher_drift_reject_ms", str(path))
        ),
        warmup_s=float(require(raw, "warmup_s", str(path))),
        measure_s=float(require(raw, "measure_s", str(path))),
    )
