"""Loads `configs/thresholds.yaml` — the pre-registered saturation and protocol
thresholds from THREATS-TO-VALIDITY.md. Values are read, never hardcoded, and this
module never writes to that file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Thresholds:
    lag_p95_ms: float
    drop_rate_pct: float
    publisher_drift_reject_ms: float
    warmup_s: float
    measure_s: float


def load_thresholds(path: Path) -> Thresholds:
    """Parse a thresholds YAML file into a `Thresholds` instance."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    saturation = data["saturation"]
    return Thresholds(
        lag_p95_ms=float(saturation["lag_p95_ms"]),
        drop_rate_pct=float(saturation["drop_rate_pct"]),
        publisher_drift_reject_ms=float(data["publisher_drift_reject_ms"]),
        warmup_s=float(data["warmup_s"]),
        measure_s=float(data["measure_s"]),
    )
