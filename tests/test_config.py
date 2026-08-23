from __future__ import annotations

from pathlib import Path

from multicam_bench.config import load_thresholds

REPO_ROOT = Path(__file__).parents[1]


def test_load_thresholds(tmp_path: Path) -> None:
    text = """\
saturation:
  lag_p95_ms: 200
  drop_rate_pct: 1.0
publisher_drift_reject_ms: 50
warmup_s: 20
measure_s: 90
"""
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")

    thresholds = load_thresholds(path)

    assert thresholds.lag_p95_ms == 200
    assert thresholds.drop_rate_pct == 1.0
    assert thresholds.publisher_drift_reject_ms == 50
    assert thresholds.warmup_s == 20
    assert thresholds.measure_s == 90


def test_repo_thresholds_match_pre_registered_values() -> None:
    # Guards THREATS-TO-VALIDITY.md §1.1: these are fixed before data collection
    # and must not silently change. configs/thresholds.yaml is never written by code.
    thresholds = load_thresholds(REPO_ROOT / "configs" / "thresholds.yaml")

    assert thresholds.lag_p95_ms == 200
    assert thresholds.drop_rate_pct == 1.0
    assert thresholds.publisher_drift_reject_ms == 50
    assert thresholds.warmup_s == 20
    assert thresholds.measure_s == 90
