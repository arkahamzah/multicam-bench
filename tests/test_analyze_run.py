"""Exercises the top-level `run_analyze` glue: full-pipeline and decode-only
results must land in separate RESULTS.md sections, never one merged table.
"""

from __future__ import annotations

from pathlib import Path

from test_analyze import _build_sweep
from test_analyze_decode_only import _build_decode_only_sweep

from multicam_bench.bench.analyze import run_analyze
from multicam_bench.config import Thresholds, load_thresholds


def _write_thresholds(path: Path) -> None:
    path.write_text(
        "saturation:\n"
        "  lag_p95_ms: 200\n"
        "  drop_rate_pct: 1.0\n"
        "publisher_drift_reject_ms: 50\n"
        "warmup_s: 20\n"
        "measure_s: 1\n",
        encoding="utf-8",
    )


def test_run_analyze_with_no_data_reports_gap(tmp_path: Path) -> None:
    thresholds_path = tmp_path / "thresholds.yaml"
    _write_thresholds(thresholds_path)
    output = tmp_path / "RESULTS.md"

    run_analyze(tmp_path / "runs", thresholds_path, output)

    text = output.read_text(encoding="utf-8")
    assert "No completed sweep runs found" in text


def test_run_analyze_keeps_full_pipeline_and_decode_only_in_separate_sections(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _build_sweep(runs_root / "sweep1")
    _build_decode_only_sweep(runs_root / "sweep2")

    thresholds_path = tmp_path / "thresholds.yaml"
    _write_thresholds(thresholds_path)
    output = tmp_path / "RESULTS.md"

    run_analyze(runs_root, thresholds_path, output)

    text = output.read_text(encoding="utf-8")
    assert "# Decode-only benchmark results" in text
    full_pipeline_section = text.split("# Decode-only benchmark results")[0]
    decode_only_section = text.split("# Decode-only benchmark results")[1]

    # The full-pipeline table's columns must never appear inside the decode-only
    # section, and vice versa — separate tables, not a merged one.
    assert "saturated" in full_pipeline_section
    assert "saturated" not in decode_only_section
    assert "hwaccel" in decode_only_section
    assert "hwaccel" not in full_pipeline_section


def test_run_analyze_handles_decode_only_data_with_no_full_pipeline_sweep(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _build_decode_only_sweep(runs_root / "sweep1")

    thresholds_path = tmp_path / "thresholds.yaml"
    _write_thresholds(thresholds_path)
    output = tmp_path / "RESULTS.md"

    run_analyze(runs_root, thresholds_path, output)

    text = output.read_text(encoding="utf-8")
    assert "# Decode-only benchmark results" in text
    assert "No completed sweep runs found" not in text


def test_thresholds_fixture_is_loadable(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    _write_thresholds(path)
    thresholds = load_thresholds(path)
    assert isinstance(thresholds, Thresholds)
