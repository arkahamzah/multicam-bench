"""Exercises analyze.py against a hand-built sweep run directory — no ffmpeg or
mediamtx needed, since analyze.py only ever reads files a sweep run already wrote.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from multicam_bench.bench.analyze import (
    analyze_sweep,
    build_results_section,
    discover_sweep_dirs,
    summarize_by_n,
)
from multicam_bench.config import Thresholds

THRESHOLDS = Thresholds(
    lag_p95_ms=200.0,
    drop_rate_pct=1.0,
    publisher_drift_reject_ms=50.0,
    warmup_s=20.0,
    measure_s=1.0,
)


def _write_camera_samples(
    path: Path, frame_indices: list[int], lag_s: float, gap: int = 0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_index", "raw_index", "t_recv", "ingest_lag_s", "gap"])
        for i, idx in enumerate(frame_indices):
            t_recv = 100.0 + i * 0.1
            writer.writerow([idx, idx % 20, t_recv, lag_s, gap if i == 0 else 0])


def _build_sweep(sweep_dir: Path) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "env.json").write_text(
        json.dumps(
            {
                "cpu_model": "Test CPU",
                "cpu_cores_physical": 4,
                "cpu_cores_logical": 8,
                "ram_total_bytes": 16_000_000_000,
                "gpus": ["Test GPU"],
            }
        ),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "n_values": [1],
                "repetitions": 2,
                "cooldown_s": 1,
                "content": {
                    "resolution": "360p",
                    "codec": "libx264",
                    "fps": 10,
                    "duration_s": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (sweep_dir / "source.json").write_text(
        json.dumps({"fps": 10.0, "frame_count": 20}), encoding="utf-8"
    )

    # rep0: a clean, unsaturated, non-rejected point.
    rep0_dir = sweep_dir / "points" / "N1_rep0_measured"
    (rep0_dir).mkdir(parents=True, exist_ok=True)
    (rep0_dir / "config.json").write_text(
        json.dumps(
            {"n_streams": 1, "repetition": 0, "order_index": 0, "publisher_only": False}
        ),
        encoding="utf-8",
    )
    cam0_samples = rep0_dir / "cam0" / "samples.csv"
    _write_camera_samples(cam0_samples, list(range(5, 15)), lag_s=0.005)
    (rep0_dir / "cameras.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": 0,
                    "samples_path": str(cam0_samples),
                    "frames_received": 10,
                    "frames_measured": 10,
                    "queue_drops": 0,
                    "drift_ms": 5.0,
                    "rejected": False,
                    "reject_reason": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    # rep1: a rejected point (publisher drift too high) — must not enter the stats.
    rep1_dir = sweep_dir / "points" / "N1_rep1_measured"
    rep1_dir.mkdir(parents=True, exist_ok=True)
    (rep1_dir / "config.json").write_text(
        json.dumps(
            {"n_streams": 1, "repetition": 1, "order_index": 0, "publisher_only": False}
        ),
        encoding="utf-8",
    )
    cam0_rep1_samples = rep1_dir / "cam0" / "samples.csv"
    _write_camera_samples(cam0_rep1_samples, list(range(5, 15)), lag_s=0.005)
    (rep1_dir / "cameras.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": 0,
                    "samples_path": str(cam0_rep1_samples),
                    "frames_received": 10,
                    "frames_measured": 10,
                    "queue_drops": 0,
                    "drift_ms": 500.0,
                    "rejected": True,
                    "reject_reason": "publisher drift 500.0ms exceeds reject threshold 50.0ms",
                }
            ]
        ),
        encoding="utf-8",
    )

    # a baseline point must be skipped entirely by analyze_sweep.
    baseline_dir = sweep_dir / "points" / "N1_rep0_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "config.json").write_text(
        json.dumps(
            {"n_streams": 1, "repetition": 0, "order_index": 0, "publisher_only": True}
        ),
        encoding="utf-8",
    )


def test_discover_sweep_dirs_finds_sweep_config(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _build_sweep(runs_root / "sweep1")
    (runs_root / "not_a_sweep").mkdir(parents=True)  # no sweep_config.json — must be ignored

    found = discover_sweep_dirs(runs_root)

    assert found == [runs_root / "sweep1"]


def test_discover_sweep_dirs_on_missing_root_returns_empty(tmp_path: Path) -> None:
    assert discover_sweep_dirs(tmp_path / "does_not_exist") == []


def test_analyze_sweep_skips_baseline_points(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    results = analyze_sweep(sweep_dir, THRESHOLDS)

    assert len(results) == 2  # rep0 measured + rep1 measured, baseline excluded


def test_analyze_sweep_computes_fps_and_lag_for_clean_point(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    rep0 = next(r for r in results if r.repetition == 0)

    assert rep0.fps_effective == 10.0  # 10 frames / measure_s(1.0)
    assert rep0.lag_p95_ms == 5.0
    assert rep0.saturated is False
    assert rep0.rejected is False


def test_analyze_sweep_marks_rejected_points(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    rep1 = next(r for r in results if r.repetition == 1)

    assert rep1.rejected is True
    assert rep1.reject_reason is not None


def test_summarize_by_n_excludes_rejected_points_from_stats(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    summaries = summarize_by_n(results)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.n_streams == 1
    assert summary.n_points == 1  # only the clean rep0 point
    assert summary.n_rejected == 1
    assert summary.fps.median == 10.0
    assert summary.saturated is False


def test_build_results_section_includes_content_and_machine_and_n_max(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    env = json.loads((sweep_dir / "env.json").read_text(encoding="utf-8"))
    sweep_config = json.loads((sweep_dir / "sweep_config.json").read_text(encoding="utf-8"))
    content = sweep_config["content"]
    results = analyze_sweep(sweep_dir, THRESHOLDS)
    summaries = summarize_by_n(results)

    section = build_results_section(sweep_dir, env, content, summaries, 0, n_max=1)

    assert "360p" in section
    assert "libx264" in section
    assert "Test CPU" in section
    assert "N_max" in section
    assert "1" in section
