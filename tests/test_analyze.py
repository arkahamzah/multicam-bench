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
    n_max_by_decode_config,
    summarize_by_config,
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


def _write_measured_point(
    sweep_dir: Path,
    n_streams: int,
    codec: str,
    backend: str,
    repetition: int,
    *,
    rejected: bool = False,
) -> None:
    dir_name = f"N{n_streams}_{codec}_{backend}_rep{repetition}_measured"
    point_dir = sweep_dir / "points" / dir_name
    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / "config.json").write_text(
        json.dumps(
            {
                "n_streams": n_streams,
                "repetition": repetition,
                "order_index": 0,
                "publisher_only": False,
                "codec": codec,
                "backend": backend,
            }
        ),
        encoding="utf-8",
    )
    cam_samples = point_dir / "cam0" / "samples.csv"
    _write_camera_samples(cam_samples, list(range(5, 15)), lag_s=0.005)
    (point_dir / "cameras.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": 0,
                    "samples_path": str(cam_samples),
                    "frames_received": 10,
                    "frames_measured": 10,
                    "queue_drops": 0,
                    "backend": backend,
                    "drift_ms": 500.0 if rejected else 5.0,
                    "rejected": rejected,
                    "reject_reason": (
                        "publisher drift 500.0ms exceeds reject threshold 50.0ms"
                        if rejected
                        else None
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )


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
                "codecs": ["libx264"],
                "backends": ["ffmpeg-cpu"],
                "repetitions": 2,
                "cooldown_s": 1,
                "resolution": "360p",
                "fps": 10,
                "duration_s": 2.0,
            }
        ),
        encoding="utf-8",
    )
    (sweep_dir / "source.json").write_text(
        json.dumps({"libx264": {"fps": 10.0, "frame_count": 20}}), encoding="utf-8"
    )

    _write_measured_point(sweep_dir, 1, "libx264", "ffmpeg-cpu", repetition=0)
    _write_measured_point(sweep_dir, 1, "libx264", "ffmpeg-cpu", repetition=1, rejected=True)

    baseline_dir = sweep_dir / "points" / "N1_libx264_ffmpeg-cpu_rep0_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "config.json").write_text(
        json.dumps(
            {
                "n_streams": 1,
                "repetition": 0,
                "order_index": 0,
                "publisher_only": True,
                "codec": "libx264",
                "backend": "ffmpeg-cpu",
            }
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
    assert rep0.codec == "libx264"
    assert rep0.backend == "ffmpeg-cpu"


def test_analyze_sweep_marks_rejected_points(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    rep1 = next(r for r in results if r.repetition == 1)

    assert rep1.rejected is True
    assert rep1.reject_reason is not None


def test_summarize_by_config_excludes_rejected_points_from_stats(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    summaries = summarize_by_config(results)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.n_streams == 1
    assert summary.codec == "libx264"
    assert summary.backend == "ffmpeg-cpu"
    assert summary.n_points == 1  # only the clean rep0 point
    assert summary.n_rejected == 1
    assert summary.fps.median == 10.0
    assert summary.saturated is False


def test_summarize_by_config_keeps_codec_backend_groups_separate(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep2"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "source.json").write_text(
        json.dumps(
            {
                "libx264": {"fps": 10.0, "frame_count": 20},
                "libx265": {"fps": 10.0, "frame_count": 20},
            }
        ),
        encoding="utf-8",
    )
    _write_measured_point(sweep_dir, 1, "libx264", "ffmpeg-cpu", repetition=0)
    _write_measured_point(sweep_dir, 1, "libx265", "qsv", repetition=0)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    summaries = summarize_by_config(results)

    keys = {(s.n_streams, s.codec, s.backend) for s in summaries}
    assert keys == {(1, "libx264", "ffmpeg-cpu"), (1, "libx265", "qsv")}


def test_n_max_by_decode_config_is_independent_per_group(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "runs" / "sweep2"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "source.json").write_text(
        json.dumps({"libx264": {"fps": 10.0, "frame_count": 20}}), encoding="utf-8"
    )
    _write_measured_point(sweep_dir, 1, "libx264", "ffmpeg-cpu", repetition=0)
    _write_measured_point(sweep_dir, 2, "libx264", "qsv", repetition=0)

    results = analyze_sweep(sweep_dir, THRESHOLDS)
    summaries = summarize_by_config(results)
    n_max = n_max_by_decode_config(summaries)

    assert n_max == {("libx264", "ffmpeg-cpu"): 1, ("libx264", "qsv"): 2}


def test_build_results_section_includes_content_machine_and_skipped_backends(
    tmp_path: Path,
) -> None:
    sweep_dir = tmp_path / "runs" / "sweep1"
    _build_sweep(sweep_dir)

    env = json.loads((sweep_dir / "env.json").read_text(encoding="utf-8"))
    sweep_config = json.loads((sweep_dir / "sweep_config.json").read_text(encoding="utf-8"))
    results = analyze_sweep(sweep_dir, THRESHOLDS)
    summaries = summarize_by_config(results)

    section = build_results_section(
        sweep_dir,
        env,
        sweep_config,
        summaries,
        total_excluded_boundary_frames=0,
        skipped_backends={"cuda": "no NVIDIA GPU detected"},
    )

    assert "360p" in section
    assert "libx264" in section
    assert "Test CPU" in section
    assert "N_max" in section
    assert "cuda" in section
    assert "no NVIDIA GPU detected" in section
