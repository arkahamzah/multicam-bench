from __future__ import annotations

import json
from pathlib import Path

from multicam_bench.bench.analyze_decode_only import (
    analyze_decode_only_sweep,
    build_decode_only_results_section,
    discover_decode_only_dirs,
    summarize_decode_only,
)


def _write_point(
    decode_only_dir: Path,
    n_streams: int,
    codec: str,
    hwaccel: str,
    repetition: int,
    fps_values: list[float],
    n_failed: int = 0,
) -> None:
    point_dir = decode_only_dir / "points" / f"N{n_streams}_{codec}_{hwaccel}_rep{repetition}"
    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / "config.json").write_text(
        json.dumps(
            {
                "n_streams": n_streams,
                "codec": codec,
                "hwaccel": hwaccel,
                "repetition": repetition,
            }
        ),
        encoding="utf-8",
    )
    results = [
        {
            "hwaccel": hwaccel,
            "succeeded": True,
            "frame_count": 100,
            "fps": fps,
            "utime_s": 1.0,
            "stime_s": 0.1,
            "rtime_s": 1.1,
            "returncode": 0,
            "error": None,
        }
        for fps in fps_values
    ] + [
        {
            "hwaccel": hwaccel,
            "succeeded": False,
            "frame_count": None,
            "fps": None,
            "utime_s": None,
            "stime_s": None,
            "rtime_s": None,
            "returncode": 1,
            "error": "decoder init failed",
        }
        for _ in range(n_failed)
    ]
    (point_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")


def _build_decode_only_sweep(sweep_dir: Path) -> Path:
    decode_only_dir = sweep_dir / "decode_only"
    decode_only_dir.mkdir(parents=True, exist_ok=True)
    (decode_only_dir / "env.json").write_text(
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
    (decode_only_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "mode": "decode_only",
                "n_values": [1, 2],
                "codecs": ["libx264"],
                "hwaccels": ["none", "cuda"],
                "repetitions": 1,
                "resolution": "360p",
                "fps": 30,
                "duration_s": 60,
            }
        ),
        encoding="utf-8",
    )
    _write_point(decode_only_dir, 1, "libx264", "none", 0, [240.0, 245.0])
    _write_point(decode_only_dir, 1, "libx264", "cuda", 0, [], n_failed=1)
    return decode_only_dir


def test_discover_decode_only_dirs_finds_it(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sweep_dir = runs_root / "sweep1"
    _build_decode_only_sweep(sweep_dir)
    (runs_root / "not_a_sweep").mkdir(parents=True)

    found = discover_decode_only_dirs(runs_root)

    assert found == [sweep_dir / "decode_only"]


def test_discover_decode_only_dirs_does_not_match_full_pipeline_sweeps(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    full_pipeline = runs_root / "sweep1"
    full_pipeline.mkdir(parents=True)
    (full_pipeline / "sweep_config.json").write_text("{}", encoding="utf-8")

    assert discover_decode_only_dirs(runs_root) == []


def test_analyze_decode_only_sweep_reads_fps_and_failures(tmp_path: Path) -> None:
    decode_only_dir = _build_decode_only_sweep(tmp_path / "runs" / "sweep1")

    results = analyze_decode_only_sweep(decode_only_dir)

    assert len(results) == 2
    clean = next(r for r in results if r.hwaccel == "none")
    failed = next(r for r in results if r.hwaccel == "cuda")
    assert clean.fps_values == [240.0, 245.0]
    assert clean.failed_count == 0
    assert failed.fps_values == []
    assert failed.failed_count == 1


def test_summarize_decode_only_groups_by_n_codec_hwaccel(tmp_path: Path) -> None:
    decode_only_dir = _build_decode_only_sweep(tmp_path / "runs" / "sweep1")
    results = analyze_decode_only_sweep(decode_only_dir)

    summaries = summarize_decode_only(results)

    keys = {(s.n_streams, s.codec, s.hwaccel) for s in summaries}
    assert keys == {(1, "libx264", "none"), (1, "libx264", "cuda")}

    none_summary = next(s for s in summaries if s.hwaccel == "none")
    assert none_summary.fps.median == 242.5
    assert none_summary.n_points == 2
    assert none_summary.n_failed == 0

    cuda_summary = next(s for s in summaries if s.hwaccel == "cuda")
    assert cuda_summary.n_points == 0
    assert cuda_summary.n_failed == 1


def test_results_section_never_mixes_with_full_pipeline_table(tmp_path: Path) -> None:
    decode_only_dir = _build_decode_only_sweep(tmp_path / "runs" / "sweep1")
    env = json.loads((decode_only_dir / "env.json").read_text(encoding="utf-8"))
    sweep_config = json.loads(
        (decode_only_dir / "sweep_config.json").read_text(encoding="utf-8")
    )
    results = analyze_decode_only_sweep(decode_only_dir)
    summaries = summarize_decode_only(results)

    section = build_decode_only_results_section(decode_only_dir, env, sweep_config, summaries)

    assert "decode-only" in section
    assert "NOT the full pipeline" in section
    assert "never compared cell-for-cell" in section
    assert "cuda" in section
    assert "hwaccel" in section
    # The full-pipeline table's column names must not appear here — they measure
    # different things and must stay visually distinct.
    assert "lag p95" not in section
    assert "saturated" not in section
