"""v0.2 sweep orchestrator: runs the N-stream sweep defined in `configs/sweep.yaml`.

For each (N, repetition) pair, in an order randomised per repetition
(THREATS-TO-VALIDITY.md T2), runs a publisher-only baseline (harness CPU cost,
measured separately per T5) followed by the measured run (N reader processes, one
per camera, per CLAUDE.md hard rule 8). Publisher pacing drift is checked on every
measured run against `publisher_drift_reject_ms`; a run that exceeds it is written
with `rejected: true` and a reason so it can be shown as a gap, not silently folded
into results (T1). `cooldown_s` separates every run so package temperature settles
(T2). Not invoked by any test — this drives real subprocesses and real wall-clock
time; see bench/sweep_plan.py for the unit-tested ordering logic it calls into.
"""

from __future__ import annotations

import contextlib
import csv
import json
import multiprocessing
import random
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from multicam_bench.bench.drift import evaluate_publisher_drift
from multicam_bench.bench.env import collect_env
from multicam_bench.bench.multi_reader import camera_process_entry
from multicam_bench.bench.sweep_config import SweepConfig, load_sweep_config
from multicam_bench.bench.sweep_plan import SweepPoint, build_sweep_plan, camera_dir, point_dir
from multicam_bench.config import Thresholds, load_thresholds
from multicam_bench.rig.generate import generate_test_video
from multicam_bench.rig.probe import VideoProbe, probe_video
from multicam_bench.rig.publisher import mediamtx_server, publisher
from multicam_bench.rig.resolution import resolve_resolution


def _content_video_path(config: SweepConfig) -> Path:
    return Path("data") / f"sweep_{config.content.resolution}_{config.content.codec}.mp4"


def _ensure_content_video(config: SweepConfig) -> Path:
    resolution = resolve_resolution(config.content.resolution)
    path = _content_video_path(config)
    if not path.exists():
        generate_test_video(
            path,
            resolution.width,
            resolution.height,
            config.content.fps,
            config.content.duration_s,
            codec=config.content.codec,
        )
    return path


def _sample_cpu_percent(
    pids: list[int], warmup_s: float, measure_s: float
) -> dict[int, float]:
    """Best-effort per-PID CPU%, primed then read after warmup+measure elapses — a
    process that has already exited by read time is simply omitted.
    """
    procs = {}
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.cpu_percent(interval=None)
            procs[pid] = p
        except psutil.NoSuchProcess:
            continue
    time.sleep(warmup_s + measure_s)
    result = {}
    for pid, p in procs.items():
        with contextlib.suppress(psutil.NoSuchProcess):
            result[pid] = p.cpu_percent(interval=None)
    return result


def _read_frame_index_and_t_recv(samples_path: Path) -> tuple[list[int], list[float]]:
    frame_indices: list[int] = []
    t_recvs: list[float] = []
    with samples_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frame_indices.append(int(row["frame_index"]))
            t_recvs.append(float(row["t_recv"]))
    return frame_indices, t_recvs


def _run_baseline(
    point: SweepPoint,
    point_dir_path: Path,
    rtsp_base_url: str,
    video_path: Path,
    thresholds: Thresholds,
) -> None:
    """Publisher-only: no reader processes, just N publishers. CPU cost is recorded
    separately from any measured-run cost (THREATS-TO-VALIDITY.md T5).
    """
    with contextlib.ExitStack() as stack:
        pids = []
        for i in range(point.n_streams):
            rtsp_url = f"{rtsp_base_url}/cam{i}"
            proc = stack.enter_context(publisher(video_path, rtsp_url))
            if proc.pid is not None:
                pids.append(proc.pid)

        cpu_percent = _sample_cpu_percent(pids, thresholds.warmup_s, thresholds.measure_s)

    point_dir_path.mkdir(parents=True, exist_ok=True)
    (point_dir_path / "publisher_cpu.json").write_text(
        json.dumps({str(pid): pct for pid, pct in cpu_percent.items()}, indent=2),
        encoding="utf-8",
    )
    (point_dir_path / "config.json").write_text(
        json.dumps(asdict(point), indent=2), encoding="utf-8"
    )


def _run_measured(
    point: SweepPoint,
    point_dir_path: Path,
    rtsp_base_url: str,
    video_path: Path,
    source: VideoProbe,
    thresholds: Thresholds,
) -> None:
    processes: list[tuple[int, multiprocessing.Process, Path, Path]] = []
    join_timeout = thresholds.warmup_s + thresholds.measure_s + 30.0

    with contextlib.ExitStack() as stack:
        for i in range(point.n_streams):
            rtsp_url = f"{rtsp_base_url}/cam{i}"
            stack.enter_context(publisher(video_path, rtsp_url))

        time.sleep(2.0)  # let publishers establish their RTSP sessions

        for i in range(point.n_streams):
            rtsp_url = f"{rtsp_base_url}/cam{i}"
            cam_dir_path = camera_dir(point_dir_path, i)
            samples_path = cam_dir_path / "samples.csv"
            summary_path = cam_dir_path / "summary.json"
            reader_proc = multiprocessing.Process(
                target=camera_process_entry,
                args=(
                    i,
                    rtsp_url,
                    samples_path,
                    source.fps,
                    source.frame_count,
                    thresholds.warmup_s,
                    thresholds.measure_s,
                    summary_path,
                ),
            )
            reader_proc.start()
            processes.append((i, reader_proc, samples_path, summary_path))

        for _, reader_proc, _, _ in processes:
            reader_proc.join(timeout=join_timeout)
            if reader_proc.is_alive():
                reader_proc.terminate()
                reader_proc.join(timeout=5.0)

    cameras = []
    for camera_id, _reader_proc, samples_path, summary_path in processes:
        if not summary_path.exists():
            cameras.append(
                {
                    "camera_id": camera_id,
                    "failed": True,
                    "reason": "reader process produced no summary (crash or timeout)",
                }
            )
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame_indices, t_recvs = _read_frame_index_and_t_recv(samples_path)
        drift_ms = None
        rejected = False
        reject_reason = None
        if len(frame_indices) >= 2:
            drift = evaluate_publisher_drift(
                frame_indices, t_recvs, source.fps, thresholds.publisher_drift_reject_ms
            )
            drift_ms = drift.drift_ms
            rejected = drift.rejected
            reject_reason = drift.reason
        summary["drift_ms"] = drift_ms
        summary["rejected"] = rejected
        summary["reject_reason"] = reject_reason
        cameras.append(summary)

    point_dir_path.mkdir(parents=True, exist_ok=True)
    (point_dir_path / "config.json").write_text(
        json.dumps(asdict(point), indent=2), encoding="utf-8"
    )
    (point_dir_path / "cameras.json").write_text(
        json.dumps(cameras, indent=2), encoding="utf-8"
    )


def run_sweep(
    sweep_config_path: Path,
    thresholds_path: Path,
    run_id: str = "",
    seed: int | None = None,
) -> Path:
    """Run the full sweep described by `sweep_config_path`. Returns the sweep run
    directory (`runs/<run_id>/`).
    """
    config = load_sweep_config(sweep_config_path)
    thresholds = load_thresholds(thresholds_path)

    resolved_run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    sweep_dir = Path("runs") / resolved_run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "env.json").write_text(json.dumps(collect_env(), indent=2), encoding="utf-8")
    (sweep_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "n_values": config.n_values,
                "repetitions": config.repetitions,
                "cooldown_s": config.cooldown_s,
                "content": asdict(config.content),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    video_path = _ensure_content_video(config)
    source = probe_video(video_path)
    (sweep_dir / "source.json").write_text(
        json.dumps({"fps": source.fps, "frame_count": source.frame_count}, indent=2),
        encoding="utf-8",
    )

    rng = random.Random(seed)
    plan = build_sweep_plan(config.n_values, config.repetitions, rng)

    with mediamtx_server(config.mediamtx_config):
        for index, point in enumerate(plan):
            point_dir_path = point_dir(sweep_dir, point)
            if point.publisher_only:
                _run_baseline(
                    point, point_dir_path, config.rtsp_base_url, video_path, thresholds
                )
            else:
                _run_measured(
                    point, point_dir_path, config.rtsp_base_url, video_path, source, thresholds
                )

            if index < len(plan) - 1:
                time.sleep(config.cooldown_s)

    return sweep_dir
