"""v0.2/v0.4 sweep orchestrator: runs the N x codec x backend sweep defined in
`configs/sweep.yaml`.

For each (N, codec, backend) configuration, in an order randomised per repetition
over the WHOLE cross product (THREATS-TO-VALIDITY.md T2), runs a publisher-only
baseline (harness CPU cost, measured separately per T5) followed by the measured
run (N reader processes, one per camera, per CLAUDE.md hard rule 8). Requested
decode backends that are not available on this machine are skipped before the plan
is even built, with the reason recorded in `backends.json` (v0.4, CLAUDE.md hard
rule 1). Publisher pacing drift is checked on every measured run against
`publisher_drift_reject_ms`; a run that exceeds it is written with `rejected: true`
and a reason so it can be shown as a gap, not silently folded into results (T1).
`cooldown_s` separates every run so package temperature settles (T2). Not invoked
by any test — this drives real subprocesses and real wall-clock time; see
bench/sweep_plan.py and rig/backends.py for the unit-tested logic it calls into.
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
from multicam_bench.bench.sweep_plan import (
    SweepPoint,
    build_sweep_plan,
    camera_dir,
    cross_product_configs,
    point_dir,
)
from multicam_bench.config import Thresholds, load_thresholds
from multicam_bench.rig.backends import (
    BackendAvailability,
    SystemBackendProbe,
    available_backend_names,
    detect_backends,
    hw_acceleration_for,
)
from multicam_bench.rig.generate import generate_test_video
from multicam_bench.rig.probe import VideoProbe, probe_video
from multicam_bench.rig.publisher import mediamtx_server, publisher
from multicam_bench.rig.resolution import resolve_resolution


def _content_video_path(resolution_name: str, codec: str) -> Path:
    return Path("data") / f"sweep_{resolution_name}_{codec}.mp4"


def _ensure_content_videos(config: SweepConfig) -> dict[str, tuple[Path, VideoProbe]]:
    """One video per codec (resolution/fps/duration are fixed for the whole sweep).
    Returns {codec: (video_path, probed_source)}.
    """
    resolution = resolve_resolution(config.resolution)
    videos: dict[str, tuple[Path, VideoProbe]] = {}
    for codec in config.codecs:
        path = _content_video_path(config.resolution, codec)
        if not path.exists():
            generate_test_video(
                path, resolution.width, resolution.height, config.fps, config.duration_s,
                codec=codec,
            )
        videos[codec] = (path, probe_video(path))
    return videos


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


def _prime_cpu_tracking(pids: list[int]) -> dict[int, psutil.Process]:
    """Start CPU%-over-time tracking for `pids` without blocking — `cpu_percent`
    measures the interval since the previous call, so priming now and reading
    later (`_read_cpu_tracking`) spans exactly whatever happens in between, no
    separate `time.sleep` needed.
    """
    procs = {}
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.cpu_percent(interval=None)
            procs[pid] = p
        except psutil.NoSuchProcess:
            continue
    return procs


def _read_cpu_tracking(procs: dict[int, psutil.Process]) -> dict[int, float]:
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
    separately from any measured-run cost (THREATS-TO-VALIDITY.md T5). `backend`
    does not affect this run (only the reader side uses it) but is still recorded
    on the point for provenance.
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
    backends: list[BackendAvailability],
) -> None:
    hw_acceleration = hw_acceleration_for(point.backend, backends)
    processes: list[tuple[int, multiprocessing.Process, Path, Path]] = []
    join_timeout = thresholds.warmup_s + thresholds.measure_s + 30.0
    publisher_pids: list[int] = []

    with contextlib.ExitStack() as stack:
        for i in range(point.n_streams):
            rtsp_url = f"{rtsp_base_url}/cam{i}"
            proc = stack.enter_context(publisher(video_path, rtsp_url))
            if proc.pid is not None:
                publisher_pids.append(proc.pid)

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
                    point.backend,
                    hw_acceleration,
                ),
            )
            reader_proc.start()
            processes.append((i, reader_proc, samples_path, summary_path))

        # Reader CPU is what the v0.6 fit model calls "cost" — priming right after
        # every process is up and reading right after the join loop below means the
        # tracked interval is exactly the warmup+measure window, no extra sleep.
        reader_pids = [p.pid for _, p, _, _ in processes if p.pid is not None]
        publisher_cpu_procs = _prime_cpu_tracking(publisher_pids)
        reader_cpu_procs = _prime_cpu_tracking(reader_pids)

        for _, reader_proc, _, _ in processes:
            reader_proc.join(timeout=join_timeout)
            if reader_proc.is_alive():
                reader_proc.terminate()
                reader_proc.join(timeout=5.0)

        publisher_cpu_pct = _read_cpu_tracking(publisher_cpu_procs)
        reader_cpu_pct = _read_cpu_tracking(reader_cpu_procs)

    point_dir_path.mkdir(parents=True, exist_ok=True)
    (point_dir_path / "measured_cpu.json").write_text(
        json.dumps(
            {
                "publisher_cpu_pct": {str(k): v for k, v in publisher_cpu_pct.items()},
                "reader_cpu_pct": {str(k): v for k, v in reader_cpu_pct.items()},
                "reader_cpu_pct_total": sum(reader_cpu_pct.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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
                "codecs": config.codecs,
                "backends": config.backends,
                "repetitions": config.repetitions,
                "cooldown_s": config.cooldown_s,
                "resolution": config.resolution,
                "fps": config.fps,
                "duration_s": config.duration_s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    backends = detect_backends(SystemBackendProbe())
    (sweep_dir / "backends.json").write_text(
        json.dumps([asdict(b) for b in backends], indent=2), encoding="utf-8"
    )
    active_backends = available_backend_names(SystemBackendProbe(), config.backends)
    skipped = [b for b in config.backends if b not in active_backends]
    if skipped:
        skipped_reasons = {b.name: b.reason for b in backends if b.name in skipped}
        (sweep_dir / "backends_skipped.json").write_text(
            json.dumps(skipped_reasons, indent=2), encoding="utf-8"
        )
    if not active_backends:
        raise RuntimeError(
            f"none of the requested backends {config.backends} are available on this "
            f"machine — see {sweep_dir / 'backends.json'} for reasons"
        )

    videos = _ensure_content_videos(config)
    (sweep_dir / "source.json").write_text(
        json.dumps(
            {
                codec: {"fps": source.fps, "frame_count": source.frame_count}
                for codec, (_path, source) in videos.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    configs = cross_product_configs(config.n_values, config.codecs, active_backends)
    rng = random.Random(seed)
    plan = build_sweep_plan(configs, config.repetitions, rng)

    with mediamtx_server(config.mediamtx_config):
        for index, point in enumerate(plan):
            point_dir_path = point_dir(sweep_dir, point)
            video_path, source = videos[point.codec]
            if point.publisher_only:
                _run_baseline(
                    point, point_dir_path, config.rtsp_base_url, video_path, thresholds
                )
            else:
                _run_measured(
                    point,
                    point_dir_path,
                    config.rtsp_base_url,
                    video_path,
                    source,
                    thresholds,
                    backends,
                )

            if index < len(plan) - 1:
                time.sleep(config.cooldown_s)

    return sweep_dir
