"""v0.4 decode-only sweep: ffmpeg-subprocess decode benchmark
(`rig/ffmpeg_decode_bench.py`), run as N concurrent processes per (N, codec,
hwaccel) configuration — entirely separate from the RTSP/cv2 pipeline sweep
(`bench/sweep.py`). No mediamtx, no publisher, no queue: just N ffmpeg processes
decoding the same local content file concurrently, which is what actually reaches
NVDEC/QSV/VAAPI on a machine where `cv2.VideoCapture` cannot (see
`rig/backends.py`).

Results are written under `runs/<id>/decode_only/`, a directory sibling kept
structurally separate from `runs/<id>/points/` so `bench/analyze.py` can never
accidentally fold decode-only fps numbers into the full-pipeline table — they
measure different things (raw ffmpeg decode throughput here, RTSP ingest_lag /
effective consumer fps there) and must not be compared cell-for-cell.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from multicam_bench.bench.env import collect_env
from multicam_bench.bench.sweep_config import SweepConfig, load_sweep_config
from multicam_bench.bench.sweep_plan import randomize_order
from multicam_bench.rig.backends import SystemBackendProbe, detect_backends
from multicam_bench.rig.ffmpeg_decode_bench import run_ffmpeg_decode_benchmark
from multicam_bench.rig.generate import generate_test_video
from multicam_bench.rig.resolution import resolve_resolution

# cv2 backend name (rig/backends.py) -> ffmpeg -hwaccel name
# (rig/ffmpeg_decode_bench.py). "ffmpeg-cpu" has no hardware acceleration at all,
# hence "none" (which means "omit -hwaccel entirely").
_CV2_BACKEND_TO_FFMPEG_HWACCEL = {
    "ffmpeg-cpu": "none",
    "d3d11va": "d3d11va",
    "qsv": "qsv",
    "cuda": "cuda",
    "vaapi": "vaapi",
}


def cv2_backend_to_ffmpeg_hwaccel(backend: str) -> str:
    try:
        return _CV2_BACKEND_TO_FFMPEG_HWACCEL[backend]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend {backend!r}; choose from "
            f"{sorted(_CV2_BACKEND_TO_FFMPEG_HWACCEL)}"
        ) from exc


@dataclass(frozen=True)
class DecodeOnlyPoint:
    n_streams: int
    codec: str
    hwaccel: str
    repetition: int

    def dir_name(self) -> str:
        return f"N{self.n_streams}_{self.codec}_{self.hwaccel}_rep{self.repetition}"


def build_decode_only_plan(
    n_values: list[int],
    codecs: list[str],
    hwaccels: list[str],
    repetitions: int,
    rng: random.Random,
) -> list[DecodeOnlyPoint]:
    """Randomised (per repetition) plan over the full (N, codec, hwaccel) cross
    product — same T2 rationale as the full-pipeline sweep's `build_sweep_plan`.
    Unlike that sweep, there is no publisher-only baseline here: a decode-only
    point has no publisher to baseline against.
    """
    configs = [
        (n, codec, hwaccel) for n in n_values for codec in codecs for hwaccel in hwaccels
    ]
    orders = randomize_order(configs, repetitions, rng)
    plan: list[DecodeOnlyPoint] = []
    for rep, order in enumerate(orders):
        for n, codec, hwaccel in order:
            plan.append(
                DecodeOnlyPoint(n_streams=n, codec=codec, hwaccel=hwaccel, repetition=rep)
            )
    return plan


def _content_video_path(resolution_name: str, codec: str) -> Path:
    return Path("data") / f"sweep_{resolution_name}_{codec}.mp4"


def _ensure_content_videos(config: SweepConfig) -> dict[str, Path]:
    resolution = resolve_resolution(config.resolution)
    videos: dict[str, Path] = {}
    for codec in config.codecs:
        path = _content_video_path(config.resolution, codec)
        if not path.exists():
            generate_test_video(
                path,
                resolution.width,
                resolution.height,
                config.fps,
                config.duration_s,
                codec=codec,
            )
        videos[codec] = path
    return videos


def _run_point(point: DecodeOnlyPoint, point_dir: Path, video_path: Path) -> None:
    """Launch `point.n_streams` ffmpeg decode benchmarks concurrently. The Python
    threads here do no decoding themselves — each just blocks on one ffmpeg
    subprocess — so the actual decode work still happens in N separate OS
    processes (CLAUDE.md hard rule 8), the threads are only a convenient way to
    start and wait on all of them at once.
    """
    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / "config.json").write_text(
        json.dumps(asdict(point), indent=2), encoding="utf-8"
    )

    with ThreadPoolExecutor(max_workers=max(1, point.n_streams)) as executor:
        futures = [
            executor.submit(run_ffmpeg_decode_benchmark, video_path, point.hwaccel)
            for _ in range(point.n_streams)
        ]
        results = [f.result() for f in futures]

    (point_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )


def run_decode_only_sweep(
    sweep_config_path: Path,
    run_id: str = "",
    seed: int | None = None,
    machine_label: str = "unlabelled-machine",
) -> Path:
    """Run the decode-only benchmark described by `sweep_config_path`'s N/codec/
    backend axes. Returns `runs/<run_id>/decode_only/`.
    """
    config = load_sweep_config(sweep_config_path)

    resolved_run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    decode_only_dir = Path("runs") / resolved_run_id / "decode_only"
    decode_only_dir.mkdir(parents=True, exist_ok=True)

    (decode_only_dir / "env.json").write_text(
        json.dumps(collect_env(machine_label=machine_label), indent=2), encoding="utf-8"
    )

    # Recorded for provenance only — unlike the full-pipeline sweep, decode-only
    # mode does NOT prefilter by cv2 availability: bypassing cv2's limits (e.g.
    # its 0-CUDA-device opencv-python-headless build) is the entire point. Every
    # requested backend is tried via ffmpeg directly; ffmpeg's own per-point
    # success/failure is the ground truth, not this detection pass.
    backends = detect_backends(SystemBackendProbe())
    (decode_only_dir / "backends.json").write_text(
        json.dumps([asdict(b) for b in backends], indent=2), encoding="utf-8"
    )
    hwaccels = [cv2_backend_to_ffmpeg_hwaccel(b) for b in config.backends]

    (decode_only_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "mode": "decode_only",
                "n_values": config.n_values,
                "codecs": config.codecs,
                "hwaccels": hwaccels,
                "repetitions": config.repetitions,
                "resolution": config.resolution,
                "fps": config.fps,
                "duration_s": config.duration_s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    videos = _ensure_content_videos(config)

    rng = random.Random(seed)
    plan = build_decode_only_plan(
        config.n_values, config.codecs, hwaccels, config.repetitions, rng
    )

    for index, point in enumerate(plan):
        point_dir = decode_only_dir / "points" / point.dir_name()
        _run_point(point, point_dir, videos[point.codec])
        if index < len(plan) - 1:
            time.sleep(config.cooldown_s)

    return decode_only_dir
