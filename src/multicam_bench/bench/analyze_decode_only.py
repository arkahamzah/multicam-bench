"""Analysis for the v0.4 decode-only (ffmpeg-subprocess) benchmark — kept in its
own module and its own RESULTS.md section/table, deliberately never merged with
`bench/analyze.py`'s full-pipeline table. The two measure different things: raw
ffmpeg decode throughput here (no RTSP, no queue, no `ingest_lag`), versus
effective consumer fps / ingest_lag / drop rate through the whole RTSP+cv2
pipeline there. A reader comparing them cell-for-cell would be comparing decode
speed to consumer throughput under contention — not a fair or meaningful
comparison, hence the hard separation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from multicam_bench.bench.aggregate import MedianIQR, median_iqr


@dataclass(frozen=True)
class DecodeOnlyPointResult:
    n_streams: int
    codec: str
    hwaccel: str
    repetition: int
    fps_values: list[float]
    failed_count: int


@dataclass(frozen=True)
class DecodeOnlyConfigSummary:
    n_streams: int
    codec: str
    hwaccel: str
    fps: MedianIQR
    n_points: int
    n_failed: int


def discover_decode_only_dirs(runs_root: Path) -> list[Path]:
    """`runs/<id>/decode_only/` directories — distinguished from full-pipeline
    sweeps by containing their own `sweep_config.json` one level deeper.
    """
    if not runs_root.is_dir():
        return []
    return sorted(d.parent for d in runs_root.glob("*/decode_only/sweep_config.json"))


def analyze_decode_only_point(point_dir: Path) -> DecodeOnlyPointResult | None:
    config_path = point_dir / "config.json"
    results_path = point_dir / "results.json"
    if not config_path.is_file() or not results_path.is_file():
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    fps_values = [r["fps"] for r in results if r.get("succeeded") and r.get("fps") is not None]
    failed_count = sum(1 for r in results if not r.get("succeeded"))

    return DecodeOnlyPointResult(
        n_streams=config["n_streams"],
        codec=config["codec"],
        hwaccel=config["hwaccel"],
        repetition=config["repetition"],
        fps_values=fps_values,
        failed_count=failed_count,
    )


def analyze_decode_only_sweep(decode_only_dir: Path) -> list[DecodeOnlyPointResult]:
    """Analyze every point in one `decode_only/` directory."""
    points_dir = decode_only_dir / "points"
    if not points_dir.is_dir():
        return []

    results: list[DecodeOnlyPointResult] = []
    for point_dir in sorted(points_dir.iterdir()):
        result = analyze_decode_only_point(point_dir)
        if result is not None:
            results.append(result)
    return results


def summarize_decode_only(
    results: list[DecodeOnlyPointResult],
) -> list[DecodeOnlyConfigSummary]:
    """Median+IQR fps per (N, codec, hwaccel), pooling every successful process
    across every repetition — never a mean (CLAUDE.md hard rule 4).
    """
    by_key: dict[tuple[int, str, str], list[DecodeOnlyPointResult]] = {}
    for r in results:
        by_key.setdefault((r.n_streams, r.codec, r.hwaccel), []).append(r)

    zero = MedianIQR(0.0, 0.0, 0.0)
    summaries = []
    for (n, codec, hwaccel), points in sorted(by_key.items()):
        all_fps = [fps for p in points for fps in p.fps_values]
        n_failed = sum(p.failed_count for p in points)
        fps_stats = median_iqr(all_fps) if all_fps else zero
        summaries.append(
            DecodeOnlyConfigSummary(
                n_streams=n,
                codec=codec,
                hwaccel=hwaccel,
                fps=fps_stats,
                n_points=len(all_fps),
                n_failed=n_failed,
            )
        )
    return summaries


def build_decode_only_results_section(
    decode_only_dir: Path,
    env: dict[str, object],
    sweep_config: dict[str, object],
    summaries: list[DecodeOnlyConfigSummary],
) -> str:
    sweep_name = decode_only_dir.parent.name
    gpus = env.get("gpus")
    gpu_str = ", ".join(str(g) for g in gpus) if isinstance(gpus, list) and gpus else "none"
    ram_bytes = env.get("ram_total_bytes")
    ram_gb = (ram_bytes if isinstance(ram_bytes, int | float) else 0) / 1e9
    hwaccels = sweep_config.get("hwaccels")
    hwaccels_str = ", ".join(str(h) for h in hwaccels) if isinstance(hwaccels, list) else "?"

    lines = [
        f"## {sweep_name} (decode-only — ffmpeg subprocess, NOT the full pipeline)",
        "",
        "This measures raw ffmpeg decode throughput (`ffmpeg -hwaccel <x> -i "
        "<file> -benchmark -f null -`), independent of RTSP/mediamtx/cv2. It "
        "answers a different question from the full-pipeline table above "
        "(effective consumer fps / ingest_lag under real ingest) and the two are "
        "never compared cell-for-cell.",
        "",
        f"**Content:** {sweep_config.get('resolution', '?')} · "
        f"{sweep_config.get('fps', '?')} fps · "
        f"{sweep_config.get('duration_s', '?')}s test clip · "
        f"hwaccels tried: {hwaccels_str}",
        f"**Machine:** {env.get('cpu_model', 'unknown')} "
        f"({env.get('cpu_cores_physical', '?')}c/{env.get('cpu_cores_logical', '?')}t), "
        f"{ram_gb:.1f} GB RAM, GPU: {gpu_str}",
        "",
        "| N | codec | hwaccel | decode fps median [IQR] | succeeded/total |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        total = s.n_points + s.n_failed
        lines.append(
            f"| {s.n_streams} | {s.codec} | {s.hwaccel} "
            f"| {s.fps.median:.2f} [{s.fps.q1:.2f}, {s.fps.q3:.2f}] "
            f"| {s.n_points}/{total} |"
        )
    lines.append("")
    return "\n".join(lines)
