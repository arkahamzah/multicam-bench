"""v0.3/v0.4 analysis: reads every sweep run under `runs/`, aggregates median+IQR
per (N, codec, backend) (CLAUDE.md hard rule 4 — never mean), applies the
pre-registered saturation criterion (THREATS-TO-VALIDITY.md §1.1) to find N_max per
decode configuration, excludes loop-boundary frames (T4, count reported), and
writes RESULTS.md with backend/codec comparison plots. Every number is written next
to its test content and machine spec — a stream count with no pixel rate or machine
attached is meaningless (CLAUDE.md hard rule 5).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — backend must be set before this import

from multicam_bench.bench.aggregate import MedianIQR, exclude_near_loop_boundary, median_iqr
from multicam_bench.bench.saturation import (
    SaturationResult,
    compute_n_max,
    detect_saturation,
    percentile_ms,
)
from multicam_bench.config import Thresholds, load_thresholds


@dataclass(frozen=True)
class CameraPointAnalysis:
    n_streams: int
    codec: str
    backend: str
    repetition: int
    camera_id: int
    fps_effective: float
    lag_p95_ms: float
    drop_rate_pct: float
    excluded_boundary_frames: int
    saturated: bool
    saturation_reason: str | None
    rejected: bool
    reject_reason: str | None


@dataclass(frozen=True)
class ConfigSummary:
    n_streams: int
    codec: str
    backend: str
    fps: MedianIQR
    lag_p95_ms: MedianIQR
    drop_rate_pct: MedianIQR
    saturated: bool
    n_points: int
    n_rejected: int


def discover_sweep_dirs(runs_root: Path) -> list[Path]:
    """Sweep run directories are those containing `sweep_config.json` — this
    distinguishes v0.2+ sweep output from a bare v0.1 single-stream `measure` run.
    """
    if not runs_root.is_dir():
        return []
    return sorted(d.parent for d in runs_root.glob("*/sweep_config.json"))


def _read_source_spec(sweep_dir: Path, codec: str) -> tuple[float, int]:
    data = json.loads((sweep_dir / "source.json").read_text(encoding="utf-8"))
    entry = data[codec]
    return float(entry["fps"]), int(entry["frame_count"])


def analyze_camera_point(
    samples_path: Path,
    period_frames: int,
    fps_source: float,
    thresholds: Thresholds,
    n_streams: int,
    codec: str,
    backend: str,
    repetition: int,
    camera_id: int,
    measure_s: float,
    queue_drops: int,
    rejected: bool,
    reject_reason: str | None,
) -> CameraPointAnalysis:
    """Read one camera's samples.csv, exclude loop-boundary frames (T4), and
    evaluate the pre-registered saturation criterion (§1.1) for this data point.
    """
    frame_indices: list[int] = []
    t_recvs: list[float] = []
    lags_s: list[float] = []
    gaps: list[int] = []
    with samples_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frame_indices.append(int(row["frame_index"]))
            t_recvs.append(float(row["t_recv"]))
            lags_s.append(float(row["ingest_lag_s"]))
            gaps.append(int(row["gap"]))

    frames_total = len(frame_indices)
    kept_indices, excluded_count = exclude_near_loop_boundary(frame_indices, period_frames)
    kept_set = set(kept_indices)

    t0 = t_recvs[0] if t_recvs else 0.0
    lag_samples_ms: list[tuple[float, float]] = []
    drop_events_t_s: list[float] = []
    for idx, t_recv, lag_s, gap in zip(frame_indices, t_recvs, lags_s, gaps, strict=True):
        if idx not in kept_set:
            continue
        t_s = t_recv - t0
        lag_samples_ms.append((t_s, lag_s * 1000.0))
        drop_events_t_s.extend([t_s] * gap)

    fps_effective = frames_total / measure_s if measure_s > 0 else 0.0

    # "we dropped it" (queue_drops, exact) + "the decoder skipped it" (embedded-index
    # gaps) both count toward the drop rate the saturation criterion cares about.
    total_drop_events = sum(gaps) + queue_drops
    expected_frames = fps_source * measure_s
    drop_rate_pct = (
        (total_drop_events / expected_frames * 100.0) if expected_frames > 0 else 0.0
    )

    if lag_samples_ms:
        saturation = detect_saturation(lag_samples_ms, drop_events_t_s, fps_source, thresholds)
        lag_values_ms = [lag for _, lag in lag_samples_ms]
        lag_p95_ms = percentile_ms(lag_values_ms, 95)
    else:
        saturation = SaturationResult(
            saturated=False, first_saturated_window_start_s=None, reason=None
        )
        lag_p95_ms = 0.0

    return CameraPointAnalysis(
        n_streams=n_streams,
        codec=codec,
        backend=backend,
        repetition=repetition,
        camera_id=camera_id,
        fps_effective=fps_effective,
        lag_p95_ms=lag_p95_ms,
        drop_rate_pct=drop_rate_pct,
        excluded_boundary_frames=excluded_count,
        saturated=saturation.saturated,
        saturation_reason=saturation.reason,
        rejected=rejected,
        reject_reason=reject_reason,
    )


def analyze_sweep(sweep_dir: Path, thresholds: Thresholds) -> list[CameraPointAnalysis]:
    """Analyze every measured (non-baseline) camera point in one sweep run."""
    results: list[CameraPointAnalysis] = []

    points_dir = sweep_dir / "points"
    if not points_dir.is_dir():
        return results

    source_cache: dict[str, tuple[float, int]] = {}

    for point_dir_path in sorted(points_dir.iterdir()):
        config_path = point_dir_path / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["publisher_only"]:
            continue  # baseline points carry no camera samples to analyze

        codec = config["codec"]
        backend = config["backend"]
        if codec not in source_cache:
            source_cache[codec] = _read_source_spec(sweep_dir, codec)
        fps_source, period_frames = source_cache[codec]

        cameras_path = point_dir_path / "cameras.json"
        if not cameras_path.is_file():
            continue
        cameras = json.loads(cameras_path.read_text(encoding="utf-8"))

        for cam in cameras:
            if cam.get("failed"):
                continue
            samples_path = Path(cam["samples_path"])
            if not samples_path.is_file():
                continue
            results.append(
                analyze_camera_point(
                    samples_path=samples_path,
                    period_frames=period_frames,
                    fps_source=fps_source,
                    thresholds=thresholds,
                    n_streams=config["n_streams"],
                    codec=codec,
                    backend=backend,
                    repetition=config["repetition"],
                    camera_id=cam["camera_id"],
                    measure_s=thresholds.measure_s,
                    queue_drops=cam["queue_drops"],
                    rejected=cam["rejected"],
                    reject_reason=cam["reject_reason"],
                )
            )
    return results


def summarize_by_config(results: list[CameraPointAnalysis]) -> list[ConfigSummary]:
    """Median+IQR per (N, codec, backend), over non-rejected camera points only
    (rejected points are excluded from the numbers but still counted, per
    THREATS-TO-VALIDITY.md T1).
    """
    by_key: dict[tuple[int, str, str], list[CameraPointAnalysis]] = {}
    for r in results:
        by_key.setdefault((r.n_streams, r.codec, r.backend), []).append(r)

    zero = MedianIQR(0.0, 0.0, 0.0)
    summaries = []
    for (n, codec, backend), points in sorted(by_key.items()):
        usable = [p for p in points if not p.rejected]
        saturated = any(p.saturated for p in usable)
        if usable:
            fps_stats = median_iqr([p.fps_effective for p in usable])
            lag_stats = median_iqr([p.lag_p95_ms for p in usable])
            drop_stats = median_iqr([p.drop_rate_pct for p in usable])
        else:
            fps_stats = lag_stats = drop_stats = zero
        summaries.append(
            ConfigSummary(
                n_streams=n,
                codec=codec,
                backend=backend,
                fps=fps_stats,
                lag_p95_ms=lag_stats,
                drop_rate_pct=drop_stats,
                saturated=saturated,
                n_points=len(usable),
                n_rejected=len(points) - len(usable),
            )
        )
    return summaries


def n_max_by_decode_config(
    summaries: list[ConfigSummary],
) -> dict[tuple[str, str], int | None]:
    """N_max (THREATS-TO-VALIDITY.md §1.1) computed independently per (codec,
    backend) — capacity is a property of the decode configuration, not one number
    for the whole sweep.
    """
    groups: dict[tuple[str, str], dict[int, bool]] = {}
    for s in summaries:
        groups.setdefault((s.codec, s.backend), {})[s.n_streams] = s.saturated
    return {key: compute_n_max(saturated_by_n) for key, saturated_by_n in groups.items()}


def _group_by_decode_config(
    summaries: list[ConfigSummary],
) -> dict[tuple[str, str], list[ConfigSummary]]:
    groups: dict[tuple[str, str], list[ConfigSummary]] = {}
    for s in summaries:
        groups.setdefault((s.codec, s.backend), []).append(s)
    return groups


def plot_fps_vs_n(summaries: list[ConfigSummary], output_path: Path) -> None:
    fig, ax = plt.subplots()
    for (codec, backend), points in sorted(_group_by_decode_config(summaries).items()):
        points = sorted(points, key=lambda p: p.n_streams)
        ns = [p.n_streams for p in points]
        medians = [p.fps.median for p in points]
        lo = [max(0.0, p.fps.median - p.fps.q1) for p in points]
        hi = [max(0.0, p.fps.q3 - p.fps.median) for p in points]
        label = f"{codec}/{backend}"
        ax.errorbar(ns, medians, yerr=[lo, hi], fmt="o-", capsize=4, label=label)
    ax.set_xlabel("N streams")
    ax.set_ylabel("effective fps (median, IQR)")
    ax.set_title("Effective fps vs N")
    if summaries:
        ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_lag_p95_vs_n(summaries: list[ConfigSummary], output_path: Path) -> None:
    fig, ax = plt.subplots()
    for (codec, backend), points in sorted(_group_by_decode_config(summaries).items()):
        points = sorted(points, key=lambda p: p.n_streams)
        ns = [p.n_streams for p in points]
        medians = [p.lag_p95_ms.median for p in points]
        lo = [max(0.0, p.lag_p95_ms.median - p.lag_p95_ms.q1) for p in points]
        hi = [max(0.0, p.lag_p95_ms.q3 - p.lag_p95_ms.median) for p in points]
        label = f"{codec}/{backend}"
        ax.errorbar(ns, medians, yerr=[lo, hi], fmt="o-", capsize=4, label=label)
    ax.set_xlabel("N streams")
    ax.set_ylabel("ingest_lag p95 (ms, median, IQR)")
    ax.set_title("ingest_lag p95 vs N")
    if summaries:
        ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_results_section(
    sweep_dir: Path,
    env: dict[str, object],
    sweep_config: dict[str, object],
    summaries: list[ConfigSummary],
    total_excluded_boundary_frames: int,
    skipped_backends: dict[str, str],
) -> str:
    gpus = env.get("gpus")
    gpu_str = ", ".join(str(g) for g in gpus) if isinstance(gpus, list) and gpus else "none"
    ram_bytes = env.get("ram_total_bytes")
    ram_gb = (ram_bytes if isinstance(ram_bytes, int | float) else 0) / 1e9

    n_max_by_config = n_max_by_decode_config(summaries)
    codecs = sweep_config.get("codecs")
    codecs_str = ", ".join(str(c) for c in codecs) if isinstance(codecs, list) else "?"

    lines = [
        f"## {sweep_dir.name}",
        "",
        f"**Content:** {sweep_config.get('resolution', '?')} · "
        f"{sweep_config.get('fps', '?')} fps · "
        f"{sweep_config.get('duration_s', '?')}s test clip · "
        f"codecs swept: {codecs_str}",
        f"**Machine:** {env.get('cpu_model', 'unknown')} "
        f"({env.get('cpu_cores_physical', '?')}c/{env.get('cpu_cores_logical', '?')}t), "
        f"{ram_gb:.1f} GB RAM, GPU: {gpu_str}",
        "",
        f"Loop-boundary frames excluded (±5 of a loop restart): "
        f"{total_excluded_boundary_frames}",
    ]

    if skipped_backends:
        lines.append("Backends requested but unavailable on this machine (v0.4):")
        for name, reason in sorted(skipped_backends.items()):
            lines.append(f"- `{name}`: {reason}")
    lines.append("")

    lines.append("N_max per decode configuration (largest N with no saturated camera):")
    for (codec, backend), n_max in sorted(n_max_by_config.items()):
        n_max_str = str(n_max) if n_max is not None else "none — every tested N saturated"
        lines.append(f"- {codec}/{backend}: {n_max_str}")
    lines.append("")

    lines.append(
        "| N | codec | backend | fps median [IQR] | lag p95 median [IQR] ms "
        "| drop rate median [IQR] % | saturated | rejected |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        total_points = s.n_points + s.n_rejected
        drop = s.drop_rate_pct
        lines.append(
            f"| {s.n_streams} | {s.codec} | {s.backend} "
            f"| {s.fps.median:.3f} [{s.fps.q1:.3f}, {s.fps.q3:.3f}] "
            f"| {s.lag_p95_ms.median:.1f} [{s.lag_p95_ms.q1:.1f}, {s.lag_p95_ms.q3:.1f}] "
            f"| {drop.median:.2f} [{drop.q1:.2f}, {drop.q3:.2f}] "
            f"| {'yes' if s.saturated else 'no'} "
            f"| {s.n_rejected}/{total_points} |"
        )
    lines.append("")
    lines.append(f"![fps vs N]({sweep_dir.name}/plots/fps_vs_n.png)")
    lines.append(f"![lag p95 vs N]({sweep_dir.name}/plots/lag_p95_vs_n.png)")
    lines.append("")
    return "\n".join(lines)


def run_analyze(runs_root: Path, thresholds_path: Path, output_path: Path) -> None:
    """Discover every sweep run under `runs_root`, analyze it, and write the
    combined `RESULTS.md` (plus one plot pair per sweep, next to its run).
    """
    thresholds = load_thresholds(thresholds_path)
    sweep_dirs = discover_sweep_dirs(runs_root)

    if not sweep_dirs:
        output_path.write_text(
            "# Results\n\nNo completed sweep runs found under `runs/`. "
            "Run `multicam-bench sweep` first.\n",
            encoding="utf-8",
        )
        return

    sections = ["# Results", ""]
    for sweep_dir in sweep_dirs:
        env = json.loads((sweep_dir / "env.json").read_text(encoding="utf-8"))
        sweep_config = json.loads(
            (sweep_dir / "sweep_config.json").read_text(encoding="utf-8")
        )
        skipped_path = sweep_dir / "backends_skipped.json"
        skipped_backends = (
            json.loads(skipped_path.read_text(encoding="utf-8"))
            if skipped_path.is_file()
            else {}
        )

        results = analyze_sweep(sweep_dir, thresholds)
        summaries = summarize_by_config(results)
        total_excluded = sum(r.excluded_boundary_frames for r in results)

        plots_dir = sweep_dir / "plots"
        plot_fps_vs_n(summaries, plots_dir / "fps_vs_n.png")
        plot_lag_p95_vs_n(summaries, plots_dir / "lag_p95_vs_n.png")

        sections.append(
            build_results_section(
                sweep_dir, env, sweep_config, summaries, total_excluded, skipped_backends
            )
        )

    output_path.write_text("\n".join(sections), encoding="utf-8")
