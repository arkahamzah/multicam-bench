"""CLI-facing glue for `multicam-bench calc`: loads a machine profile from disk
and formats `evaluate_capacity`'s verdict as text. No profile ships with this
repo — every number in one would need a real sweep behind it (CLAUDE.md hard rule
5), and no sweep has been run yet.
"""

from __future__ import annotations

from pathlib import Path

from multicam_bench.model.machine_profile import (
    CapacityVerdict,
    MachineProfile,
    evaluate_capacity,
    load_machine_profile,
)


def run_calc(profile_path: Path, cameras: int, resolution: str, fps: float) -> str:
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"machine profile not found: {profile_path}. A profile is built from "
            "real sweep data (multicam-bench sweep, then a fit step) — none ships "
            "with this repo since no sweep has been run yet."
        )
    profile = load_machine_profile(profile_path)
    verdict = evaluate_capacity(profile, cameras, resolution, fps)
    return format_verdict(profile, cameras, resolution, fps, verdict)


def format_verdict(
    profile: MachineProfile,
    cameras: int,
    resolution: str,
    fps: float,
    verdict: CapacityVerdict,
) -> str:
    lines = [
        f"machine: {profile.machine_name}",
        f"request: {cameras} cameras @ {resolution} @ {fps} fps "
        f"(pixel_rate={verdict.pixel_rate:.0f})",
        f"fits: {'yes' if verdict.fits else 'no'}",
        f"limiting subsystem: {verdict.limiting_subsystem}",
        "",
        "per-subsystem predicted cost vs budget (headroom = budget - cost):",
    ]
    for name in sorted(verdict.predicted_cost):
        fit = profile.subsystems[name]
        lines.append(
            f"  {name}: cost={verdict.predicted_cost[name]:.2f} "
            f"budget={fit.budget:.2f} headroom={verdict.headroom[name]:.2f}"
        )
    return "\n".join(lines)
