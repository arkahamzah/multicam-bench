"""Machine capacity profile: per-subsystem fitted coefficients (`model/fit.py`)
plus a measured budget (the max sustainable cost before saturation), used to
answer the v0.6 calculator question — does (cameras, resolution, fps) fit, what's
the limiting subsystem, how much headroom (`multicam-bench calc`).

THREATS-TO-VALIDITY.md T9: fps is always an input to these functions, never
inferred. T7: cross-machine prediction is only made — and only valid — within the
same bottleneck regime, checked explicitly in `predict_cross_machine`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from multicam_bench.rig.resolution import resolve_resolution


@dataclass(frozen=True)
class SubsystemFit:
    name: str
    a: float  # cost per unit pixel_rate
    b: float  # cost per additional stream
    budget: float  # max sustainable cost before this subsystem saturates
    content_name: str  # what test content the fit was measured on
    machine_name: str


@dataclass(frozen=True)
class MachineProfile:
    machine_name: str
    binding_subsystem: str  # which subsystem was empirically the constraint here
    subsystems: dict[str, SubsystemFit]


def save_machine_profile(profile: MachineProfile, path: Path) -> None:
    data = {
        "machine_name": profile.machine_name,
        "binding_subsystem": profile.binding_subsystem,
        "subsystems": {name: asdict(fit) for name, fit in profile.subsystems.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_machine_profile(path: Path) -> MachineProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    subsystems = {
        name: SubsystemFit(**fit_data) for name, fit_data in data["subsystems"].items()
    }
    return MachineProfile(
        machine_name=data["machine_name"],
        binding_subsystem=data["binding_subsystem"],
        subsystems=subsystems,
    )


@dataclass(frozen=True)
class CapacityVerdict:
    fits: bool
    pixel_rate: float
    predicted_cost: dict[str, float]
    headroom: dict[str, float]  # budget - predicted_cost, per subsystem
    limiting_subsystem: str | None  # smallest headroom (may be negative)


def evaluate_capacity(
    profile: MachineProfile, cameras: int, resolution_name: str, fps: float
) -> CapacityVerdict:
    """Does `cameras` streams at `resolution_name`/`fps` fit on `profile`'s
    machine? `fps` is a required input (THREATS-TO-VALIDITY.md T9) — this
    function never infers a required fps from content or scene dynamics
    (PRIOR-ART.md §4.2).
    """
    if cameras <= 0:
        raise ValueError("cameras must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not profile.subsystems:
        raise ValueError(f"machine profile {profile.machine_name!r} has no subsystem fits")

    resolution = resolve_resolution(resolution_name)
    pixel_rate = resolution.width * resolution.height * fps

    predicted_cost: dict[str, float] = {}
    headroom: dict[str, float] = {}
    for name, fit in profile.subsystems.items():
        cost = fit.a * pixel_rate + fit.b * cameras
        predicted_cost[name] = cost
        headroom[name] = fit.budget - cost

    limiting_subsystem = min(headroom, key=lambda k: headroom[k])
    fits = all(h >= 0 for h in headroom.values())

    return CapacityVerdict(
        fits=fits,
        pixel_rate=pixel_rate,
        predicted_cost=predicted_cost,
        headroom=headroom,
        limiting_subsystem=limiting_subsystem,
    )


@dataclass(frozen=True)
class CrossMachinePrediction:
    valid: bool
    assumed_binding_subsystem: str
    reason: str
    predicted_cost: float | None


def predict_cross_machine(
    source_profile: MachineProfile,
    target_binding_subsystem: str,
    cameras: int,
    resolution_name: str,
    fps: float,
) -> CrossMachinePrediction:
    """THREATS-TO-VALIDITY.md T7: "Every prediction states the assumed binding
    subsystem. Predictions across a regime change are labelled invalid and are
    not made." `source_profile.binding_subsystem` is the subsystem the source
    machine's own sweep found to be the constraint; if the target machine's
    binding subsystem differs, no prediction is computed — `predicted_cost` stays
    `None` and `valid` is `False`.
    """
    if cameras <= 0:
        raise ValueError("cameras must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    assumed = source_profile.binding_subsystem
    if assumed != target_binding_subsystem:
        return CrossMachinePrediction(
            valid=False,
            assumed_binding_subsystem=assumed,
            reason=(
                f"source profile {source_profile.machine_name!r} assumes "
                f"{assumed!r} is binding, but the target machine's binding "
                f"subsystem is {target_binding_subsystem!r} — "
                "THREATS-TO-VALIDITY.md T7: cross-machine prediction is only "
                "valid within the same bottleneck regime"
            ),
            predicted_cost=None,
        )

    if assumed not in source_profile.subsystems:
        raise ValueError(
            f"source profile {source_profile.machine_name!r} has no fit for its "
            f"own binding subsystem {assumed!r}"
        )

    resolution = resolve_resolution(resolution_name)
    pixel_rate = resolution.width * resolution.height * fps
    fit = source_profile.subsystems[assumed]
    predicted = fit.a * pixel_rate + fit.b * cameras

    return CrossMachinePrediction(
        valid=True,
        assumed_binding_subsystem=assumed,
        reason="same bottleneck regime assumed on both machines",
        predicted_cost=predicted,
    )
