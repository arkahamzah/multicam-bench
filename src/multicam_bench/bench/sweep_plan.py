"""Sweep plan construction: which (N, codec, backend, repetition, mode) points to
run and in what order. Pure logic, no subprocesses — kept separate from
`sweep.py`'s process orchestration so the ordering/randomisation behaviour is
unit-testable without ffmpeg or mediamtx.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepConfigPoint:
    """One (N, codec, backend) configuration to run, before baseline/measured
    pairing and repetition are applied. `backend` is a decode-backend name
    (v0.4 axis) — it does not affect the publisher-only baseline, which only
    reads N and codec, but is threaded through so every measured run's provenance
    is unambiguous.
    """

    n_streams: int
    codec: str
    backend: str


@dataclass(frozen=True)
class SweepPoint:
    n_streams: int
    repetition: int
    order_index: int
    publisher_only: bool
    codec: str
    backend: str

    @property
    def mode(self) -> str:
        return "baseline" if self.publisher_only else "measured"

    def dir_name(self) -> str:
        return (
            f"N{self.n_streams}_{self.codec}_{self.backend}_"
            f"rep{self.repetition}_{self.mode}"
        )


def randomize_order[T](
    values: Sequence[T], repetitions: int, rng: random.Random
) -> list[list[T]]:
    """Return `repetitions` independently shuffled orderings of `values`.

    THREATS-TO-VALIDITY.md T2: configuration order is randomised per repetition so
    ascending-N (or ascending-anything) thermal drift does not confound the swept
    configuration with run order. Works over any config type, not just N — v0.4
    shuffles the full (N, codec, backend) cross product together.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if not values:
        raise ValueError("values must not be empty")

    orders = []
    for _ in range(repetitions):
        order = list(values)
        rng.shuffle(order)
        orders.append(order)
    return orders


def build_sweep_plan(
    configs: Sequence[SweepConfigPoint], repetitions: int, rng: random.Random
) -> list[SweepPoint]:
    """Full run plan: per repetition, a randomised order over the full (N, codec,
    backend) cross product; per configuration, a publisher-only baseline
    immediately followed by the measured run (THREATS-TO-VALIDITY.md T5).
    """
    orders = randomize_order(configs, repetitions, rng)
    plan: list[SweepPoint] = []
    for rep, order in enumerate(orders):
        for order_index, cfg in enumerate(order):
            plan.append(
                SweepPoint(
                    n_streams=cfg.n_streams,
                    repetition=rep,
                    order_index=order_index,
                    publisher_only=True,
                    codec=cfg.codec,
                    backend=cfg.backend,
                )
            )
            plan.append(
                SweepPoint(
                    n_streams=cfg.n_streams,
                    repetition=rep,
                    order_index=order_index,
                    publisher_only=False,
                    codec=cfg.codec,
                    backend=cfg.backend,
                )
            )
    return plan


def cross_product_configs(
    n_values: Sequence[int], codecs: Sequence[str], backends: Sequence[str]
) -> list[SweepConfigPoint]:
    """Every (N, codec, backend) combination — `backends` should already be
    filtered to what `rig.backends.available_backend_names` reports as usable.
    """
    return [
        SweepConfigPoint(n_streams=n, codec=codec, backend=backend)
        for n in n_values
        for codec in codecs
        for backend in backends
    ]


def point_dir(sweep_dir: Path, point: SweepPoint) -> Path:
    return sweep_dir / "points" / point.dir_name()


def camera_dir(point_dir_path: Path, camera_id: int) -> Path:
    return point_dir_path / f"cam{camera_id}"
