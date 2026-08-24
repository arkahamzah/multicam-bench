"""Sweep plan construction: which (N, repetition, mode) points to run and in what
order. Pure logic, no subprocesses — kept separate from `sweep.py`'s process
orchestration so the ordering/randomisation behaviour is unit-testable without
ffmpeg or mediamtx.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepPoint:
    n_streams: int
    repetition: int
    order_index: int
    publisher_only: bool

    @property
    def mode(self) -> str:
        return "baseline" if self.publisher_only else "measured"

    def dir_name(self) -> str:
        return f"N{self.n_streams}_rep{self.repetition}_{self.mode}"


def randomize_order(
    values: Sequence[int], repetitions: int, rng: random.Random
) -> list[list[int]]:
    """Return `repetitions` independently shuffled orderings of `values`.

    THREATS-TO-VALIDITY.md T2: configuration order is randomised per repetition so
    ascending-N thermal drift does not confound N with run order.
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
    n_values: Sequence[int], repetitions: int, rng: random.Random
) -> list[SweepPoint]:
    """Full run plan: per repetition, a randomised N order; per N, a publisher-only
    baseline immediately followed by the measured run (THREATS-TO-VALIDITY.md T5).
    """
    orders = randomize_order(n_values, repetitions, rng)
    plan: list[SweepPoint] = []
    for rep, order in enumerate(orders):
        for order_index, n in enumerate(order):
            plan.append(
                SweepPoint(
                    n_streams=n, repetition=rep, order_index=order_index, publisher_only=True
                )
            )
            plan.append(
                SweepPoint(
                    n_streams=n, repetition=rep, order_index=order_index, publisher_only=False
                )
            )
    return plan


def point_dir(sweep_dir: Path, point: SweepPoint) -> Path:
    return sweep_dir / "points" / point.dir_name()


def camera_dir(point_dir_path: Path, camera_id: int) -> Path:
    return point_dir_path / f"cam{camera_id}"
