from __future__ import annotations

from pathlib import Path

import pytest

from multicam_bench.model.calculator import run_calc
from multicam_bench.model.machine_profile import (
    MachineProfile,
    SubsystemFit,
    save_machine_profile,
)


def _write_profile(path: Path) -> MachineProfile:
    profile = MachineProfile(
        machine_name="test-laptop",
        binding_subsystem="cpu",
        subsystems={
            "cpu": SubsystemFit(
                name="cpu", a=0.0001, b=5.0, budget=800.0,
                content_name="360p/libx264/30fps synthetic", machine_name="test-laptop",
            ),
        },
    )
    save_machine_profile(profile, path)
    return profile


def test_run_calc_missing_profile_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        run_calc(tmp_path / "does_not_exist.json", cameras=4, resolution="360p", fps=5.0)


def test_run_calc_reports_fit_and_limiting_subsystem(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _write_profile(path)

    output = run_calc(path, cameras=4, resolution="360p", fps=5.0)

    assert "test-laptop" in output
    assert "fits: yes" in output
    assert "limiting subsystem: cpu" in output
    assert "4 cameras @ 360p @ 5.0 fps" in output


def test_run_calc_reports_no_fit_when_over_budget(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _write_profile(path)

    output = run_calc(path, cameras=100000, resolution="1080p", fps=25.0)

    assert "fits: no" in output
