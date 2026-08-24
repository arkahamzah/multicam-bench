from __future__ import annotations

from pathlib import Path

import pytest

from multicam_bench.model.machine_profile import (
    MachineProfile,
    SubsystemFit,
    evaluate_capacity,
    load_machine_profile,
    predict_cross_machine,
    save_machine_profile,
)


def _profile() -> MachineProfile:
    cpu = SubsystemFit(
        name="cpu",
        a=0.0001,
        b=5.0,
        budget=800.0,  # e.g. 8 logical cores at 100% each
        content_name="360p/libx264/30fps synthetic",
        machine_name="test-laptop",
    )
    decode = SubsystemFit(
        name="decode",
        a=0.0002,
        b=2.0,
        budget=500.0,
        content_name="360p/libx264/30fps synthetic",
        machine_name="test-laptop",
    )
    return MachineProfile(
        machine_name="test-laptop",
        binding_subsystem="cpu",
        subsystems={"cpu": cpu, "decode": decode},
    )


def test_evaluate_capacity_fits_when_under_every_budget() -> None:
    verdict = evaluate_capacity(_profile(), cameras=2, resolution_name="360p", fps=5.0)
    assert verdict.fits is True
    assert verdict.pixel_rate == pytest.approx(640 * 360 * 5.0)


def test_evaluate_capacity_does_not_fit_when_over_budget() -> None:
    verdict = evaluate_capacity(_profile(), cameras=200, resolution_name="1080p", fps=25.0)
    assert verdict.fits is False


def test_evaluate_capacity_reports_limiting_subsystem() -> None:
    # decode has a much smaller budget (500) than cpu (800) for the same load —
    # it should run out of headroom first.
    verdict = evaluate_capacity(_profile(), cameras=50, resolution_name="720p", fps=10.0)
    assert verdict.limiting_subsystem == "decode"
    assert verdict.headroom["decode"] < verdict.headroom["cpu"]


def test_evaluate_capacity_headroom_matches_budget_minus_predicted_cost() -> None:
    verdict = evaluate_capacity(_profile(), cameras=4, resolution_name="360p", fps=10.0)
    for name in verdict.predicted_cost:
        assert verdict.headroom[name] == pytest.approx(
            _profile().subsystems[name].budget - verdict.predicted_cost[name]
        )


def test_evaluate_capacity_rejects_non_positive_cameras() -> None:
    with pytest.raises(ValueError, match="cameras"):
        evaluate_capacity(_profile(), cameras=0, resolution_name="360p", fps=5.0)


def test_evaluate_capacity_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        evaluate_capacity(_profile(), cameras=1, resolution_name="360p", fps=0.0)


def test_evaluate_capacity_rejects_empty_profile() -> None:
    empty = MachineProfile(machine_name="empty", binding_subsystem="cpu", subsystems={})
    with pytest.raises(ValueError, match="no subsystem fits"):
        evaluate_capacity(empty, cameras=1, resolution_name="360p", fps=5.0)


def test_save_and_load_machine_profile_roundtrip(tmp_path: Path) -> None:
    profile = _profile()
    path = tmp_path / "test-laptop.json"
    save_machine_profile(profile, path)
    loaded = load_machine_profile(path)
    assert loaded == profile


def test_predict_cross_machine_valid_when_binding_subsystem_matches() -> None:
    prediction = predict_cross_machine(
        _profile(), target_binding_subsystem="cpu", cameras=4, resolution_name="360p", fps=5.0
    )
    assert prediction.valid is True
    assert prediction.predicted_cost is not None
    assert prediction.assumed_binding_subsystem == "cpu"


def test_predict_cross_machine_invalid_when_binding_subsystem_differs() -> None:
    # THREATS-TO-VALIDITY.md T7: a CPU-bound laptop tells you little about a
    # GPU/decode-bound server — the prediction must be refused, not made.
    prediction = predict_cross_machine(
        _profile(),
        target_binding_subsystem="decode",
        cameras=4,
        resolution_name="360p",
        fps=5.0,
    )
    assert prediction.valid is False
    assert prediction.predicted_cost is None
    assert "T7" in prediction.reason


def test_predict_cross_machine_rejects_non_positive_cameras() -> None:
    with pytest.raises(ValueError, match="cameras"):
        predict_cross_machine(
            _profile(), target_binding_subsystem="cpu", cameras=0,
            resolution_name="360p", fps=5.0,
        )


def test_predict_cross_machine_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        predict_cross_machine(
            _profile(), target_binding_subsystem="cpu", cameras=1,
            resolution_name="360p", fps=0.0,
        )
