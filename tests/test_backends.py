from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from multicam_bench.rig.backends import (
    available_backend_names,
    detect_backends,
    hw_acceleration_for,
)


@dataclass
class FakeProbe:
    ffmpeg: bool = True
    system: str = "Windows"
    gpus: Sequence[str] = field(default_factory=lambda: ["Intel Iris Xe Graphics"])
    cv2_attrs: set[str] = field(
        default_factory=lambda: {"VIDEO_ACCELERATION_D3D11", "VIDEO_ACCELERATION_MFX"}
    )
    cuda_devices: int = 0

    def ffmpeg_on_path(self) -> bool:
        return self.ffmpeg

    def platform_system(self) -> str:
        return self.system

    def gpu_names(self) -> Sequence[str]:
        return self.gpus

    def cv2_has_attr(self, name: str) -> bool:
        return name in self.cv2_attrs

    def cuda_device_count(self) -> int:
        return self.cuda_devices


def test_this_laptop_profile_matches_expectations() -> None:
    # Iris Xe + MX550 + opencv-python-headless (no CUDA codec support): the exact
    # profile of the dev machine this repo targets.
    probe = FakeProbe(gpus=["Intel Iris Xe Graphics", "NVIDIA GeForce MX550"], cuda_devices=0)
    results = {b.name: b for b in detect_backends(probe)}

    assert results["ffmpeg-cpu"].available is True
    assert results["d3d11va"].available is True
    assert results["qsv"].available is True
    assert results["cuda"].available is False
    assert "CUDA" in results["cuda"].reason


def test_ffmpeg_missing_is_unavailable_with_reason() -> None:
    probe = FakeProbe(ffmpeg=False)
    results = {b.name: b for b in detect_backends(probe)}
    assert results["ffmpeg-cpu"].available is False
    assert "PATH" in results["ffmpeg-cpu"].reason


def test_d3d11va_unavailable_off_windows() -> None:
    probe = FakeProbe(system="Linux")
    results = {b.name: b for b in detect_backends(probe)}
    assert results["d3d11va"].available is False
    assert "Windows" in results["d3d11va"].reason


def test_qsv_unavailable_without_intel_gpu() -> None:
    probe = FakeProbe(gpus=["NVIDIA GeForce MX550"])
    results = {b.name: b for b in detect_backends(probe)}
    assert results["qsv"].available is False
    assert "Intel" in results["qsv"].reason


def test_qsv_unavailable_without_opencv_mfx_support() -> None:
    probe = FakeProbe(cv2_attrs={"VIDEO_ACCELERATION_D3D11"})
    results = {b.name: b for b in detect_backends(probe)}
    assert results["qsv"].available is False
    assert "opencv" in results["qsv"].reason


def test_cuda_available_with_gpu_and_device_count() -> None:
    probe = FakeProbe(gpus=["NVIDIA GeForce MX550"], cuda_devices=1)
    results = {b.name: b for b in detect_backends(probe)}
    assert results["cuda"].available is True
    assert results["cuda"].hw_acceleration is not None


def test_every_backend_name_gets_a_result_never_silently_omitted() -> None:
    probe = FakeProbe(ffmpeg=False, system="Linux", gpus=[], cv2_attrs=set())
    results = detect_backends(probe)
    assert {b.name for b in results} == {"ffmpeg-cpu", "d3d11va", "qsv", "cuda"}
    assert all(not b.available for b in results)
    assert all(b.reason for b in results)


def test_available_backend_names_filters_to_available_only() -> None:
    probe = FakeProbe(gpus=["Intel Iris Xe Graphics"], cuda_devices=0)
    names = available_backend_names(probe, ["ffmpeg-cpu", "d3d11va", "qsv", "cuda"])
    assert names == ["ffmpeg-cpu", "d3d11va", "qsv"]


def test_available_backend_names_preserves_requested_order() -> None:
    probe = FakeProbe(gpus=["Intel Iris Xe Graphics"])
    names = available_backend_names(probe, ["qsv", "ffmpeg-cpu"])
    assert names == ["qsv", "ffmpeg-cpu"]


def test_available_backend_names_rejects_unknown_backend() -> None:
    probe = FakeProbe()
    with pytest.raises(ValueError, match="unknown backend"):
        available_backend_names(probe, ["vaapi"])


def test_hw_acceleration_for_ffmpeg_cpu_is_none() -> None:
    probe = FakeProbe()
    backends = detect_backends(probe)
    assert hw_acceleration_for("ffmpeg-cpu", backends) is None


def test_hw_acceleration_for_available_backend_returns_flag() -> None:
    probe = FakeProbe(gpus=["Intel Iris Xe Graphics"])
    backends = detect_backends(probe)
    assert hw_acceleration_for("d3d11va", backends) is not None


def test_hw_acceleration_for_unavailable_backend_raises() -> None:
    probe = FakeProbe(gpus=[])
    backends = detect_backends(probe)
    with pytest.raises(ValueError, match="not available"):
        hw_acceleration_for("qsv", backends)


def test_hw_acceleration_for_unknown_name_raises() -> None:
    probe = FakeProbe()
    backends = detect_backends(probe)
    with pytest.raises(ValueError, match="unknown backend"):
        hw_acceleration_for("vaapi", backends)
