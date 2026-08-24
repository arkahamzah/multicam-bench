"""Decode backend detection — v0.4 sweep axis (PROJECT-CHARTER-v2.md §2: "backend
decode" is a sumbu, not an assumption).

Backends: `ffmpeg-cpu` (software), `d3d11va` (Windows Media Foundation / DXVA2),
`qsv` (Intel Quick Sync, e.g. Iris Xe), `cuda` (NVDEC), `vaapi` (Linux Video
Acceleration API — the standard Intel/AMD hardware decode path on Linux, and
`d3d11va`'s counterpart there). Availability is best-effort and hardware/library
based — actually opening a hardware-accelerated capture is what the sweep itself
does; this module only decides which backends are worth trying and records *why*
the others were skipped, per CLAUDE.md hard rule 1 ("no code for anything that
cannot run on this machine"). In particular, `opencv-python-headless` (this
repo's pinned OpenCV wheel) ships with no CUDA codec support, so `cuda` reliably
reports unavailable on a machine installed from that wheel even when an NVIDIA
GPU is physically present — that is not a bug, it is the wheel's real capability
being reported honestly. See `rig/ffmpeg_decode_bench.py` for the separate,
ffmpeg-subprocess-based decode path that does not go through cv2 at all and can
actually reach NVDEC/QSV/VAAPI on a machine where they work.
"""

from __future__ import annotations

import platform
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

BACKEND_NAMES = ("ffmpeg-cpu", "d3d11va", "qsv", "cuda", "vaapi")


@dataclass(frozen=True)
class BackendAvailability:
    name: str
    available: bool
    reason: str
    hw_acceleration: int | None  # a cv2.VIDEO_ACCELERATION_* constant, or None


class BackendProbe(Protocol):
    """What backend detection needs to know about the host. A real implementation
    touches ffmpeg/cv2/GPU state; tests supply a fake for deterministic behaviour.
    """

    def ffmpeg_on_path(self) -> bool: ...
    def platform_system(self) -> str: ...
    def gpu_names(self) -> Sequence[str]: ...
    def cv2_has_attr(self, name: str) -> bool: ...
    def cuda_device_count(self) -> int: ...
    def has_vaapi_device(self) -> bool: ...


class SystemBackendProbe:
    """The real, hardware-touching probe used at sweep time."""

    def ffmpeg_on_path(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def platform_system(self) -> str:
        return platform.system()

    def gpu_names(self) -> Sequence[str]:
        from multicam_bench.bench.env import _gpu_list

        return _gpu_list()

    def cv2_has_attr(self, name: str) -> bool:
        return hasattr(cv2, name)

    def cuda_device_count(self) -> int:
        try:
            return int(cv2.cuda.getCudaEnabledDeviceCount())
        except (AttributeError, cv2.error):
            return 0

    def has_vaapi_device(self) -> bool:
        dri = Path("/dev/dri")
        if not dri.is_dir():
            return False
        return any(p.name.startswith("renderD") for p in dri.iterdir())


def detect_backends(probe: BackendProbe) -> list[BackendAvailability]:
    """One `BackendAvailability` per `BACKEND_NAMES`, available or not, always with
    a reason — never silently omitted, so a skipped backend shows up in results.
    """
    results: list[BackendAvailability] = []

    if probe.ffmpeg_on_path():
        results.append(BackendAvailability("ffmpeg-cpu", True, "ffmpeg found on PATH", None))
    else:
        results.append(
            BackendAvailability("ffmpeg-cpu", False, "ffmpeg not found on PATH", None)
        )

    if probe.platform_system() != "Windows":
        results.append(
            BackendAvailability(
                "d3d11va", False, f"not Windows (platform={probe.platform_system()})", None
            )
        )
    elif not probe.cv2_has_attr("VIDEO_ACCELERATION_D3D11"):
        results.append(
            BackendAvailability(
                "d3d11va", False, "installed opencv build lacks VIDEO_ACCELERATION_D3D11", None
            )
        )
    else:
        results.append(
            BackendAvailability(
                "d3d11va",
                True,
                "Windows + opencv build supports VIDEO_ACCELERATION_D3D11",
                cv2.VIDEO_ACCELERATION_D3D11,
            )
        )

    gpu_text = " ".join(probe.gpu_names()).lower()
    has_intel_gpu = "intel" in gpu_text or "iris" in gpu_text
    if not has_intel_gpu:
        results.append(BackendAvailability("qsv", False, "no Intel GPU detected", None))
    elif not probe.cv2_has_attr("VIDEO_ACCELERATION_MFX"):
        results.append(
            BackendAvailability(
                "qsv", False, "installed opencv build lacks VIDEO_ACCELERATION_MFX", None
            )
        )
    else:
        results.append(
            BackendAvailability(
                "qsv",
                True,
                "Intel GPU detected + opencv build supports VIDEO_ACCELERATION_MFX",
                cv2.VIDEO_ACCELERATION_MFX,
            )
        )

    has_nvidia_gpu = "nvidia" in gpu_text
    cuda_devices = probe.cuda_device_count()
    if not has_nvidia_gpu:
        results.append(BackendAvailability("cuda", False, "no NVIDIA GPU detected", None))
    elif cuda_devices == 0:
        results.append(
            BackendAvailability(
                "cuda",
                False,
                "NVIDIA GPU present but opencv build reports 0 CUDA-enabled devices "
                "(opencv-python-headless ships without CUDA codec support)",
                None,
            )
        )
    else:
        results.append(
            BackendAvailability(
                "cuda",
                True,
                f"NVIDIA GPU detected + {cuda_devices} CUDA-enabled device(s) "
                "in this opencv build",
                cv2.VIDEO_ACCELERATION_ANY,
            )
        )

    if probe.platform_system() != "Linux":
        results.append(
            BackendAvailability(
                "vaapi", False, f"not Linux (platform={probe.platform_system()})", None
            )
        )
    elif not probe.cv2_has_attr("VIDEO_ACCELERATION_VAAPI"):
        results.append(
            BackendAvailability(
                "vaapi", False, "installed opencv build lacks VIDEO_ACCELERATION_VAAPI", None
            )
        )
    elif not probe.has_vaapi_device():
        results.append(
            BackendAvailability("vaapi", False, "no /dev/dri/renderD* device found", None)
        )
    else:
        results.append(
            BackendAvailability(
                "vaapi",
                True,
                "Linux + a /dev/dri render device + opencv build supports "
                "VIDEO_ACCELERATION_VAAPI",
                cv2.VIDEO_ACCELERATION_VAAPI,
            )
        )

    return results


def available_backend_names(probe: BackendProbe, requested: Sequence[str]) -> list[str]:
    """Of `requested` backend names, the ones `detect_backends` found available —
    order preserved, unknown names rejected loudly rather than silently dropped.
    """
    known = {b.name for b in detect_backends(probe)}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"unknown backend(s) {unknown}; choose from {BACKEND_NAMES}")

    availability = {b.name: b.available for b in detect_backends(probe)}
    return [name for name in requested if availability[name]]


def hw_acceleration_for(name: str, backends: list[BackendAvailability]) -> int | None:
    """The `cv2.VIDEO_ACCELERATION_*` flag for an available backend by name."""
    for b in backends:
        if b.name == name:
            if not b.available:
                raise ValueError(f"backend {name!r} is not available: {b.reason}")
            return b.hw_acceleration
    raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")
