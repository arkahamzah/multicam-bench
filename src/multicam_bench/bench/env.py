"""Collects `env.json`: the machine and tool-version record every run must carry.

CLAUDE.md: "A number without its machine is noise." Best-effort only — a field that
cannot be determined on this platform is recorded as `None` rather than raising.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

import cv2
import numpy
import pandas
import psutil


def _tool_version(name: str, *args: str) -> str | None:
    exe = shutil.which(name)
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=10, check=False
        )
    except OSError:
        return None
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0] if text else None


def _gpu_list() -> list[str]:
    nvidia = shutil.which("nvidia-smi")
    if nvidia is not None:
        try:
            out = subprocess.run(
                [nvidia, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
            if names:
                return names
        except OSError:
            pass
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController).Name",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
        except OSError:
            pass
    return []


def collect_env() -> dict[str, Any]:
    """Build the env.json payload: CPU, cores, RAM, GPU list, OS build, tool versions."""
    mem = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "cpu_model": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_bytes": mem.total,
        "gpus": _gpu_list(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "ffmpeg_version": _tool_version("ffmpeg", "-version"),
        "ffprobe_version": _tool_version("ffprobe", "-version"),
        "mediamtx_version": _tool_version("mediamtx", "--version"),
    }
