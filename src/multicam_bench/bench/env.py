"""Collects `env.json`: the machine and tool-version record every run must carry.

CLAUDE.md: "A number without its machine is noise." Best-effort only — a field that
cannot be determined on this platform is recorded as `None` rather than raising.

This file is published (CLAUDE.md hard rule 2): it must record hardware capability
(core counts, CPU model, GPU list, RAM, tool versions) but never anything that
identifies *this specific machine or person* — no hostname (`platform.node()` /
`socket.gethostname()`), no username (`getpass.getuser()` / `os.getlogin()`), and
no absolute filesystem path. The one machine-identifying field this module writes
is `machine_label`, and it is always an explicit caller-supplied string (see
`--machine-label` on the CLI) — never auto-detected from the OS.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path
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


def _cpu_model(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str | None:
    """`platform.processor()` returns the real CPU string on Windows but is a
    well-known blank string on most Linux distributions — read `/proc/cpuinfo`'s
    `model name` field there instead so `cpu_model` is never silently empty.
    `cpuinfo_path` is injectable for tests; real callers never pass it.
    """
    processor = platform.processor()
    if processor:
        return processor

    if cpuinfo_path.is_file():
        try:
            text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                value = value.strip()
                if value:
                    return value
    return None


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
            names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
            if names:
                return names
        except OSError:
            pass
    elif platform.system() == "Linux":
        lspci = shutil.which("lspci")
        if lspci is not None:
            try:
                out = subprocess.run(
                    [lspci], capture_output=True, text=True, timeout=10, check=False
                )
                names = [
                    re.sub(r"^.*?:\s*", "", line.strip())
                    for line in out.stdout.splitlines()
                    if re.search(r"\b(VGA|3D controller|Display controller)\b", line)
                ]
                if names:
                    return names
            except OSError:
                pass

    return []


def collect_env(machine_label: str = "unlabelled-machine") -> dict[str, Any]:
    """Build the env.json payload: machine label, CPU, cores, RAM, GPU list, OS
    build, tool versions. `machine_label` is always explicit — see the module
    docstring for why it is never auto-detected.
    """
    mem = psutil.virtual_memory()
    return {
        "machine_label": machine_label,
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
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
