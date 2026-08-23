"""Process wrappers for the RTSP rig: the mediamtx server and the ffmpeg publisher."""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def mediamtx_server(config_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Run mediamtx as the RTSP server for the duration of the `with` block."""
    proc = subprocess.Popen(
        ["mediamtx", str(config_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.0)  # let the RTSP listener come up before anything connects
        yield proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@contextlib.contextmanager
def publisher(video_path: Path, rtsp_url: str) -> Iterator[subprocess.Popen[bytes]]:
    """Loop `video_path` into `rtsp_url` at its native pace: ffmpeg -re -stream_loop -1."""
    cmd = [
        "ffmpeg",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        str(video_path),
        "-c",
        "copy",
        "-f",
        "rtsp",
        rtsp_url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)
