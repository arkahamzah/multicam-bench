"""Renders the marked test video used as the publisher's source."""

from __future__ import annotations

import subprocess
from pathlib import Path

from multicam_bench.rig.marker import build_filter


def generate_test_video(
    output: Path, width: int, height: int, fps: int, duration: float
) -> Path:
    """Run ffmpeg with the marker filter, writing an libx264 test clip to `output`."""
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_str = build_filter(width, height, fps, duration)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        filter_str,
        "-frames:v",
        str(round(fps * duration)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output
