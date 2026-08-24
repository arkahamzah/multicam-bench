"""Renders the marked test video used as the publisher's source."""

from __future__ import annotations

import subprocess
from pathlib import Path

from multicam_bench.rig.marker import build_filter
from multicam_bench.rig.resolution import validate_codec


def generate_test_video(
    output: Path,
    width: int,
    height: int,
    fps: int,
    duration: float,
    codec: str = "libx264",
) -> Path:
    """Run ffmpeg with the marker filter, writing a `codec`-encoded test clip to
    `output`. `codec` must be one of `rig.resolution.CODECS` (libx264/libx265) —
    Ultralytics/DeepStream are never involved here, this is plain ffmpeg encoding.
    """
    validate_codec(codec)
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
        codec,
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output
