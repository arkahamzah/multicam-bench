"""Probes the marked test video for fps and frame count via ffprobe.

This is the single source of truth for fps at measurement time. `measure` never
takes fps as an input — the generator's fps and the reader's fps are two separate
values that can drift apart if either is hardcoded or passed by hand, and a drift
shows up as ingest_lag growing linearly, indistinguishable from real saturation.
Reading it straight from the file both encoder and reader agree on eliminates that
class of bug.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoProbe:
    fps: float
    frame_count: int


def _parse_frame_rate(raw: str) -> float:
    """Parse ffprobe's r_frame_rate ("30/1", "30000/1001", ...) into a float."""
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(num) / float(den)
    return float(raw)


def probe_video(video: Path) -> VideoProbe:
    """Read fps and total frame count from `video` via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_read_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    frame_rate_raw, frame_count_raw = out.stdout.strip().split(",")
    return VideoProbe(
        fps=_parse_frame_rate(frame_rate_raw), frame_count=int(frame_count_raw)
    )
