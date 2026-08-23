from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicam_bench.rig.marker import BITS, BLOCK, MARGIN, PITCH, build_filter, read_index

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def test_read_index_on_synthetic_frame() -> None:
    # Bypasses ffmpeg entirely: paints the marker bits directly and checks decode.
    width = BITS * PITCH + 2 * MARGIN
    height = BLOCK + 2 * MARGIN
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    index = 0b1011
    for i in range(BITS):
        if index & (1 << i):
            x = i * PITCH + MARGIN
            frame[MARGIN : MARGIN + BLOCK, x : x + BLOCK] = 255
    assert read_index(frame) == index


@requires_ffmpeg
def test_marker_roundtrip_through_encode(tmp_path: Path) -> None:
    width, height, fps, duration = 320, 240, 10, 2.0
    n_frames = round(fps * duration)
    out = tmp_path / "marker_roundtrip.mp4"

    filt = build_filter(width, height, fps, duration)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            filt,
            "-frames:v",
            str(n_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    cap = cv2.VideoCapture(str(out))
    try:
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(read_index(frame))
    finally:
        cap.release()

    assert decoded == list(range(len(decoded)))
    assert len(decoded) == n_frames
