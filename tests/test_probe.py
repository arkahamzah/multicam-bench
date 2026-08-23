"""Guards the single-source-of-truth fix: fps must come from probing the actual
video file, never from a caller-supplied or hardcoded value that could drift from
what the generator actually encoded.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path

import pytest

from multicam_bench.cli import measure
from multicam_bench.rig.probe import probe_video

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@requires_ffmpeg
def test_probe_video_reads_fps_and_frame_count_from_file(tmp_path: Path) -> None:
    # Deliberately not 30 — the CLI's old default — so a regression to a hardcoded
    # fps would fail this assertion instead of accidentally matching.
    fps, duration = 24, 1.0
    out = tmp_path / "probe_check.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=64x64:rate={fps}:duration={duration}",
            "-frames:v",
            str(round(fps * duration)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    result = probe_video(out)

    assert result.fps == pytest.approx(24.0)
    assert result.frame_count == 24


def test_probe_video_parses_fractional_frame_rate() -> None:
    from multicam_bench.rig.probe import _parse_frame_rate

    assert _parse_frame_rate("30/1") == pytest.approx(30.0)
    assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=1e-2)
    assert _parse_frame_rate("25") == pytest.approx(25.0)


def test_measure_command_takes_no_fps_parameter() -> None:
    # fps must be probed from the video at measure time, never accepted as an
    # input — an fps parameter here would let the source and the reader disagree.
    sig = inspect.signature(measure)
    assert "fps" not in sig.parameters
