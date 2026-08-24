from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from multicam_bench.rig.ffmpeg_decode_bench import (
    build_ffmpeg_decode_bench_command,
    parse_ffmpeg_benchmark_output,
    run_ffmpeg_decode_benchmark,
)

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)

REALISTIC_STDERR = """\
ffmpeg version N-126247-gb79d4c4c0a-20260823 Copyright (c) 2000-2026 the FFmpeg developers
  built with gcc 13
Input #0, mov,mp4,m2a,3gp,3g2,mj2, from 'clip.mp4':
  Duration: 00:00:60.00, start: 0.000000, bitrate: 512 kb/s
Output #0, null, to 'pipe:':
frame=  600 fps=0.0 q=-0.0 size=N/A time=00:00:20.00 bitrate=N/A speed=  40x
frame= 1200 fps=239 q=-0.0 size=N/A time=00:00:40.00 bitrate=N/A speed=  40x
frame= 1800 fps=245 q=-0.0 Lsize=N/A time=00:01:00.00 bitrate=N/A speed=8.17x
video:0kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
bench: utime=6.812s stime=0.395s rtime=7.325s
bench: maxrss=123456kB
"""


def test_parse_extracts_final_frame_and_fps() -> None:
    parsed = parse_ffmpeg_benchmark_output(REALISTIC_STDERR)
    assert parsed.frame_count == 1800
    assert parsed.fps == pytest.approx(245.0)


def test_parse_extracts_bench_timings() -> None:
    parsed = parse_ffmpeg_benchmark_output(REALISTIC_STDERR)
    assert parsed.utime_s == pytest.approx(6.812)
    assert parsed.stime_s == pytest.approx(0.395)
    assert parsed.rtime_s == pytest.approx(7.325)


def test_parse_missing_bench_line_returns_none_fields() -> None:
    parsed = parse_ffmpeg_benchmark_output("frame= 100 fps=30 q=-0.0 time=00:00:03.33\n")
    assert parsed.frame_count == 100
    assert parsed.fps == pytest.approx(30.0)
    assert parsed.utime_s is None
    assert parsed.stime_s is None
    assert parsed.rtime_s is None


def test_parse_empty_output_returns_all_none() -> None:
    parsed = parse_ffmpeg_benchmark_output("")
    assert parsed.frame_count is None
    assert parsed.fps is None
    assert parsed.rtime_s is None


def test_parse_error_output_has_no_frame_line() -> None:
    stderr = "clip.mp4: No such file or directory\n"
    parsed = parse_ffmpeg_benchmark_output(stderr)
    assert parsed.frame_count is None
    assert parsed.fps is None


def test_build_command_omits_hwaccel_flag_for_none() -> None:
    cmd = build_ffmpeg_decode_bench_command(Path("clip.mp4"), "none")
    assert "-hwaccel" not in cmd
    assert cmd[0] == "ffmpeg"
    assert cmd[-4:] == ["-benchmark", "-f", "null", "-"]


def test_build_command_includes_hwaccel_flag_for_cuda() -> None:
    cmd = build_ffmpeg_decode_bench_command(Path("clip.mp4"), "cuda")
    assert "-hwaccel" in cmd
    idx = cmd.index("-hwaccel")
    assert cmd[idx + 1] == "cuda"


def test_build_command_rejects_unknown_hwaccel() -> None:
    with pytest.raises(ValueError, match="unknown hwaccel"):
        build_ffmpeg_decode_bench_command(Path("clip.mp4"), "made-up")


@requires_ffmpeg
def test_run_ffmpeg_decode_benchmark_on_real_tiny_clip(tmp_path: Path) -> None:
    # A ~1s, 64x64 clip: fast enough to run as a unit test, real enough to prove
    # the parser matches this ffmpeg build's actual -benchmark output format.
    clip = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
            "-frames:v", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True, capture_output=True,
    )

    result = run_ffmpeg_decode_benchmark(clip, hwaccel="none", timeout_s=30.0)

    assert result.succeeded is True
    assert result.frame_count == 10
    assert result.fps is not None and result.fps > 0
    assert result.rtime_s is not None and result.rtime_s > 0
    assert result.error is None


@requires_ffmpeg
def test_run_ffmpeg_decode_benchmark_reports_failure_for_missing_file(tmp_path: Path) -> None:
    result = run_ffmpeg_decode_benchmark(
        tmp_path / "does_not_exist.mp4", hwaccel="none", timeout_s=30.0
    )
    assert result.succeeded is False
    assert result.returncode != 0
    assert result.error is not None
