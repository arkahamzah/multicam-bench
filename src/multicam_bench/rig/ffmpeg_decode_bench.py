"""Decode-only benchmark via ffmpeg subprocess — measures raw decode throughput
directly, independent of the OpenCV/`cv2.VideoCapture` pipeline used elsewhere in
this repo.

This exists because the installed OpenCV build (`opencv-python-headless`) reports
0 CUDA-enabled devices — NVDEC is unreachable through `cv2.VideoCapture` on this
wheel no matter what hardware is present (see `rig/backends.py`), and even the
`d3d11va`/`qsv`/`vaapi` paths that cv2 *does* expose either silently fall back to
software decode or fail in ways that are hard to distinguish from the pipeline's
own overhead. ffmpeg's own `-hwaccel` flag is the same acceleration path
DeepStream/NVIDIA's published numbers and most other benchmarks use, so this is
also the more externally-comparable measurement (PROJECT-CHARTER-v2.md §5).

No RTSP, no mediamtx, no queue: decodes a local file straight to `-f null -` and
parses ffmpeg's own `-benchmark` stats. This measures decode speed, not ingest
pacing — a different question from `ingest_lag`, not a replacement for it
(THREATS-TO-VALIDITY.md T11 applies here too: name what was actually measured).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

HWACCELS = ("none", "cuda", "qsv", "d3d11va", "vaapi")

# ffmpeg prints progress like "frame= 1800 fps=245 q=-0.0 ..." repeatedly to
# stderr; the LAST occurrence is the final tally.
_FRAME_RE = re.compile(r"frame=\s*(\d+)\s+fps=\s*([\d.]+)")
# ffmpeg -benchmark prints one summary line: "bench: utime=6.812s stime=0.395s rtime=7.325s"
_BENCH_RE = re.compile(r"bench:\s*utime=([\d.]+)s\s+stime=([\d.]+)s\s+rtime=([\d.]+)s")


@dataclass(frozen=True)
class ParsedBenchmark:
    frame_count: int | None
    fps: float | None
    utime_s: float | None
    stime_s: float | None
    rtime_s: float | None


@dataclass(frozen=True)
class DecodeBenchResult:
    hwaccel: str
    succeeded: bool
    frame_count: int | None
    fps: float | None
    utime_s: float | None
    stime_s: float | None
    rtime_s: float | None
    returncode: int
    error: str | None


def parse_ffmpeg_benchmark_output(stderr_text: str) -> ParsedBenchmark:
    """Parse ffmpeg's `-benchmark` stderr. Any field is `None` if its line was
    never found (e.g. ffmpeg exited before printing it, or hit an error).
    """
    frame_matches = _FRAME_RE.findall(stderr_text)
    frame_count: int | None = None
    fps: float | None = None
    if frame_matches:
        last_frame, last_fps = frame_matches[-1]
        frame_count = int(last_frame)
        fps = float(last_fps)

    bench_match = _BENCH_RE.search(stderr_text)
    utime_s = stime_s = rtime_s = None
    if bench_match:
        utime_s, stime_s, rtime_s = (float(g) for g in bench_match.groups())

    return ParsedBenchmark(
        frame_count=frame_count, fps=fps, utime_s=utime_s, stime_s=stime_s, rtime_s=rtime_s
    )


def build_ffmpeg_decode_bench_command(video_path: Path, hwaccel: str) -> list[str]:
    """`hwaccel="none"` omits the `-hwaccel` flag entirely (pure software decode)."""
    if hwaccel not in HWACCELS:
        raise ValueError(f"unknown hwaccel {hwaccel!r}; choose from {HWACCELS}")
    cmd = ["ffmpeg", "-y"]
    if hwaccel != "none":
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(video_path), "-benchmark", "-f", "null", "-"]
    return cmd


def run_ffmpeg_decode_benchmark(
    video_path: Path, hwaccel: str, timeout_s: float = 120.0
) -> DecodeBenchResult:
    """Run one `ffmpeg [-hwaccel <hwaccel>] -i <video_path> -benchmark -f null -`
    and parse its result.
    """
    cmd = build_ffmpeg_decode_bench_command(video_path, hwaccel)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return DecodeBenchResult(
            hwaccel=hwaccel,
            succeeded=False,
            frame_count=None,
            fps=None,
            utime_s=None,
            stime_s=None,
            rtime_s=None,
            returncode=-1,
            error=f"ffmpeg decode benchmark timed out after {timeout_s}s",
        )

    parsed = parse_ffmpeg_benchmark_output(proc.stderr)

    # ffmpeg's own "fps=" field is sampled periodically and reports 0.0 for a
    # clip that finishes before its first sample — real on very short clips
    # (see tests), not just a synthetic edge case. frame_count/rtime_s is the
    # more reliable throughput figure whenever both are available.
    fps = parsed.fps
    if (not fps) and parsed.frame_count and parsed.rtime_s:
        fps = parsed.frame_count / parsed.rtime_s

    succeeded = proc.returncode == 0 and fps is not None and parsed.rtime_s is not None
    error = None
    if not succeeded:
        stderr_lines = proc.stderr.strip().splitlines()
        error = stderr_lines[-1] if stderr_lines else f"ffmpeg exited {proc.returncode}"

    return DecodeBenchResult(
        hwaccel=hwaccel,
        succeeded=succeeded,
        frame_count=parsed.frame_count,
        fps=fps,
        utime_s=parsed.utime_s,
        stime_s=parsed.stime_s,
        rtime_s=parsed.rtime_s,
        returncode=proc.returncode,
        error=error,
    )
