"""RTSP consumer: measures ingest_lag and embedded-index continuity, writes samples.csv."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from multicam_bench.bench.lag import Anchor, LoopUnwrapper, ingest_lag_s
from multicam_bench.rig.marker import read_index


@dataclass(frozen=True)
class MeasureResult:
    samples_path: Path
    frames_received: int
    frames_measured: int


def measure_stream(
    rtsp_url: str,
    samples_path: Path,
    fps_source: float,
    period_frames: int,
    warmup_s: float,
    measure_s: float,
) -> MeasureResult:
    """Read `rtsp_url`, discard `warmup_s`, then record `measure_s` of samples.

    The anchor (frame_index, wall time) is set on the first frame received *after*
    warm-up elapses — never on the stream's first frame — per THREATS-TO-VALIDITY.md.
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {rtsp_url}")

    unwrapper = LoopUnwrapper(period=period_frames)
    t_start = time.perf_counter()
    anchor: Anchor | None = None
    frames_received = 0
    frames_measured = 0
    last_frame_index: int | None = None

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with samples_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame_index", "raw_index", "t_recv", "ingest_lag_s", "gap"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t_recv = time.perf_counter()
                raw_index = read_index(frame)
                frame_index = unwrapper.unwrap(raw_index)
                frames_received += 1

                if anchor is None:
                    if t_recv - t_start >= warmup_s:
                        anchor = Anchor(frame_index=frame_index, t_wall=t_recv)
                    else:
                        continue

                if t_recv - anchor.t_wall >= measure_s:
                    break

                gap = 0 if last_frame_index is None else frame_index - last_frame_index - 1
                last_frame_index = frame_index
                lag = ingest_lag_s(anchor, frame_index, t_recv, fps_source)
                writer.writerow([frame_index, raw_index, t_recv, lag, gap])
                frames_measured += 1
    finally:
        cap.release()

    return MeasureResult(
        samples_path=samples_path,
        frames_received=frames_received,
        frames_measured=frames_measured,
    )
