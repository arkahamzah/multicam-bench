"""Per-camera reader for the multi-stream sweep (CLAUDE.md hard rule 8: one process
per camera, never threads for decode — so one decoder crash cannot take down its
siblings). `run_camera_reader` is the target of one `multiprocessing.Process` per
camera.

Inside that single process, decode still happens on only one thread (a capture
thread doing `cv2.VideoCapture.read()`), decoupled from measurement via a
`BoundedDropOldestQueue(maxsize=1)` (hard rule 3). This is what turns "we dropped
it" (a queue eviction, counted here as `queue_drops`) into a number distinct from
"the decoder skipped it" (an embedded-index gap — see bench/lag.py), per
THREATS-TO-VALIDITY.md T6.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

import cv2
import numpy as np

from multicam_bench.bench.lag import Anchor, LoopUnwrapper, ingest_lag_s
from multicam_bench.bench.queue_policy import BoundedDropOldestQueue
from multicam_bench.rig.marker import read_index


@dataclass(frozen=True)
class CapturedFrame:
    frame: np.ndarray
    t_recv: float


@dataclass(frozen=True)
class CameraResult:
    camera_id: int
    samples_path: Path
    frames_received: int
    frames_measured: int
    queue_drops: int


def _capture_loop(
    cap: cv2.VideoCapture,
    frame_queue: BoundedDropOldestQueue[CapturedFrame],
    stop: threading.Event,
) -> None:
    """Decode as fast as frames arrive; the queue, not this loop, absorbs backlog."""
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        frame_queue.put(CapturedFrame(frame=frame, t_recv=time.perf_counter()))


def run_camera_reader(
    camera_id: int,
    rtsp_url: str,
    samples_path: Path,
    fps_source: float,
    period_frames: int,
    warmup_s: float,
    measure_s: float,
    queue_maxsize: int = 1,
    get_timeout_s: float = 10.0,
) -> CameraResult:
    """Read `rtsp_url` for this one camera, discard `warmup_s`, then record
    `measure_s` of samples to `samples_path`. Blocks until measurement completes,
    the stream ends, or no frame arrives within `get_timeout_s`.
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"camera {camera_id}: could not open {rtsp_url}")

    frame_queue: BoundedDropOldestQueue[CapturedFrame] = BoundedDropOldestQueue(
        maxsize=queue_maxsize
    )
    stop = threading.Event()
    capture_thread = threading.Thread(
        target=_capture_loop, args=(cap, frame_queue, stop), daemon=True
    )
    capture_thread.start()

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
                try:
                    captured = frame_queue.get(timeout=get_timeout_s)
                except Empty:
                    break
                t_recv = captured.t_recv
                raw_index = read_index(captured.frame)
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
        stop.set()
        cap.release()
        capture_thread.join(timeout=5.0)

    return CameraResult(
        camera_id=camera_id,
        samples_path=samples_path,
        frames_received=frames_received,
        frames_measured=frames_measured,
        queue_drops=frame_queue.drops,
    )


def camera_process_entry(
    camera_id: int,
    rtsp_url: str,
    samples_path: Path,
    fps_source: float,
    period_frames: int,
    warmup_s: float,
    measure_s: float,
    summary_path: Path,
) -> None:
    """`multiprocessing.Process` target: run this camera's reader and write its
    result as JSON to `summary_path`, so the parent (which spawned this process
    with Windows `spawn`, not `fork`) can recover the result without a pipe.
    """
    result = run_camera_reader(
        camera_id=camera_id,
        rtsp_url=rtsp_url,
        samples_path=samples_path,
        fps_source=fps_source,
        period_frames=period_frames,
        warmup_s=warmup_s,
        measure_s=measure_s,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "camera_id": result.camera_id,
                "samples_path": str(result.samples_path),
                "frames_received": result.frames_received,
                "frames_measured": result.frames_measured,
                "queue_drops": result.queue_drops,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
