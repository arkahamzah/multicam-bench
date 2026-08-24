"""Detection-fps gate (PROJECT-CHARTER-v2.md §3 Layer 1 "[gate] opsional skip fps"):
decode always runs at the source's native fps, but the detector only needs to run
on a subsample. v0.5 makes that subsample rate a first-class axis, independent of
decode fps (1, 2, 5, 10, 25 in `configs/detect.yaml`).
"""

from __future__ import annotations


def detect_frame_indices(decode_fps: float, detect_fps: float, total_frames: int) -> list[int]:
    """Which of `total_frames` decoded frame indices (0-based) get run through the
    detector, spaced as evenly as achievable so the sampled rate is as close to
    `detect_fps` as `decode_fps` allows. Frame 0 is always included.
    """
    if decode_fps <= 0:
        raise ValueError("decode_fps must be positive")
    if detect_fps <= 0:
        raise ValueError("detect_fps must be positive")
    if detect_fps > decode_fps:
        raise ValueError(
            f"detect_fps ({detect_fps}) cannot exceed decode_fps ({decode_fps}) — "
            "the detector cannot see frames that were never decoded"
        )
    if total_frames < 0:
        raise ValueError("total_frames must not be negative")

    step = decode_fps / detect_fps
    indices = []
    next_idx = 0.0
    for i in range(total_frames):
        if i >= next_idx:
            indices.append(i)
            next_idx += step
    return indices
