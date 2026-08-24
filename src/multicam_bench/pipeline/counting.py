"""Vehicle counting via line crossing (THREATS-TO-VALIDITY.md §1.3): the primary
accuracy metric is vehicle count error on a fixed counting line, not mAP —
counting is what the application produces, mAP is only a proxy for it.

Counting needs to know when the SAME object crosses, so this module includes a
minimal nearest-centroid association between consecutive frames' detections —
deliberately not ByteTrack (tracking is a later PROJECT-CHARTER-v2.md milestone);
its only job here is to avoid double-counting an object that is visible over
several frames while it crosses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class CountingLine:
    a: Point
    b: Point


@dataclass(frozen=True)
class Detection:
    cx: float
    cy: float
    class_name: str
    score: float


def side_of_line(point: Point, line: CountingLine) -> float:
    """Signed area of the triangle (line.a, line.b, point): positive on one side,
    negative on the other, zero exactly on the line.
    """
    ax, ay = line.a
    bx, by = line.b
    px, py = point
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def associate_nearest(
    prev: Sequence[Detection], curr: Sequence[Detection], max_distance: float
) -> list[tuple[int, int]]:
    """Greedy nearest-centroid matching between two frames' detections, same class
    only, each detection used at most once. Returns (prev_index, curr_index) pairs.
    """
    if max_distance < 0:
        raise ValueError("max_distance must not be negative")

    candidates = []
    for i, p in enumerate(prev):
        for j, c in enumerate(curr):
            if p.class_name != c.class_name:
                continue
            d = _distance((p.cx, p.cy), (c.cx, c.cy))
            if d <= max_distance:
                candidates.append((d, i, j))
    candidates.sort(key=lambda t: t[0])

    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _d, i, j in candidates:
        if i in matched_prev or j in matched_curr:
            continue
        matched_prev.add(i)
        matched_curr.add(j)
        pairs.append((i, j))
    return pairs


def count_crossings(
    frames: Sequence[Sequence[Detection]], line: CountingLine, max_match_distance: float
) -> int:
    """Total number of line crossings across a sequence of per-frame detections.

    A crossing is counted when an object associated across two consecutive frames
    has its centroid's side-of-line sign flip. Detections that appear only once
    (no association in the next frame) never count — that mirrors a real object
    entering or leaving the scene without crossing the line.
    """
    total = 0
    prev = frames[0] if frames else []
    for frame in frames[1:]:
        pairs = associate_nearest(prev, frame, max_match_distance)
        for i, j in pairs:
            prev_side = side_of_line((prev[i].cx, prev[i].cy), line)
            curr_side = side_of_line((frame[j].cx, frame[j].cy), line)
            if prev_side == 0.0 or curr_side == 0.0:
                continue
            if (prev_side > 0) != (curr_side > 0):
                total += 1
        prev = frame
    return total


def count_retention(golden_count: int, candidate_count: int) -> float:
    """Fraction of the golden (pseudo-ground-truth) count retained by a cheaper
    configuration, per PRIOR-ART.md §4.1 — reported as retention, never as an
    absolute accuracy figure (THREATS-TO-VALIDITY.md T10). Clamped to [0, 1]: a
    candidate that overshoots the golden count retains no *more* than all of it.
    """
    if golden_count < 0 or candidate_count < 0:
        raise ValueError("counts must not be negative")
    if golden_count == 0:
        return 1.0 if candidate_count == 0 else 0.0
    retained = 1.0 - abs(candidate_count - golden_count) / golden_count
    return max(0.0, min(1.0, retained))


def intersecting_frame_indices(
    golden_indices: Sequence[int], candidate_indices: Sequence[int]
) -> set[int]:
    """Frame indices present in both configurations' sampled sets.

    THREATS-TO-VALIDITY.md §1.3: "Cross-fps comparison is performed on the
    intersecting frame set only, or on the downstream count, never frame-by-frame
    across differing sample rates." This repo's reported accuracy metric
    (`count_retention`) uses the downstream-count branch — golden and candidate are
    compared by their total line-crossing count over the whole window, which needs
    no frame alignment since a lower detect_fps changing which frames are sampled
    is the entire point of the axis. This helper exists for any future metric that
    *does* need frame-level correspondence (e.g. comparing raw detections directly)
    so that comparison is never made across differing sample rates by accident.
    """
    return set(golden_indices) & set(candidate_indices)
