"""Binary frame-index marker: encode/decode.

Frame index is written into the video as N white/black blocks along the top edge,
least-significant bit first. Font-free (so it behaves identically on Windows and
Linux) and decoded by sampling N pixels rather than running OCR.
"""

from __future__ import annotations

import numpy as np

BITS = 16
BLOCK = 16
MARGIN = 8
PITCH = 20
THRESHOLD = 128


def build_filter(width: int, height: int, fps: int, duration: float) -> str:
    """ffmpeg lavfi filter string that renders a test pattern plus the marker."""
    parts = [f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}"]
    # Opaque backing bar: without it, a cleared bit samples the underlying test
    # pattern, which is often bright. Every bit must have a known background.
    bar_w = BITS * PITCH + MARGIN
    parts.append(
        f"drawbox=x=0:y=0:w={bar_w}:h={BLOCK + 2 * MARGIN}:color=black:t=fill"
    )
    for i in range(BITS):
        x = i * PITCH + MARGIN
        p = 1 << i
        parts.append(
            f"drawbox=x={x}:y={MARGIN}:w={BLOCK}:h={BLOCK}"
            f":color=white:t=fill:enable='eq(mod(floor(n/{p})\\,2)\\,1)'"
        )
    return ",".join(parts)


def read_index(frame: np.ndarray) -> int:
    """Recover the frame index from a decoded BGR frame."""
    idx = 0
    cy = MARGIN + BLOCK // 2
    for i in range(BITS):
        cx = i * PITCH + MARGIN + BLOCK // 2
        # average a 5x5 patch: tolerates compression ringing at block edges
        patch = frame[cy - 2 : cy + 3, cx - 2 : cx + 3]
        if patch.mean() > THRESHOLD:
            idx |= 1 << i
    return idx
