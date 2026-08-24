"""Resolution presets for the test-video generator (360p/720p/1080p) and the set of
codecs it supports (libx264/libx265). PROJECT-CHARTER-v2.md §2 lists resolution and
codec as sweep axes, not defaults — this is the enumeration those axes draw from.
"""

from __future__ import annotations

from dataclasses import dataclass

CODECS = ("libx264", "libx265")


@dataclass(frozen=True)
class Resolution:
    name: str
    width: int
    height: int


RESOLUTIONS: dict[str, Resolution] = {
    "360p": Resolution("360p", 640, 360),
    "720p": Resolution("720p", 1280, 720),
    "1080p": Resolution("1080p", 1920, 1080),
}


def resolve_resolution(name: str) -> Resolution:
    try:
        return RESOLUTIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown resolution {name!r}; choose from {sorted(RESOLUTIONS)}"
        ) from exc


def validate_codec(codec: str) -> str:
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; choose from {CODECS}")
    return codec
