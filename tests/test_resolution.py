from __future__ import annotations

import pytest

from multicam_bench.rig.resolution import resolve_resolution, validate_codec


def test_resolve_resolution_360p() -> None:
    r = resolve_resolution("360p")
    assert (r.width, r.height) == (640, 360)


def test_resolve_resolution_720p() -> None:
    r = resolve_resolution("720p")
    assert (r.width, r.height) == (1280, 720)


def test_resolve_resolution_1080p() -> None:
    r = resolve_resolution("1080p")
    assert (r.width, r.height) == (1920, 1080)


def test_resolve_resolution_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown resolution"):
        resolve_resolution("4k")


def test_validate_codec_accepts_h264_and_h265() -> None:
    assert validate_codec("libx264") == "libx264"
    assert validate_codec("libx265") == "libx265"


def test_validate_codec_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown codec"):
        validate_codec("libvpx")
