from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from multicam_bench.pipeline.detector import BoundingBox, RtDetrV2Detector

_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


def test_bounding_box_to_centroid_detection() -> None:
    box = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0, class_name="car", score=0.87)
    det = box.to_centroid_detection()
    assert det.cx == pytest.approx(20.0)
    assert det.cy == pytest.approx(30.0)
    assert det.class_name == "car"
    assert det.score == pytest.approx(0.87)


@pytest.mark.skipif(
    _TORCH_INSTALLED,
    reason="torch is installed — the optional-extra ImportError path can't be exercised",
)
def test_detect_without_optional_extra_raises_actionable_import_error() -> None:
    # This repo's base install deliberately excludes torch/transformers (v0.5:
    # detection is optional, `uv sync --extra detect` opts in). Without them,
    # calling detect() must fail loudly with instructions, not an opaque traceback.
    detector = RtDetrV2Detector("PekingU/rtdetr_v2_r18vd")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match="extra detect"):
        detector.detect(frame)


def test_lazy_import_means_construction_never_touches_torch() -> None:
    # Constructing the detector must not require torch/transformers at all —
    # only detect() (which loads the model) does.
    detector = RtDetrV2Detector("PekingU/rtdetr_v2_r18vd", device="cuda")
    assert detector.model_name == "PekingU/rtdetr_v2_r18vd"
    assert detector.device == "cuda"
    assert detector.resolved_device is None
