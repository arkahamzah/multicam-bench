"""RT-DETRv2 detector wrapper (Apache-2.0, via `transformers`) — the v0.5 optional
detection stage. Default OFF (`configs/detect.yaml` `enabled: false`);
`transformers`/`torch` are an optional extras group
(`pip install multicam-bench[detect]` / `uv sync --extra detect`), not a base
dependency, so a plain decode-capacity sweep never needs a multi-GB ML stack.
CLAUDE.md hard rule 6 keeps Ultralytics out of the base install for AGPL reasons;
this keeps torch/transformers out for a different reason — they are simply not
needed unless the detection axis is turned on.

The dev machine this repo targets has an MX550 (2GB VRAM). Even the smallest
RT-DETRv2 checkpoint may not fit alongside a live decode workload on that budget —
that is a documented limitation, not a claim the detector runs well here. The
stage stays fully optional so a capacity-only sweep is never blocked by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from multicam_bench.pipeline.counting import Detection

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    score: float

    def to_centroid_detection(self) -> Detection:
        return Detection(
            cx=(self.x1 + self.x2) / 2.0,
            cy=(self.y1 + self.y2) / 2.0,
            class_name=self.class_name,
            score=self.score,
        )


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[BoundingBox]: ...


class RtDetrV2Detector:
    """Lazy-loads `transformers`/`torch` on first `detect()` call, not on
    construction or import — so this module (and code that only type-checks
    against it) never requires either package to be installed.
    """

    def __init__(
        self, model_name: str, device: str = "cpu", score_threshold: float = 0.5
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.score_threshold = score_threshold
        self.resolved_device: str | None = None
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "RT-DETRv2 needs the optional 'detect' extra: "
                "uv sync --extra detect (installs torch + transformers)"
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModelForObjectDetection.from_pretrained(self.model_name)

        resolved_device = self.device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"  # recorded on self, not silently hidden from the caller
        self.resolved_device = resolved_device
        self._model = model.to(resolved_device)

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        self._load()
        import torch

        model = self._model
        processor = self._processor
        assert model is not None
        assert processor is not None

        inputs = processor(images=frame, return_tensors="pt").to(self.resolved_device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.score_threshold
        )[0]

        id2label = model.config.id2label
        boxes: list[BoundingBox] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"], strict=True
        ):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            boxes.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_name=id2label[int(label)],
                    score=float(score),
                )
            )
        return boxes
