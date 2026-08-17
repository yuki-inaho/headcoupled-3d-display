"""Detection -> optional margin -> landmark estimation, composed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import BBox, FaceLandmarks
from .protocols import FaceDetector, LandmarkEstimator
from .runtime import Backend
from .uniface_models import DEFAULT_MODELS_DIR, UnifaceFaceDetector, UnifaceFaceMesh


@dataclass(frozen=True)
class FaceMeshResult:
    """Everything one frame produced, ready for drawing or serialisation."""

    boxes: list[BBox]
    faces: list[FaceLandmarks]

    def __len__(self) -> int:
        return len(self.faces)


class FaceMeshPipeline:
    """Runs a :class:`FaceDetector` and feeds its boxes to a :class:`LandmarkEstimator`.

    ``margin_ratio`` expands every detected box before the mesh stage. It defaults to 0.0
    because the bundled mesh applies its own 25% expansion while aligning the crop —
    expanding here as well would double it and push the face out of frame. Raise it only
    for a detector that frames faces more tightly than YOLOv8-Face.
    """

    def __init__(
        self,
        detector: FaceDetector,
        estimator: LandmarkEstimator,
        *,
        margin_ratio: float = 0.0,
    ) -> None:
        self.detector = detector
        self.estimator = estimator
        self.margin_ratio = margin_ratio

    @classmethod
    def create(
        cls,
        backend: Backend = Backend.CUDA,
        models_dir: Path = DEFAULT_MODELS_DIR,
        *,
        detection_threshold: float = 0.5,
        landmark_threshold: float = 0.5,
        margin_ratio: float = 0.0,
        device_id: int = 0,
    ) -> FaceMeshPipeline:
        """Build the default YOLOv8-Face + FaceMesh V2_478 pipeline.

        Weights are fetched into ``models_dir`` on first use.
        """
        detector = UnifaceFaceDetector(
            backend,
            score_threshold=detection_threshold,
            device_id=device_id,
            models_dir=models_dir,
        )
        estimator = UnifaceFaceMesh(
            backend,
            score_threshold=landmark_threshold,
            device_id=device_id,
            models_dir=models_dir,
        )
        return cls(detector, estimator, margin_ratio=margin_ratio)

    def process(self, image_bgr: np.ndarray) -> FaceMeshResult:
        height, width = image_bgr.shape[:2]
        boxes = [
            box.expanded(self.margin_ratio, width, height)
            for box in self.detector.detect(image_bgr)
        ]
        return FaceMeshResult(boxes=boxes, faces=self.estimator.estimate(image_bgr, boxes))
