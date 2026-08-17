"""The two interfaces the pipeline is built on.

Keeping them free of any model or runtime import is what lets a caller drop in their own
detector or landmarker without touching this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .geometry import BBox, FaceLandmarks


@runtime_checkable
class FaceDetector(Protocol):
    """Detects faces and returns boxes in source-image pixel coordinates.

    Implementations should populate :attr:`BBox.keypoints` with the 5-point alignment
    template when they can: dense-mesh estimators use the eye points to rotate the crop
    upright, which is what keeps landmarks accurate on tilted faces.
    """

    def detect(self, image_bgr: np.ndarray) -> list[BBox]: ...


@runtime_checkable
class LandmarkEstimator(Protocol):
    """Turns detected boxes into dense landmarks in source-image pixel coordinates."""

    def estimate(self, image_bgr: np.ndarray, boxes: Sequence[BBox]) -> list[FaceLandmarks]: ...
