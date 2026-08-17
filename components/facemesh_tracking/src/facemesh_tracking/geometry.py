"""Value objects shared by detection, landmark estimation and drawing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

NUM_LANDMARKS = 468
NUM_LANDMARKS_WITH_IRISES = 478
#: MediaPipe ships two dense meshes; the 478-point one is the 468-point one plus 10 iris
#: points appended at the end, so the first 468 indices mean the same thing in both.
SUPPORTED_LANDMARK_COUNTS = frozenset({NUM_LANDMARKS, NUM_LANDMARKS_WITH_IRISES})


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in *source image* pixel coordinates, ``x2``/``y2`` exclusive.

    ``keypoints`` is the detector's optional 5-point alignment template (left eye, right
    eye, nose, left mouth corner, right mouth corner) as ``(5, 2)`` pixel coordinates.
    Dense-mesh estimators use the two eye points to rotate the crop upright, which is what
    keeps landmarks accurate on tilted faces. It is excluded from equality and repr so a
    box stays comparable as a plain rectangle.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    keypoints: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def clipped(self, image_width: int, image_height: int) -> BBox:
        return replace(
            self,
            x1=int(np.clip(self.x1, 0, image_width)),
            y1=int(np.clip(self.y1, 0, image_height)),
            x2=int(np.clip(self.x2, 0, image_width)),
            y2=int(np.clip(self.y2, 0, image_height)),
        )

    def expanded(self, ratio: float, image_width: int, image_height: int) -> BBox:
        """Grow the box by ``ratio`` of its size on each side, then clip to the image.

        FaceMesh needs the whole face inside the crop; head/face detectors often return a
        tight box, hence the ~25% margin recommended by the model card.
        """
        margin_x = int(round(self.width * ratio))
        margin_y = int(round(self.height * ratio))
        return replace(
            self,
            x1=self.x1 - margin_x,
            y1=self.y1 - margin_y,
            x2=self.x2 + margin_x,
            y2=self.y2 + margin_y,
        ).clipped(image_width, image_height)


@dataclass(frozen=True)
class FaceLandmarks:
    """Dense FaceMesh keypoints in source-image pixel coordinates.

    ``points`` has shape ``(468, 3)`` or ``(478, 3)``: columns are X, Y (pixels) and Z
    (relative depth, same scale as X). ``score`` is the model's landmark presence
    probability.
    """

    points: np.ndarray
    score: float
    bbox: BBox

    def __post_init__(self) -> None:
        shape = self.points.shape
        if len(shape) != 2 or shape[0] not in SUPPORTED_LANDMARK_COUNTS or shape[1] != 3:
            expected = " or ".join(str(n) for n in sorted(SUPPORTED_LANDMARK_COUNTS))
            raise ValueError(f"points must have shape ({expected}, 3), got {shape}")

    @property
    def xy(self) -> np.ndarray:
        """``(N, 2)`` array of pixel coordinates."""
        return self.points[:, :2]

    @property
    def has_irises(self) -> bool:
        return self.points.shape[0] == NUM_LANDMARKS_WITH_IRISES

    @property
    def irises(self) -> np.ndarray:
        """``(10, 2)`` iris points, empty when the model does not predict them."""
        return self.xy[NUM_LANDMARKS:] if self.has_irises else self.xy[:0]
