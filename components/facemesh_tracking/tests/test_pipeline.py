"""Pipeline composition tests using stand-in stages (no ONNX involved)."""

from __future__ import annotations

import numpy as np
import pytest

from facemesh_tracking.geometry import NUM_LANDMARKS, BBox, FaceLandmarks
from facemesh_tracking.pipeline import FaceMeshPipeline
from facemesh_tracking.protocols import FaceDetector


class StubDetector:
    def __init__(self, boxes: list[BBox]) -> None:
        self.boxes = boxes
        self.calls = 0

    def detect(self, image_bgr: np.ndarray) -> list[BBox]:
        self.calls += 1
        return list(self.boxes)


class RecordingEstimator:
    """Accepts whatever boxes the pipeline hands over and echoes them back."""

    def __init__(self) -> None:
        self.received: list[BBox] = []

    def estimate(self, image_bgr: np.ndarray, boxes) -> list[FaceLandmarks]:
        self.received = list(boxes)
        return [FaceLandmarks(np.zeros((NUM_LANDMARKS, 3), np.int32), 1.0, box) for box in boxes]


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((400, 600, 3), dtype=np.uint8)


def test_stub_detector_satisfies_the_protocol():
    assert isinstance(StubDetector([]), FaceDetector)


def test_boxes_reach_the_estimator_unchanged_without_margin(frame):
    box = BBox(100, 100, 200, 200, score=0.8)
    estimator = RecordingEstimator()
    result = FaceMeshPipeline(StubDetector([box]), estimator).process(frame)

    assert estimator.received == [box]
    assert result.boxes == [box]
    assert len(result) == 1


def test_margin_is_applied_before_cropping(frame):
    estimator = RecordingEstimator()
    pipeline = FaceMeshPipeline(
        StubDetector([BBox(100, 100, 200, 200)]), estimator, margin_ratio=0.25
    )

    (expanded,) = pipeline.process(frame).boxes

    assert (expanded.x1, expanded.y1, expanded.x2, expanded.y2) == (75, 75, 225, 225)
    assert estimator.received == [expanded]


def test_margin_never_leaves_the_frame(frame):
    pipeline = FaceMeshPipeline(
        StubDetector([BBox(0, 0, 600, 400)]), RecordingEstimator(), margin_ratio=0.5
    )
    (box,) = pipeline.process(frame).boxes
    assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 600, 400)


def test_no_detections_yields_an_empty_result(frame):
    result = FaceMeshPipeline(StubDetector([]), RecordingEstimator()).process(frame)
    assert result.boxes == [] and result.faces == [] and len(result) == 0


def test_detector_is_called_once_per_frame(frame):
    detector = StubDetector([])
    pipeline = FaceMeshPipeline(detector, RecordingEstimator())
    for _ in range(3):
        pipeline.process(frame)
    assert detector.calls == 3
