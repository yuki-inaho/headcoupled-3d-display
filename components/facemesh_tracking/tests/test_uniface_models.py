"""Integration tests for the UniFace-backed stages (CPU backend, ~16 MB of weights).

Assert the contract the pipeline depends on — protocol conformance, coordinate space,
keypoint propagation, batching — not detection quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_uniface

from facemesh_tracking.geometry import NUM_LANDMARKS_WITH_IRISES, BBox
from facemesh_tracking.protocols import FaceDetector, LandmarkEstimator
from facemesh_tracking.runtime import Backend
from facemesh_tracking.uniface_models import (
    DEFAULT_MODELS_DIR,
    MESH_V1_468,
    MESH_V2_478,
    UnifaceFaceDetector,
    UnifaceFaceMesh,
)

pytestmark = requires_uniface()


@pytest.fixture(scope="module")
def detector() -> UnifaceFaceDetector:
    return UnifaceFaceDetector(Backend.CPU, models_dir=DEFAULT_MODELS_DIR)


@pytest.fixture(scope="module")
def mesher() -> UnifaceFaceMesh:
    return UnifaceFaceMesh(Backend.CPU, score_threshold=0.0, models_dir=DEFAULT_MODELS_DIR)


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def test_stages_satisfy_the_pipeline_protocols(detector, mesher):
    assert isinstance(detector, FaceDetector)
    assert isinstance(mesher, LandmarkEstimator)


def test_detector_returns_no_faces_for_noise(detector, frame):
    assert detector.detect(frame) == []


def test_mesh_returns_478_float_points_inside_the_frame(mesher, frame):
    box = BBox(200, 150, 400, 400)
    (face,) = mesher.estimate(frame, [box])

    assert face.points.shape == (NUM_LANDMARKS_WITH_IRISES, 3)
    assert face.points.dtype == np.float32
    assert face.has_irises
    assert face.bbox == box
    assert 0.0 <= face.score <= 1.0


def test_mesh_handles_multiple_boxes_in_one_call(mesher, frame):
    boxes = [BBox(0, 0, 200, 200), BBox(200, 150, 400, 400), BBox(400, 250, 620, 460)]
    faces = mesher.estimate(frame, boxes)
    assert [f.bbox for f in faces] == boxes


def test_mesh_ignores_degenerate_boxes(mesher, frame):
    assert mesher.estimate(frame, [BBox(10, 10, 10, 50)]) == []
    assert mesher.estimate(frame, []) == []


def test_mesh_accepts_boxes_with_alignment_keypoints(mesher, frame):
    """Keypoints are the whole point of this engine: they must reach UniFace intact."""
    kps = np.array([[240, 220], [330, 225], [285, 280], [245, 330], [325, 335]], np.float32)
    box = BBox(200, 150, 400, 400, keypoints=kps)
    aligned = mesher.estimate(frame, [box])[0]
    unaligned = mesher.estimate(frame, [BBox(200, 150, 400, 400)])[0]

    assert aligned.points.shape == unaligned.points.shape
    assert not np.allclose(aligned.xy, unaligned.xy), "alignment keypoints had no effect"


def test_v1_468_variant_drops_the_irises():
    mesher = UnifaceFaceMesh(
        Backend.CPU, model_name=MESH_V1_468, score_threshold=0.0, models_dir=DEFAULT_MODELS_DIR
    )
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)

    (face,) = mesher.estimate(frame, [BBox(50, 50, 250, 250)])

    assert not face.has_irises
    assert mesher.model_name == MESH_V1_468 != MESH_V2_478
