from __future__ import annotations

import numpy as np
import pytest

from facemesh_tracking.geometry import (
    NUM_LANDMARKS,
    NUM_LANDMARKS_WITH_IRISES,
    BBox,
    FaceLandmarks,
)


def test_size_properties():
    box = BBox(10, 20, 40, 60, score=0.9)
    assert (box.width, box.height) == (30, 40)
    assert box.is_valid


def test_degenerate_box_is_invalid():
    assert not BBox(10, 10, 10, 30).is_valid
    assert not BBox(10, 10, 30, 10).is_valid


def test_expanded_applies_ratio_on_every_side():
    box = BBox(100, 100, 200, 300).expanded(0.25, 1000, 1000)
    assert (box.x1, box.y1, box.x2, box.y2) == (75, 50, 225, 350)


def test_expanded_clips_to_image_bounds():
    box = BBox(10, 10, 110, 110).expanded(0.5, 120, 120)
    assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 120, 120)


def test_expanded_preserves_score():
    assert BBox(10, 10, 20, 20, score=0.42).expanded(0.1, 100, 100).score == pytest.approx(0.42)


def test_keypoints_are_carried_through_expansion_and_clipping():
    kps = np.arange(10, dtype=np.float32).reshape(5, 2)
    box = BBox(100, 100, 200, 200, keypoints=kps).expanded(0.1, 1000, 1000)
    np.testing.assert_array_equal(box.keypoints, kps)


def test_keypoints_do_not_affect_equality():
    """Equality must stay a plain rectangle comparison, or every box test breaks."""
    plain = BBox(10, 10, 20, 20)
    with_kps = BBox(10, 10, 20, 20, keypoints=np.zeros((5, 2), np.float32))
    assert plain == with_kps
    assert repr(with_kps) == repr(plain)


def test_face_landmarks_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        FaceLandmarks(points=np.zeros((10, 3)), score=1.0, bbox=BBox(0, 0, 1, 1))


@pytest.mark.parametrize("count", [NUM_LANDMARKS, NUM_LANDMARKS_WITH_IRISES])
def test_face_landmarks_accepts_both_mesh_sizes(count):
    face = FaceLandmarks(np.zeros((count, 3), np.float32), 1.0, BBox(0, 0, 1, 1))
    assert face.xy.shape == (count, 2)


def test_iris_points_are_exposed_only_by_the_478_point_mesh():
    with_irises = FaceLandmarks(
        np.zeros((NUM_LANDMARKS_WITH_IRISES, 3), np.float32), 1.0, BBox(0, 0, 1, 1)
    )
    without = FaceLandmarks(np.zeros((NUM_LANDMARKS, 3), np.float32), 1.0, BBox(0, 0, 1, 1))

    assert with_irises.has_irises and with_irises.irises.shape == (10, 2)
    assert not without.has_irises and without.irises.shape == (0, 2)


def test_face_landmarks_xy_view():
    points = np.arange(NUM_LANDMARKS * 3, dtype=np.int32).reshape(NUM_LANDMARKS, 3)
    face = FaceLandmarks(points=points, score=0.99, bbox=BBox(0, 0, 10, 10))
    assert face.xy.shape == (NUM_LANDMARKS, 2)
    np.testing.assert_array_equal(face.xy[0], [0, 1])
