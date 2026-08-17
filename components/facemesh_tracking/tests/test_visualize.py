from __future__ import annotations

import numpy as np
import pytest

from facemesh_tracking.geometry import (
    NUM_LANDMARKS,
    NUM_LANDMARKS_WITH_IRISES,
    BBox,
    FaceLandmarks,
)
from facemesh_tracking.visualize import DrawingMode, draw_boxes, edges_for, render


@pytest.fixture
def face() -> FaceLandmarks:
    rng = np.random.default_rng(0)
    points = np.empty((NUM_LANDMARKS, 3), dtype=np.int32)
    points[:, 0] = rng.integers(10, 90, NUM_LANDMARKS)
    points[:, 1] = rng.integers(10, 90, NUM_LANDMARKS)
    points[:, 2] = 0
    return FaceLandmarks(points=points, score=0.99, bbox=BBox(5, 5, 95, 95, score=0.9))


@pytest.mark.parametrize(
    ("mode", "expected_edges"),
    [(DrawingMode.FULL, 2556), (DrawingMode.PARTIAL, 124), (DrawingMode.POINTS, 0)],
)
def test_edge_lists_match_the_bundled_asset(mode, expected_edges):
    edges = edges_for(mode)
    assert edges.shape == (expected_edges, 2)
    if edges.size:
        assert edges.max() < NUM_LANDMARKS


@pytest.mark.parametrize("mode", list(DrawingMode))
def test_render_draws_without_touching_the_input_frame(face, mode):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas = render(frame, [face], [face.bbox], mode=mode)

    assert canvas.shape == frame.shape
    assert canvas.any(), "nothing was drawn"
    assert not frame.any(), "the source frame was mutated"


def test_render_without_background_starts_from_black(face):
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    canvas = render(frame, [face], show_background=False, show_boxes=False)
    assert (canvas == 200).sum() < canvas.size // 2


def test_render_with_no_faces_returns_a_copy_of_the_frame():
    frame = np.full((20, 20, 3), 7, dtype=np.uint8)
    canvas = render(frame, [], [])
    np.testing.assert_array_equal(canvas, frame)
    assert canvas is not frame


@pytest.mark.parametrize("mode", list(DrawingMode))
def test_iris_points_are_drawn_in_every_mode(mode):
    """The tessellation covers only 468 points, so irises need their own dots."""
    rng = np.random.default_rng(1)
    points = np.zeros((NUM_LANDMARKS_WITH_IRISES, 3), dtype=np.float32)
    points[:NUM_LANDMARKS, :2] = 1  # base mesh crammed into a corner
    points[NUM_LANDMARKS:, 0] = rng.integers(60, 90, 10)
    points[NUM_LANDMARKS:, 1] = rng.integers(60, 90, 10)
    face = FaceLandmarks(points, 1.0, BBox(0, 0, 100, 100))

    canvas = render(np.zeros((100, 100, 3), np.uint8), [face], mode=mode, show_boxes=False)

    iris_region = canvas[55:95, 55:95]
    assert (iris_region[..., 2] == 255).any(), "iris dots missing"


def test_subpixel_landmarks_are_rounded_not_truncated():
    """float32 meshes carry sub-pixel positions; 10.6 must land on 11, not 10."""
    points = np.full((NUM_LANDMARKS, 3), 10.6, dtype=np.float32)
    face = FaceLandmarks(points, 1.0, BBox(0, 0, 50, 50))
    canvas = render(
        np.zeros((50, 50, 3), np.uint8), [face], mode=DrawingMode.POINTS, show_boxes=False
    )

    ys, xs = np.nonzero(canvas.any(axis=2))
    assert (round(xs.mean()), round(ys.mean())) == (11, 11)


def test_draw_boxes_marks_the_border():
    canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_boxes(canvas, [BBox(10, 10, 40, 40)], color=(255, 0, 0), thickness=1)
    assert canvas[10, 20].tolist() == [255, 0, 0]
    assert canvas[25, 25].tolist() == [0, 0, 0]
