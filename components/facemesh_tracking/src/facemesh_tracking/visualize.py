"""Drawing of FaceMesh results.

Kept free of any inference dependency so the same renderer works for landmarks that came
from a file, a different model, or a test fixture.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from enum import Enum
from functools import lru_cache
from importlib import resources

import cv2
import numpy as np

from .geometry import NUM_LANDMARKS, BBox, FaceLandmarks

Color = tuple[int, int, int]  # BGR


class DrawingMode(str, Enum):
    """How much of the mesh to draw."""

    FULL = "full"  # dense MediaPipe tessellation (2556 edges)
    PARTIAL = "partial"  # contours only (124 edges) + points
    POINTS = "points"  # landmarks only


@lru_cache(maxsize=1)
def _tessellations() -> dict[str, np.ndarray]:
    """MediaPipe FaceMesh edge lists, bundled as an asset (see tools/extract_tessellation.py)."""
    raw = resources.files(__package__).joinpath("assets/tessellation.json").read_text()
    return {key: np.asarray(edges, dtype=np.int32) for key, edges in json.loads(raw).items()}


def edges_for(mode: DrawingMode) -> np.ndarray:
    return {
        DrawingMode.FULL: lambda: _tessellations()["full"],
        DrawingMode.PARTIAL: lambda: _tessellations()["partial"],
        DrawingMode.POINTS: lambda: np.empty((0, 2), dtype=np.int32),
    }[mode]()


def draw_landmarks(
    canvas: np.ndarray,
    faces: Iterable[FaceLandmarks],
    *,
    mode: DrawingMode = DrawingMode.FULL,
    mesh_color: Color = (0, 255, 0),
    point_color: Color = (255, 209, 0),
    iris_color: Color = (0, 0, 255),
    point_radius: int = 1,
    thickness: int = 1,
) -> np.ndarray:
    """Draw the mesh/points of every face onto ``canvas`` in place and return it.

    The tessellation only spans the 468 base points; the 10 iris points of the 478-point
    model are always drawn as dots in ``iris_color`` so they stay visible in every mode.
    """
    edges = edges_for(mode)
    for face in faces:
        xy = np.rint(face.xy).astype(np.int32)
        if edges.size:
            segments = xy[:NUM_LANDMARKS][edges]  # (E, 2, 2)
            cv2.polylines(canvas, segments, isClosed=False, color=mesh_color, thickness=thickness)
        if mode is not DrawingMode.FULL:
            for x, y in xy[:NUM_LANDMARKS]:
                cv2.circle(canvas, (int(x), int(y)), point_radius, point_color, thickness)
        for x, y in xy[NUM_LANDMARKS:]:
            cv2.circle(canvas, (int(x), int(y)), point_radius + 1, iris_color, -1)
    return canvas


def draw_boxes(
    canvas: np.ndarray,
    boxes: Iterable[BBox],
    *,
    color: Color = (255, 0, 0),
    thickness: int = 2,
    show_score: bool = False,
) -> np.ndarray:
    for box in boxes:
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
        if show_score:
            cv2.putText(
                canvas,
                f"{box.score:.2f}",
                (box.x1, max(box.y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def render(
    frame_bgr: np.ndarray,
    faces: Sequence[FaceLandmarks],
    boxes: Sequence[BBox] = (),
    *,
    mode: DrawingMode = DrawingMode.FULL,
    show_background: bool = True,
    show_boxes: bool = True,
) -> np.ndarray:
    """Compose a display frame; never mutates ``frame_bgr``."""
    canvas = frame_bgr.copy() if show_background else np.zeros_like(frame_bgr)
    if show_boxes:
        draw_boxes(canvas, boxes)
    return draw_landmarks(canvas, faces, mode=mode)
