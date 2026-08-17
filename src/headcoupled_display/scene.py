"""Placement of drawable content on and around the physical display plane.

The renderer must not carry hard-coded offsets. Every asset is placed by combining its
own axis-aligned bounding box with a :class:`~headcoupled_display.models.SceneProfile`,
so that changing what is displayed never silently changes the calibrated geometry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import SceneProfile, Vector3

FloatArray = NDArray[np.float64]


def scene_model_matrix(
    scene: SceneProfile,
    *,
    bounds_min: Vector3,
    bounds_max: Vector3,
) -> FloatArray:
    """Build ``T(anchor) @ S @ T(-aabb_center)`` for an asset with the given bounds.

    The asset is uniformly scaled so that its longest bounding-box edge measures
    ``scene.longest_edge_m`` on the physical screen, then its bounding-box midpoint is
    moved onto ``scene.anchor_display_m``. Coordinates are display metres: origin at the
    centre of the active area, +X right, +Y up, +Z toward the observer, screen at z=0.

    The midpoint is deliberately used rather than the centroid: the centroid follows the
    point density of the model, so a denser base would push the visible object off the
    window even though nothing about the display changed.
    """

    minimum = np.asarray(bounds_min, dtype=np.float64)
    maximum = np.asarray(bounds_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("bounds_min and bounds_max must be 3-vectors")
    span = maximum - minimum
    if not np.all(np.isfinite(span)):
        raise ValueError("asset bounds must be finite")
    if np.any(span < 0.0):
        raise ValueError("bounds_max must be greater than or equal to bounds_min")

    longest_edge = float(span.max())
    if longest_edge <= 0.0:
        raise ValueError("asset bounding box is degenerate; cannot derive a uniform scale")

    scale = scene.longest_edge_m / longest_edge
    centre = 0.5 * (minimum + maximum)
    anchor = np.asarray(scene.anchor_display_m, dtype=np.float64)

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = scale
    matrix[:3, 3] = anchor - scale * centre
    return matrix


def screen_frame_segments(scene: SceneProfile, width_m: float, height_m: float) -> FloatArray:
    """Line segments outlining the screen plane and its tick marks, in display metres.

    Returned as ``(N, 2, 3)``: the reference frame lives exactly on z=0, which is the one
    depth at which the projection is independent of the observer's eye. It is therefore
    the only usable visual datum for confirming that the off-axis projection is correct.
    """

    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("display extents must be positive")

    half_width = width_m / 2.0
    half_height = height_m / 2.0
    segments: list[list[list[float]]] = [
        [[-half_width, -half_height, 0.0], [half_width, -half_height, 0.0]],
        [[half_width, -half_height, 0.0], [half_width, half_height, 0.0]],
        [[half_width, half_height, 0.0], [-half_width, half_height, 0.0]],
        [[-half_width, half_height, 0.0], [-half_width, -half_height, 0.0]],
    ]
    segments.extend(_tick_segments(scene.grid_spacing_m, half_width, half_height))
    return np.asarray(segments, dtype=np.float64)


def _tick_segments(
    spacing_m: float, half_width: float, half_height: float
) -> list[list[list[float]]]:
    """Ruler ticks along the screen edges, one per grid spacing, pointing inward."""

    tick = spacing_m / 4.0
    segments: list[list[list[float]]] = []
    for x in _symmetric_steps(spacing_m, half_width):
        segments.append([[x, -half_height, 0.0], [x, -half_height + tick, 0.0]])
        segments.append([[x, half_height, 0.0], [x, half_height - tick, 0.0]])
    for y in _symmetric_steps(spacing_m, half_height):
        segments.append([[-half_width, y, 0.0], [-half_width + tick, y, 0.0]])
        segments.append([[half_width, y, 0.0], [half_width - tick, y, 0.0]])
    return segments


def _symmetric_steps(spacing_m: float, limit_m: float) -> list[float]:
    count = int(limit_m // spacing_m)
    return [step * spacing_m for step in range(-count, count + 1)]


def back_wall_segments(scene: SceneProfile, width_m: float, height_m: float) -> FloatArray:
    """A grid on the plane ``z = scene.back_wall_z_m``, as ``(N, 2, 3)`` display metres.

    The wall is sized to cover the screen aperture as seen from a nominal viewing
    distance, so it stays visible for the head motions this display is built for.
    """

    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("display extents must be positive")

    z = scene.back_wall_z_m
    half_width = width_m / 2.0 + abs(z) * 0.5
    half_height = height_m / 2.0 + abs(z) * 0.5
    spacing = scene.grid_spacing_m

    segments: list[list[list[float]]] = []
    for x in _symmetric_steps(spacing, half_width):
        segments.append([[x, -half_height, z], [x, half_height, z]])
    for y in _symmetric_steps(spacing, half_height):
        segments.append([[-half_width, y, z], [half_width, y, z]])
    return np.asarray(segments, dtype=np.float64)


def floor_segments(scene: SceneProfile, width_m: float) -> FloatArray:
    """A grid on the plane ``y = scene.floor_y_m`` receding from the observer.

    The floor spans ``floor_near_z_m`` down to ``floor_far_z_m`` so that it visually
    joins the back wall; a floor that stopped short would read as a floating slab.
    """

    if width_m <= 0.0:
        raise ValueError("display width must be positive")

    y = scene.floor_y_m
    near = scene.floor_near_z_m
    far = scene.floor_far_z_m
    half_width = width_m / 2.0 + abs(far) * 0.5
    spacing = scene.grid_spacing_m

    segments: list[list[list[float]]] = []
    for x in _symmetric_steps(spacing, half_width):
        segments.append([[x, y, near], [x, y, far]])
    depth_steps = int((near - far) // spacing)
    for step in range(depth_steps + 1):
        z = near - step * spacing
        segments.append([[-half_width, y, z], [half_width, y, z]])
    return np.asarray(segments, dtype=np.float64)
