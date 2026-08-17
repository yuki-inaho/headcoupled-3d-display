"""Coordinate transforms and asymmetric projection geometry."""

from __future__ import annotations

from math import acos, atan2, cos, degrees, hypot, radians, sin
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .models import CameraMount, DisplaySpec, Matrix4, MountSummary

FloatArray = NDArray[np.float64]


def normalize(vector: FloatArray, *, epsilon: float = 1e-12) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm < epsilon:
        raise ValueError("cannot normalize a near-zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def matrix_to_tuple(matrix: FloatArray) -> Matrix4:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {value.shape}")
    return cast(Matrix4, tuple(tuple(float(component) for component in row) for row in value))


def tuple_to_matrix(matrix: Matrix4) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {value.shape}")
    return value


def build_camera_to_display_matrix(mount: CameraMount) -> FloatArray:
    """Build ``T_S_C`` from the human-readable mount parameters.

    A camera with no yaw/pitch/roll faces along display +Z. Because image-right is the
    viewer's left and image-down is display-down, its zero-orientation basis is
    ``diag(-1, -1, +1)`` in display coordinates.
    """

    pitch = radians(mount.pitch_down_deg)
    yaw = radians(mount.yaw_right_deg)
    roll = radians(mount.roll_clockwise_deg)

    forward = normalize(
        np.array(
            [sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch)],
            dtype=np.float64,
        )
    )
    display_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    image_right_zero_roll = normalize(np.cross(forward, display_up))
    image_down_zero_roll = normalize(np.cross(forward, image_right_zero_roll))

    image_right = normalize(cos(roll) * image_right_zero_roll + sin(roll) * image_down_zero_roll)
    image_down = normalize(-sin(roll) * image_right_zero_roll + cos(roll) * image_down_zero_roll)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack([image_right, image_down, forward])
    transform[:3, 3] = np.array(
        [
            mount.horizontal_offset_m,
            mount.height_above_center_m,
            mount.forward_offset_m,
        ],
        dtype=np.float64,
    )
    return transform


def invert_rigid_transform(transform: FloatArray) -> FloatArray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {matrix.shape}")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("transform rotation is not orthonormal")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_point(transform: FloatArray, point: FloatArray) -> FloatArray:
    matrix = np.asarray(transform, dtype=np.float64)
    point_array = np.asarray(point, dtype=np.float64)
    if matrix.shape != (4, 4) or point_array.shape != (3,):
        raise ValueError("transform_point expects a 4x4 transform and a 3-vector")
    return matrix[:3, :3] @ point_array + matrix[:3, 3]


def transform_direction(transform: FloatArray, direction: FloatArray) -> FloatArray:
    matrix = np.asarray(transform, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    if matrix.shape != (4, 4) or direction_array.shape != (3,):
        raise ValueError("transform_direction expects a 4x4 transform and a 3-vector")
    return normalize(matrix[:3, :3] @ direction_array)


def mount_summary_from_camera_to_display(
    camera_to_display: FloatArray,
    *,
    centered_tolerance_m: float = 0.005,
) -> MountSummary:
    transform = np.asarray(camera_to_display, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    forward = normalize(rotation[:, 2])

    pitch_down = degrees(atan2(-forward[1], hypot(forward[0], forward[2])))
    yaw_right = degrees(atan2(forward[0], forward[2]))
    total_tilt = degrees(acos(float(np.clip(forward[2], -1.0, 1.0))))

    display_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    zero_roll_right = normalize(np.cross(forward, display_up))
    zero_roll_down = normalize(np.cross(forward, zero_roll_right))
    image_right = normalize(rotation[:, 0])
    roll_clockwise = degrees(
        atan2(
            float(np.dot(image_right, zero_roll_down)), float(np.dot(image_right, zero_roll_right))
        )
    )

    return MountSummary(
        horizontal_offset_cm=float(translation[0] * 100.0),
        height_above_center_cm=float(translation[1] * 100.0),
        forward_offset_cm=float(translation[2] * 100.0),
        pitch_down_deg=float(pitch_down),
        yaw_right_deg=float(yaw_right),
        roll_clockwise_deg=float(roll_clockwise),
        total_axis_tilt_from_display_normal_deg=float(total_tilt),
        horizontally_centered=abs(float(translation[0])) <= centered_tolerance_m,
    )


def display_target_point(display: DisplaySpec, target_uv: tuple[float, float]) -> FloatArray:
    """Convert normalized browser coordinates to a point on the physical screen plane."""

    u, v = target_uv
    return np.array(
        [
            (u - 0.5) * display.width_m,
            (0.5 - v) * display.height_m,
            0.0,
        ],
        dtype=np.float64,
    )


def frustum_extents(
    display: DisplaySpec,
    eye_display_m: FloatArray,
    *,
    near_m: float = 0.05,
) -> tuple[float, float, float, float]:
    eye = np.asarray(eye_display_m, dtype=np.float64)
    if eye.shape != (3,):
        raise ValueError("eye_display_m must be a 3-vector")
    if near_m <= 0:
        raise ValueError("near_m must be positive")
    distance = float(eye[2])
    if distance <= 1e-6:
        raise ValueError("eye must be in front of the display plane (+Z)")
    left = near_m * (-display.width_m / 2.0 - eye[0]) / distance
    right = near_m * (display.width_m / 2.0 - eye[0]) / distance
    bottom = near_m * (-display.height_m / 2.0 - eye[1]) / distance
    top = near_m * (display.height_m / 2.0 - eye[1]) / distance
    return float(left), float(right), float(bottom), float(top)


def view_matrix(eye_display_m: FloatArray) -> FloatArray:
    """World-to-eye transform for a head-coupled window.

    The observer's orientation never rotates the frustum: the window is fixed to the
    physical screen and only the eye position moves, so the view transform is a pure
    translation. Head rotation is deliberately ignored here; it changes what the observer
    looks at, not where the window is.
    """

    eye = np.asarray(eye_display_m, dtype=np.float64)
    if eye.shape != (3,):
        raise ValueError("eye_display_m must be a 3-vector")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = -eye
    return matrix


def project_display_point(
    display: DisplaySpec,
    eye_display_m: FloatArray,
    point_display_m: FloatArray,
    *,
    near_m: float = 0.05,
    far_m: float = 10.0,
) -> FloatArray:
    """Project a display-frame point to normalized device coordinates.

    This is the Python counterpart of what the WebGL renderer computes, so geometry can
    be asserted numerically instead of by comparing screenshots.
    """

    point = np.asarray(point_display_m, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("point_display_m must be a 3-vector")
    projection = asymmetric_projection_matrix(display, eye_display_m, near_m=near_m, far_m=far_m)
    view = view_matrix(eye_display_m)
    clip = projection @ view @ np.array([*point, 1.0], dtype=np.float64)
    if clip[3] <= 1e-12:
        raise ValueError("point is at or behind the observer and cannot be projected")
    return cast(FloatArray, clip[:3] / clip[3])


def asymmetric_projection_matrix(
    display: DisplaySpec,
    eye_display_m: FloatArray,
    *,
    near_m: float = 0.05,
    far_m: float = 10.0,
) -> FloatArray:
    if far_m <= near_m:
        raise ValueError("far_m must be greater than near_m")
    left, right, bottom, top = frustum_extents(display, eye_display_m, near_m=near_m)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = 2.0 * near_m / (right - left)
    matrix[1, 1] = 2.0 * near_m / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[2, 2] = -(far_m + near_m) / (far_m - near_m)
    matrix[2, 3] = -(2.0 * far_m * near_m) / (far_m - near_m)
    matrix[3, 2] = -1.0
    return matrix
