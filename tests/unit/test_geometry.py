from __future__ import annotations

import numpy as np
import pytest

from headcoupled_display.geometry import (
    asymmetric_projection_matrix,
    build_camera_to_display_matrix,
    frustum_extents,
    invert_rigid_transform,
    mount_summary_from_camera_to_display,
    transform_point,
)
from headcoupled_display.models import CameraMount, DisplaySpec


def test_mount_summary_round_trip_for_centered_downward_camera() -> None:
    mount = CameraMount(
        horizontal_offset_m=0.0,
        height_above_center_m=0.2,
        forward_offset_m=0.025,
        pitch_down_deg=10.0,
        yaw_right_deg=0.0,
        roll_clockwise_deg=0.0,
    )
    transform = build_camera_to_display_matrix(mount)
    summary = mount_summary_from_camera_to_display(transform)

    assert summary.horizontally_centered is True
    assert summary.height_above_center_cm == pytest.approx(20.0, abs=1e-9)
    assert summary.forward_offset_cm == pytest.approx(2.5, abs=1e-9)
    assert summary.pitch_down_deg == pytest.approx(10.0, abs=1e-9)
    assert summary.total_axis_tilt_from_display_normal_deg == pytest.approx(10.0, abs=1e-9)


def test_rigid_transform_inverse_restores_point() -> None:
    transform = build_camera_to_display_matrix(
        CameraMount(
            horizontal_offset_m=0.012,
            height_above_center_m=0.19,
            forward_offset_m=0.03,
            pitch_down_deg=12.0,
            yaw_right_deg=-3.0,
            roll_clockwise_deg=1.5,
        )
    )
    inverse = invert_rigid_transform(transform)
    point = np.array([0.04, -0.02, 0.71])
    restored = transform_point(inverse, transform_point(transform, point))
    assert restored == pytest.approx(point, abs=1e-10)


def test_centered_eye_produces_symmetric_frustum() -> None:
    display = DisplaySpec(pixel_width=1920, pixel_height=1080, width_m=0.53, height_m=0.298)
    left, right, bottom, top = frustum_extents(display, np.array([0.0, 0.0, 0.65]))
    assert left == pytest.approx(-right)
    assert bottom == pytest.approx(-top)
    matrix = asymmetric_projection_matrix(display, np.array([0.0, 0.0, 0.65]))
    assert matrix.shape == (4, 4)
    assert matrix[0, 2] == pytest.approx(0.0)
    assert matrix[1, 2] == pytest.approx(0.0)


def test_lateral_eye_shift_changes_projection_center() -> None:
    display = DisplaySpec(pixel_width=1920, pixel_height=1080, width_m=0.53, height_m=0.298)
    centered = asymmetric_projection_matrix(display, np.array([0.0, 0.0, 0.65]))
    shifted = asymmetric_projection_matrix(display, np.array([0.08, 0.0, 0.65]))
    assert shifted[0, 2] != pytest.approx(centered[0, 2])
