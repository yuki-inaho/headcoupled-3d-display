from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from headcoupled_display.geometry import (
    asymmetric_projection_matrix,
    build_camera_to_display_matrix,
    frustum_extents,
    invert_rigid_transform,
    mount_summary_from_camera_to_display,
    project_display_point,
    transform_point,
)
from headcoupled_display.models import CameraMount, DisplaySpec, HardwareProfile

ROOT = Path(__file__).resolve().parents[2]


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


def test_local_hardware_profile_mount_round_trips_to_confirmed_geometry() -> None:
    """Confirmed on-hardware mount (15 cm up, 0 cm forward, 12 deg down) round-trips.

    Builds ``T_S_C`` from ``config/hardware_profile.local.json``'s ``camera_mount`` and
    recovers the mount summary from that transform, matching the profile-level check in
    ``test_profiles.py`` but exercised at the geometry layer directly.
    """

    profile = HardwareProfile.load(ROOT / "config" / "hardware_profile.local.json")
    transform = build_camera_to_display_matrix(profile.camera_mount)
    summary = mount_summary_from_camera_to_display(transform)

    assert summary.horizontally_centered is True
    assert summary.height_above_center_cm == pytest.approx(15.0, abs=1e-9)
    assert summary.forward_offset_cm == pytest.approx(0.0, abs=1e-9)
    assert summary.pitch_down_deg == pytest.approx(12.0, abs=1e-9)
    assert summary.total_axis_tilt_from_display_normal_deg == pytest.approx(12.0, abs=1e-9)


def _demo_display() -> DisplaySpec:
    return DisplaySpec(pixel_width=2560, pixel_height=1440, width_m=0.596, height_m=0.335)


def _screen_crossing_x(eye: np.ndarray, point: np.ndarray) -> float:
    """Where the eye-to-point ray crosses the physical screen plane, by similar triangles.

    Closed form kept independent of the projection matrix so the test cannot pass by
    reproducing the same mistake twice.
    """

    ratio = eye[2] / (eye[2] - point[2])
    return float(eye[0] + ratio * (point[0] - eye[0]))


@pytest.mark.parametrize(
    "eye",
    [
        (0.0, 0.0, 0.60),
        (0.12, 0.05, 0.55),
        (-0.15, -0.04, 0.75),
    ],
)
def test_points_on_the_display_plane_project_to_a_fixed_screen_position(
    eye: tuple[float, float, float],
) -> None:
    """z=0 is the one depth whose image does not move with the head. It is the datum."""

    display = _demo_display()
    anchor = np.array([0.0, 0.0, 0.0])
    corner = np.array([display.width_m / 2.0, display.height_m / 2.0, 0.0])

    assert project_display_point(display, np.asarray(eye), anchor)[:2] == pytest.approx(
        [0.0, 0.0], abs=1e-12
    )
    assert project_display_point(display, np.asarray(eye), corner)[:2] == pytest.approx(
        [1.0, 1.0], abs=1e-12
    )


def test_background_follows_the_head_and_foreground_opposes_it() -> None:
    """Differential parallax is the whole illusion: the two must move in opposite senses."""

    display = _demo_display()
    centred = np.array([0.0, 0.0, 0.60])
    shifted_right = np.array([0.08, 0.0, 0.60])

    behind = np.array([0.0, 0.0, -0.30])
    in_front = np.array([0.0, 0.0, 0.07])

    behind_shift = (
        project_display_point(display, shifted_right, behind)[0]
        - project_display_point(display, centred, behind)[0]
    )
    front_shift = (
        project_display_point(display, shifted_right, in_front)[0]
        - project_display_point(display, centred, in_front)[0]
    )

    assert behind_shift > 0.0, "a point behind the window must track the head"
    assert front_shift < 0.0, "a point in front of the window must move against the head"


def test_projection_matches_the_independent_screen_crossing_formula() -> None:
    display = _demo_display()
    eye = np.array([0.11, -0.03, 0.62])
    for z in (-0.30, -0.072, 0.0, 0.072):
        point = np.array([0.04, 0.02, z])
        ndc_x = project_display_point(display, eye, point)[0]
        expected = _screen_crossing_x(eye, point) / (display.width_m / 2.0)
        assert ndc_x == pytest.approx(expected, abs=1e-9)


def test_parallax_magnitude_grows_with_depth_behind_the_window() -> None:
    display = _demo_display()
    centred = np.array([0.0, 0.0, 0.60])
    shifted = np.array([0.08, 0.0, 0.60])

    shifts = [
        abs(
            project_display_point(display, shifted, np.array([0.0, 0.0, z]))[0]
            - project_display_point(display, centred, np.array([0.0, 0.0, z]))[0]
        )
        for z in (-0.05, -0.15, -0.30)
    ]
    assert shifts[0] < shifts[1] < shifts[2]


def test_a_point_at_the_eye_plane_cannot_be_projected() -> None:
    display = _demo_display()
    eye = np.array([0.0, 0.0, 0.60])
    with pytest.raises(ValueError):
        project_display_point(display, eye, np.array([0.0, 0.0, 0.60]))
