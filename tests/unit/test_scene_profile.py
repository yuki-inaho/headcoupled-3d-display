"""Scene profile contract: how the point cloud is anchored on the physical display plane.

The hardware profile answers "where is the camera relative to the screen". This scene
profile answers "where in front of and behind the screen do we place what we draw", and
the two must not be merged: a scene change must never look like a calibration change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from headcoupled_display.models import SceneProfile
from headcoupled_display.scene import scene_model_matrix

# Ground truth for src/headcoupled_display/static/assets/bunny.pcd (13810 points),
# computed independently with NumPy. ``center`` is the AABB midpoint, not the centroid.
BUNNY_MIN = (-0.5836868, -0.5006086, -0.5753633)
BUNNY_MAX = (0.5809016, 1.1862427, 0.4353630)
BUNNY_CENTER = (-0.0013926, 0.34281705, -0.07000015)
BUNNY_LONGEST_EDGE_M = 1.6868513


def default_scene() -> SceneProfile:
    return SceneProfile(scene_id="test-scene", point_cloud_asset="/static/assets/bunny.pcd")


def test_default_scene_matches_the_documented_physical_layout() -> None:
    scene = default_scene()
    assert scene.anchor_display_m == (0.0, 0.0, 0.0)
    assert scene.longest_edge_m == pytest.approx(0.24)
    assert scene.grid_spacing_m == pytest.approx(0.05)
    assert scene.back_wall_z_m == pytest.approx(-0.30)
    assert scene.floor_y_m == pytest.approx(-0.14)
    assert scene.floor_near_z_m == pytest.approx(0.05)
    assert scene.floor_far_z_m == pytest.approx(-0.30)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SceneProfile(
            scene_id="typo",
            point_cloud_asset="/static/assets/bunny.pcd",
            grid_spacing_metres=0.05,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("spacing", [0.0, -0.05])
def test_non_positive_grid_spacing_is_rejected(spacing: float) -> None:
    with pytest.raises(ValidationError):
        SceneProfile(
            scene_id="bad-grid",
            point_cloud_asset="/static/assets/bunny.pcd",
            grid_spacing_m=spacing,
        )


@pytest.mark.parametrize("z", [0.0, 0.05])
def test_back_wall_on_the_observer_side_is_rejected(z: float) -> None:
    """The back wall is a depth cue behind the window; +Z is the observer side."""

    with pytest.raises(ValidationError):
        SceneProfile(
            scene_id="wall-in-front",
            point_cloud_asset="/static/assets/bunny.pcd",
            back_wall_z_m=z,
        )


def test_floor_must_recede_away_from_the_observer() -> None:
    with pytest.raises(ValidationError):
        SceneProfile(
            scene_id="inverted-floor",
            point_cloud_asset="/static/assets/bunny.pcd",
            floor_near_z_m=-0.30,
            floor_far_z_m=0.05,
        )


def test_floor_may_not_start_behind_the_back_wall() -> None:
    with pytest.raises(ValidationError):
        SceneProfile(
            scene_id="floor-past-wall",
            point_cloud_asset="/static/assets/bunny.pcd",
            back_wall_z_m=-0.30,
            floor_far_z_m=-0.40,
        )


@pytest.mark.parametrize("edge", [0.0, -0.24])
def test_non_positive_longest_edge_is_rejected(edge: float) -> None:
    with pytest.raises(ValidationError):
        SceneProfile(
            scene_id="bad-scale",
            point_cloud_asset="/static/assets/bunny.pcd",
            longest_edge_m=edge,
        )


def test_scene_round_trips_through_json(tmp_path: Path) -> None:
    scene = default_scene()
    path = tmp_path / "scene.json"
    scene.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["scene_id"] == "test-scene"
    assert SceneProfile.load(path) == scene


def test_model_matrix_centres_the_bunny_aabb_on_the_display_plane() -> None:
    """``T(anchor) @ S @ T(-aabb_center)`` must put the AABB midpoint exactly at the anchor."""

    scene = default_scene()
    matrix = scene_model_matrix(scene, bounds_min=BUNNY_MIN, bounds_max=BUNNY_MAX)

    corners = np.array(
        [
            [x, y, z]
            for x in (BUNNY_MIN[0], BUNNY_MAX[0])
            for y in (BUNNY_MIN[1], BUNNY_MAX[1])
            for z in (BUNNY_MIN[2], BUNNY_MAX[2])
        ],
        dtype=np.float64,
    )
    homogeneous = np.hstack([corners, np.ones((corners.shape[0], 1))])
    transformed = (matrix @ homogeneous.T).T[:, :3]

    centre = 0.5 * (transformed.min(axis=0) + transformed.max(axis=0))
    assert centre == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_model_matrix_scales_the_longest_edge_to_the_configured_size() -> None:
    scene = default_scene()
    matrix = scene_model_matrix(scene, bounds_min=BUNNY_MIN, bounds_max=BUNNY_MAX)
    scale = float(matrix[0, 0])

    assert scale == pytest.approx(scene.longest_edge_m / BUNNY_LONGEST_EDGE_M)
    assert matrix[1, 1] == pytest.approx(scale)
    assert matrix[2, 2] == pytest.approx(scale)

    span = (np.array(BUNNY_MAX) - np.array(BUNNY_MIN)) * scale
    assert span.max() == pytest.approx(scene.longest_edge_m)


def test_model_matrix_honours_a_non_zero_anchor() -> None:
    scene = SceneProfile(
        scene_id="offset",
        point_cloud_asset="/static/assets/bunny.pcd",
        anchor_display_m=(0.10, -0.02, -0.05),
    )
    matrix = scene_model_matrix(scene, bounds_min=BUNNY_MIN, bounds_max=BUNNY_MAX)
    centre = matrix @ np.array([*BUNNY_CENTER, 1.0])

    assert centre[:3] == pytest.approx(scene.anchor_display_m, abs=1e-9)


def test_model_matrix_rejects_a_degenerate_bounding_box() -> None:
    scene = default_scene()
    with pytest.raises(ValueError):
        scene_model_matrix(scene, bounds_min=(0.0, 0.0, 0.0), bounds_max=(0.0, 0.0, 0.0))


def test_transformed_bunny_straddles_the_display_plane() -> None:
    """The 0.24 m scene deliberately pokes through the window so front and back
    surfaces show opposite parallax for the same object."""

    scene = default_scene()
    matrix = scene_model_matrix(scene, bounds_min=BUNNY_MIN, bounds_max=BUNNY_MAX)
    near = matrix @ np.array([0.0, 0.0, BUNNY_MAX[2], 1.0])
    far = matrix @ np.array([0.0, 0.0, BUNNY_MIN[2], 1.0])

    assert near[2] > 0.0
    assert far[2] < 0.0
