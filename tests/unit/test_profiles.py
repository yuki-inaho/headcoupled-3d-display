from __future__ import annotations

import json
from pathlib import Path

import pytest

from headcoupled_display.profiles import load_tagcal_calibration, load_user_profile


def test_load_tagcal_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "image_width": 1280,
                "image_height": 720,
                "camera_matrix": [[901.0, 0.0, 640.0], [0.0, 903.0, 360.0], [0.0, 0.0, 1.0]],
                "distortion_coefficients": [-0.02, 0.01, 0.0, 0.0, -0.003],
                "distortion_model": "plumb_bob",
                "rms_reprojection_error_px": 0.21,
            }
        ),
        encoding="utf-8",
    )
    intrinsics = load_tagcal_calibration(path)
    assert intrinsics.image_width_px == 1280
    assert intrinsics.camera_matrix[0][0] == pytest.approx(901.0)
    assert intrinsics.rms_reprojection_error_px == pytest.approx(0.21)


def test_user_profile_resolves_personal_mesh_relative_to_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "user.json"
    profile_path.write_text('{"face_model_path": "models/shape.pcd"}', encoding="utf-8")

    profile = load_user_profile(profile_path)

    assert profile.face_model_path == str((tmp_path / "models" / "shape.pcd").resolve())
