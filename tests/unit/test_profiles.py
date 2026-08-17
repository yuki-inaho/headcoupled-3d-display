from __future__ import annotations

import json
from pathlib import Path

import pytest

from headcoupled_display.models import HardwareProfile
from headcoupled_display.profiles import (
    load_tagcal_calibration,
    load_user_profile,
    summarize_profile,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_local_profile_matches_confirmed_camera_mount() -> None:
    """The local profile's *derived* mount geometry, not its raw JSON fields, is the contract.

    ``summarize_profile`` recovers the mount from the resolved 4x4 camera-to-display
    transform (see README's "settings vs. derived summary" policy), so this test guards
    against a transform that silently drifts from the confirmed on-hardware mount values.
    """

    profile = HardwareProfile.load(ROOT / "config" / "hardware_profile.local.json")
    summary = summarize_profile(profile)

    assert summary.horizontally_centered is True
    assert summary.height_above_center_cm == pytest.approx(15.0, abs=1e-9)
    assert summary.forward_offset_cm == pytest.approx(0.0, abs=1e-9)
    assert summary.pitch_down_deg == pytest.approx(12.0, abs=1e-9)
    assert summary.yaw_right_deg == pytest.approx(0.0, abs=1e-9)
    assert summary.roll_clockwise_deg == pytest.approx(0.0, abs=1e-9)


def test_local_profile_provenance_is_distinct_from_demo_and_not_measured() -> None:
    """local profile must not claim to be the artificial demo, nor a fully measured rig.

    Camera intrinsics (K, D) are still the demo placeholder pending a tagcal replacement,
    so provenance must not lie by claiming "measured".
    """

    demo = HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")
    local = HardwareProfile.load(ROOT / "config" / "hardware_profile.local.json")

    assert local.provenance != demo.provenance
    assert local.provenance != "measured"
