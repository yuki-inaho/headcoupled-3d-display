from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from headcoupled_display.api import create_app
from headcoupled_display.face_model import canonical_face_model
from headcoupled_display.models import HardwareProfile
from headcoupled_display.tracking import HeadPoseEstimator

ROOT = Path(__file__).resolve().parents[2]


def make_client() -> TestClient:
    app = create_app(
        profile_path=ROOT / "config" / "hardware_profile.demo.json",
        user_profile_path=ROOT / "config" / "user_profile.demo.json",
        scene_path=ROOT / "config" / "scene_profile.default.json",
        source="synthetic",
    )
    return TestClient(app)


def test_profile_endpoint_exposes_the_scene_separately_from_calibration() -> None:
    with make_client() as client:
        payload = client.get("/api/profile").json()
        scene = payload["scene_profile"]

        assert scene["point_cloud_asset"] == "/static/assets/bunny.pcd"
        assert scene["anchor_display_m"] == [0.0, 0.0, 0.0]
        assert scene["longest_edge_m"] == 0.24
        assert scene["grid_spacing_m"] == 0.05
        assert scene["back_wall_z_m"] == -0.3
        assert scene["floor_y_m"] == -0.14
        # The scene must not leak into, or be derived from, the calibrated mount geometry.
        assert "longest_edge_m" not in payload["hardware_profile"]
        assert payload["mount_summary"]["height_above_center_cm"] == 20.0


def test_scene_profile_with_a_back_wall_in_front_of_the_screen_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "scene.json"
    broken.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": "wall-in-front",
                "point_cloud_asset": "/static/assets/bunny.pcd",
                "back_wall_z_m": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        create_app(
            profile_path=ROOT / "config" / "hardware_profile.demo.json",
            user_profile_path=ROOT / "config" / "user_profile.demo.json",
            scene_path=broken,
            source="synthetic",
        )


def test_profile_and_runtime_endpoints() -> None:
    with make_client() as client:
        profile = client.get("/api/profile")
        assert profile.status_code == 200
        payload = profile.json()
        assert payload["mount_summary"]["height_above_center_cm"] == 20.0
        assert payload["mount_summary"]["pitch_down_deg"] == 10.0
        assert payload["mount_summary"]["horizontally_centered"] is True
        assert payload["warning"] is not None

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["runtime"]["running"] is True


def test_pose_and_camera_websockets() -> None:
    with make_client() as client:
        with client.websocket_connect("/ws/pose") as socket:
            message = socket.receive_json()
            assert message["type"] == "tracking"
            assert message["payload"]["source"] == "synthetic"
            assert len(message["payload"]["cyclopean_eye_display_m"]) == 3

        with client.websocket_connect("/ws/camera") as socket:
            frame = socket.receive_bytes()
            assert frame.startswith(b"\xff\xd8")
            assert len(frame) > 1000


def test_synthetic_calibration_endpoint() -> None:
    with make_client() as client:
        response = client.post("/api/calibration/synthetic")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["dataset"]["sample_count"] == 36
        comparison = payload["result"]["comparison_to_ground_truth"]
        assert comparison["height_error_mm"] < 0.5
        assert comparison["pitch_error_deg"] < 0.35


def test_latest_jpeg_endpoint_becomes_ready() -> None:
    with make_client() as client:
        with client.websocket_connect("/ws/camera") as socket:
            frame = socket.receive_bytes()
            assert frame.startswith(b"\xff\xd8")
        response = client.get("/api/frame.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


def test_ipc_frame_pair_drives_pose_and_camera_streams() -> None:
    hardware = HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")
    indices = HeadPoseEstimator.LANDMARK_INDICES
    projected, _ = cv2.projectPoints(
        canonical_face_model().pnp_points_opencv_m[indices],
        np.array([[0.02], [-0.03], [0.01]], dtype=np.float64),
        np.array([[0.01], [0.0], [0.70]], dtype=np.float64),
        np.asarray(hardware.camera.camera_matrix, dtype=np.float64),
        np.asarray(hardware.camera.distortion_coefficients, dtype=np.float64),
    )
    landmarks = np.zeros((478, 3), dtype=np.float64)
    landmarks[indices, :2] = projected.reshape(-1, 2)
    encoded_ok, encoded = cv2.imencode(".jpg", np.full((720, 1280, 3), 64, dtype=np.uint8))
    assert encoded_ok
    app = create_app(
        profile_path=ROOT / "config" / "hardware_profile.demo.json",
        user_profile_path=ROOT / "config" / "user_profile.demo.json",
        source="ipc",
    )
    payload = {
        "frame_index": 7,
        "faces": [{"score": 0.99, "landmarks": landmarks.tolist()}],
        "frame_jpeg_base64": base64.b64encode(encoded).decode("ascii"),
    }

    with TestClient(app) as client:
        accepted = client.post("/api/input/facemesh", json=payload)
        assert accepted.status_code == 202
        assert accepted.json()["accepted_version"] == 1
        with client.websocket_connect("/ws/pose") as socket:
            message = socket.receive_json()
            assert message["payload"]["source"] == "ipc"
            assert message["payload"]["diagnostics"]["input_frame"] == 7
        with client.websocket_connect("/ws/camera") as socket:
            assert socket.receive_bytes().startswith(b"\xff\xd8")
