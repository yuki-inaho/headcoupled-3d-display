from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from headcoupled_display.api import create_app

ROOT = Path(__file__).resolve().parents[2]


def make_client() -> TestClient:
    app = create_app(
        profile_path=ROOT / "config" / "hardware_profile.demo.json",
        user_profile_path=ROOT / "config" / "user_profile.demo.json",
        source="synthetic",
    )
    return TestClient(app)


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
