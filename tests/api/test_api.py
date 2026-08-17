from __future__ import annotations

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
from headcoupled_display.protocol import ControlPacket, encode_control_packet
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


def _make_ipc_app(*, source: str = "ipc"):
    return create_app(
        profile_path=ROOT / "config" / "hardware_profile.demo.json",
        user_profile_path=ROOT / "config" / "user_profile.demo.json",
        scene_path=ROOT / "config" / "scene_profile.default.json",
        source=source,
    )


def _sample_control_packet_bytes(*, sequence: int, **overrides: object) -> bytes:
    """A control packet whose 12 points actually reproject to a convergeable pose,
    reusing the same synthetic projection the old single-lane IPC test used."""
    hardware = HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")
    indices = HeadPoseEstimator.LANDMARK_INDICES
    projected, _ = cv2.projectPoints(
        canonical_face_model().pnp_points_opencv_m[indices],
        np.array([[0.02], [-0.03], [0.01]], dtype=np.float64),
        np.array([[0.01], [0.0], [0.70]], dtype=np.float64),
        np.asarray(hardware.camera.camera_matrix, dtype=np.float64),
        np.asarray(hardware.camera.distortion_coefficients, dtype=np.float64),
    )
    landmarks_px = tuple((float(x), float(y)) for x, y in projected.reshape(-1, 2))
    fields: dict[str, object] = {
        "landmarks_px": landmarks_px,
        "score": 0.99,
        "sequence": sequence,
        "capture_monotonic_ns": 1_000,
        "capture_unix_ns": 1_755_000_000_000_000_000,
        "inference_monotonic_ns": 1_010,
        "inference_unix_ns": 1_755_000_000_010_000_000,
    }
    fields.update(overrides)
    return encode_control_packet(ControlPacket(**fields))  # type: ignore[arg-type]


def _preview_jpeg_bytes(width: int, height: int) -> bytes:
    encoded_ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), 64, dtype=np.uint8))
    assert encoded_ok
    return encoded.tobytes()


def test_ipc_control_endpoint_accepts_a_packet_and_returns_its_sequence() -> None:
    with TestClient(_make_ipc_app()) as client:
        response = client.post(
            "/api/input/facemesh/control",
            content=_sample_control_packet_bytes(sequence=7),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200
        assert response.json()["accepted_sequence"] == 7


def test_ws_pose_reports_ipc_source_after_a_control_publish() -> None:
    with TestClient(_make_ipc_app()) as client:
        client.post(
            "/api/input/facemesh/control",
            content=_sample_control_packet_bytes(sequence=7),
            headers={"Content-Type": "application/octet-stream"},
        )
        with client.websocket_connect("/ws/pose") as socket:
            message = socket.receive_json()
            assert message["payload"]["source"] == "ipc"
            assert message["payload"]["diagnostics"]["input_frame"] == 7


def test_ipc_preview_endpoint_forwards_bytes_unchanged_to_ws_camera() -> None:
    preview = _preview_jpeg_bytes(640, 360)
    with TestClient(_make_ipc_app()) as client:
        # Preview before control, deliberately: the background runtime loop only reads
        # /ws/camera's next frame on a *control* update (see IpcFaceMeshInput.next_frame).
        # Publishing the preview first guarantees it is already the latest preview by the
        # time our control publish is the one the loop observes, instead of racing a poll
        # that might consume our control before this preview POST has landed.
        response = client.post(
            "/api/input/facemesh/preview", content=preview, headers={"Content-Type": "image/jpeg"}
        )
        assert response.status_code == 204
        client.post(
            "/api/input/facemesh/control",
            content=_sample_control_packet_bytes(sequence=1),
            headers={"Content-Type": "application/octet-stream"},
        )
        with client.websocket_connect("/ws/camera") as socket:
            assert socket.receive_bytes() == preview


def test_ipc_preview_endpoint_rejects_the_wrong_resolution() -> None:
    with TestClient(_make_ipc_app()) as client:
        response = client.post(
            "/api/input/facemesh/preview",
            content=_preview_jpeg_bytes(1280, 720),
            headers={"Content-Type": "image/jpeg"},
        )
        assert response.status_code == 422


def test_ipc_control_endpoint_rejects_a_malformed_packet() -> None:
    with TestClient(_make_ipc_app()) as client:
        response = client.post(
            "/api/input/facemesh/control",
            content=b"not a control packet",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 422


def test_ipc_endpoints_are_refused_unless_the_server_was_started_with_source_ipc() -> None:
    with TestClient(_make_ipc_app(source="synthetic")) as client:
        control_response = client.post(
            "/api/input/facemesh/control",
            content=_sample_control_packet_bytes(sequence=1),
            headers={"Content-Type": "application/octet-stream"},
        )
        preview_response = client.post(
            "/api/input/facemesh/preview",
            content=_preview_jpeg_bytes(640, 360),
            headers={"Content-Type": "image/jpeg"},
        )
        assert control_response.status_code == 409
        assert preview_response.status_code == 409


def test_ipc_provider_never_re_encodes_a_forwarded_preview() -> None:
    preview = _preview_jpeg_bytes(640, 360)
    app = _make_ipc_app()
    with TestClient(app) as client:
        # Preview before control -- see the comment in
        # test_ipc_preview_endpoint_forwards_bytes_unchanged_to_ws_camera for why the
        # order matters here.
        client.post(
            "/api/input/facemesh/preview", content=preview, headers={"Content-Type": "image/jpeg"}
        )
        client.post(
            "/api/input/facemesh/control",
            content=_sample_control_packet_bytes(sequence=1),
            headers={"Content-Type": "application/octet-stream"},
        )
        with client.websocket_connect("/ws/camera") as socket:
            assert socket.receive_bytes() == preview
        # No public accessor exists for the running provider instance; reach into the
        # coordinator's own reference rather than construct a second provider that would
        # not reflect what the app actually ran (workdoc step 38, DoD item 14).
        provider = app.state.runtime._provider
        assert provider.preview_encode_count == 0
