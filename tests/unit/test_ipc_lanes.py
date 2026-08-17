"""RED/GREEN tests for the two-lane IPC input port (workdoc steps 36-38).

Covers: control-only progress without a preview, stale-preview reuse across control
updates, no-backlog control publishing, protocol/dimension rejection, and the absolute
absence of a JPEG-decode call anywhere in tracking.py's source.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from headcoupled_display import tracking
from headcoupled_display.face_model import canonical_face_model
from headcoupled_display.models import HardwareProfile, UserProfile
from headcoupled_display.protocol import (
    NUM_LANDMARKS,
    ControlPacket,
    ProtocolError,
    encode_control_packet,
)
from headcoupled_display.synthetic import SyntheticTrackingProvider
from headcoupled_display.tracking import (
    FaceMeshPoseProvider,
    HeadPoseEstimator,
    IpcFaceMeshInput,
)

ROOT = Path(__file__).resolve().parents[2]


def _hardware() -> HardwareProfile:
    return HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")


def _control_bytes(sequence: int, **overrides: object) -> bytes:
    fields: dict[str, object] = {
        "landmarks_px": tuple((100.0 + 10.0 * i, 200.0 + 5.0 * i) for i in range(NUM_LANDMARKS)),
        "score": 0.9,
        "sequence": sequence,
        "capture_monotonic_ns": 1_000 + sequence,
        "capture_unix_ns": 1_755_000_000_000_000_000 + sequence,
        "inference_monotonic_ns": 1_010 + sequence,
        "inference_unix_ns": 1_755_000_000_010_000_000 + sequence,
    }
    fields.update(overrides)
    return encode_control_packet(ControlPacket(**fields))  # type: ignore[arg-type]


def _convergent_control_bytes(
    hardware: HardwareProfile, sequence: int, **overrides: object
) -> bytes:
    """A control packet whose 12 points reproject from a real pose, so
    ``HeadPoseEstimator.estimate`` (called by ``FaceMeshPoseProvider.sample``) actually
    converges -- unlike ``_control_bytes``'s placeholder grid, which is fine for
    ``IpcFaceMeshInput``-only tests that never reach PnP."""
    indices = HeadPoseEstimator.LANDMARK_INDICES
    projected, _ = cv2.projectPoints(
        canonical_face_model().pnp_points_opencv_m[indices],
        np.array([[0.02], [-0.03], [0.01]], dtype=np.float64),
        np.array([[0.01], [0.0], [0.70]], dtype=np.float64),
        np.asarray(hardware.camera.camera_matrix, dtype=np.float64),
        np.asarray(hardware.camera.distortion_coefficients, dtype=np.float64),
    )
    fields: dict[str, object] = {
        "landmarks_px": tuple((float(x), float(y)) for x, y in projected.reshape(-1, 2)),
        "score": 0.9,
        "sequence": sequence,
        "capture_monotonic_ns": 1_000 + sequence,
        "capture_unix_ns": 1_755_000_000_000_000_000 + sequence,
        "inference_monotonic_ns": 1_010 + sequence,
        "inference_unix_ns": 1_755_000_000_010_000_000 + sequence,
    }
    fields.update(overrides)
    return encode_control_packet(ControlPacket(**fields))  # type: ignore[arg-type]


def _preview_jpeg(width: int, height: int) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_publish_control_then_next_frame_has_no_raw_pixels_and_no_preview_yet() -> None:
    port = IpcFaceMeshInput(_hardware())

    port.publish_control(_control_bytes(sequence=5))
    frame = port.next_frame()

    assert frame.frame_bgr is None
    assert frame.preview_jpeg is None
    assert frame.frame_index == 5
    assert len(frame.faces) == 1


def test_control_progresses_even_though_preview_is_never_published() -> None:
    port = IpcFaceMeshInput(_hardware())

    for sequence in range(1, 4):
        port.publish_control(_control_bytes(sequence=sequence))
        frame = port.next_frame()
        assert frame.frame_index == sequence
        assert frame.preview_jpeg is None


def test_stale_preview_is_reused_across_multiple_control_updates() -> None:
    port = IpcFaceMeshInput(_hardware())
    preview = _preview_jpeg(640, 360)
    port.publish_preview(preview)

    for sequence in (1, 2, 3):
        port.publish_control(_control_bytes(sequence=sequence))
        frame = port.next_frame()
        assert frame.preview_jpeg == preview


def test_control_publish_never_accumulates_a_backlog() -> None:
    port = IpcFaceMeshInput(_hardware())

    for sequence in range(1, 101):
        port.publish_control(_control_bytes(sequence=sequence))
    frame = port.next_frame()

    assert frame.frame_index == 100


def test_publish_control_rejects_a_malformed_packet() -> None:
    port = IpcFaceMeshInput(_hardware())

    with pytest.raises(ProtocolError):
        port.publish_control(b"not a valid control packet")


def test_publish_preview_rejects_the_wrong_resolution() -> None:
    port = IpcFaceMeshInput(_hardware())

    with pytest.raises(ValueError):
        port.publish_preview(_preview_jpeg(1280, 720))


def test_tracking_module_source_never_decodes_a_jpeg() -> None:
    source = inspect.getsource(tracking)
    assert "imdecode" not in source


# --- Control packet timestamps -> TrackingState.diagnostics (team-lead follow-up) ----


def test_control_packet_inference_unix_ns_reaches_pose_diagnostics() -> None:
    """This is what /ws/pose actually serializes: TrackingState.diagnostics."""
    hardware = _hardware()
    port = IpcFaceMeshInput(hardware)
    provider = FaceMeshPoseProvider(hardware, UserProfile(), source="ipc", frame_source=port)

    port.publish_control(
        _convergent_control_bytes(hardware, sequence=1, inference_unix_ns=1_755_000_000_010_000_001)
    )
    state, _frame = provider.sample()

    assert state.diagnostics["producer_inference_unix_ns"] == 1_755_000_000_010_000_001


def test_source_timestamp_keys_separate_monotonic_and_unix_clock_domains() -> None:
    hardware = _hardware()
    port = IpcFaceMeshInput(hardware)
    provider = FaceMeshPoseProvider(hardware, UserProfile(), source="ipc", frame_source=port)

    port.publish_control(_convergent_control_bytes(hardware, sequence=1))
    state, _frame = provider.sample()

    monotonic_keys = {
        key
        for key in state.diagnostics
        if key.startswith("producer_") and key.endswith("_monotonic_ns")
    }
    unix_keys = {
        key for key in state.diagnostics if key.startswith("producer_") and key.endswith("_unix_ns")
    }
    assert monotonic_keys == {"producer_capture_monotonic_ns", "producer_inference_monotonic_ns"}
    assert unix_keys == {"producer_capture_unix_ns", "producer_inference_unix_ns"}
    assert monotonic_keys.isdisjoint(unix_keys)


def test_synthetic_source_diagnostics_never_gain_producer_timestamp_keys() -> None:
    provider = SyntheticTrackingProvider(_hardware())

    state, _frame = provider.sample()

    assert not any(key.startswith("producer_") for key in state.diagnostics)
