"""RED/GREEN tests for the binary control-lane packet protocol (workdoc step 35).

Covers: encode/decode round trip, magic/version/truncation/NaN rejection, and the
landmark-index-order cross-check against ``HeadPoseEstimator.LANDMARK_INDICES``.
"""

from __future__ import annotations

import math
import struct

import pytest

from headcoupled_display.protocol import (
    LANDMARK_INDICES,
    NUM_LANDMARKS,
    PACKET_FORMAT,
    PACKET_SIZE,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ControlPacket,
    ProtocolError,
    decode_control_packet,
    encode_control_packet,
)


def _sample_landmarks() -> tuple[tuple[float, float], ...]:
    return tuple((100.0 + 10.0 * i, 200.0 + 5.0 * i) for i in range(NUM_LANDMARKS))


def _sample_packet(**overrides: object) -> ControlPacket:
    fields: dict[str, object] = {
        "landmarks_px": _sample_landmarks(),
        "score": 2.71828,
        "sequence": 42,
        "capture_monotonic_ns": 1_000_000_000,
        "capture_unix_ns": 1_755_000_000_000_000_000,
        "inference_monotonic_ns": 1_000_004_500,
        "inference_unix_ns": 1_755_000_000_004_500_000,
    }
    fields.update(overrides)
    return ControlPacket(**fields)  # type: ignore[arg-type]


def _pack_raw(
    *,
    magic: bytes = PROTOCOL_MAGIC,
    version: int = PROTOCOL_VERSION,
    landmarks: tuple[float, ...] | None = None,
    score: float = 0.5,
    sequence: int = 1,
    capture_monotonic_ns: int = 1,
    capture_unix_ns: int = 1,
    inference_monotonic_ns: int = 2,
    inference_unix_ns: int = 2,
) -> bytes:
    """Pack raw wire bytes with ``struct`` directly, bypassing ControlPacket validation.

    Used to construct deliberately malformed packets (bad magic/version/NaN) that the
    validated ControlPacket constructor would otherwise refuse to build.
    """

    flat_landmarks = list(landmarks) if landmarks is not None else [0.0] * (NUM_LANDMARKS * 2)
    return struct.pack(
        PACKET_FORMAT,
        magic,
        version,
        *flat_landmarks,
        score,
        sequence,
        capture_monotonic_ns,
        capture_unix_ns,
        inference_monotonic_ns,
        inference_unix_ns,
    )


def test_encode_decode_round_trip_preserves_all_fields() -> None:
    packet = _sample_packet()
    data = encode_control_packet(packet)
    decoded = decode_control_packet(data)

    decoded_flat = [value for pair in decoded.landmarks_px for value in pair]
    expected_flat = [value for pair in packet.landmarks_px for value in pair]
    assert decoded_flat == pytest.approx(expected_flat)
    assert decoded.score == pytest.approx(packet.score)
    assert decoded.sequence == packet.sequence
    assert decoded.capture_monotonic_ns == packet.capture_monotonic_ns
    assert decoded.capture_unix_ns == packet.capture_unix_ns
    assert decoded.inference_monotonic_ns == packet.inference_monotonic_ns
    assert decoded.inference_unix_ns == packet.inference_unix_ns


def test_packet_size_matches_struct_calcsize() -> None:
    assert struct.calcsize(PACKET_FORMAT) == PACKET_SIZE
    assert len(encode_control_packet(_sample_packet())) == PACKET_SIZE


def test_decode_rejects_wrong_magic() -> None:
    data = _pack_raw(magic=b"XXXX")
    with pytest.raises(ProtocolError):
        decode_control_packet(data)


def test_decode_rejects_future_version() -> None:
    data = _pack_raw(version=PROTOCOL_VERSION + 1)
    with pytest.raises(ProtocolError):
        decode_control_packet(data)


def test_decode_rejects_truncated_packet() -> None:
    data = _pack_raw()[: PACKET_SIZE - 1]
    with pytest.raises(ProtocolError):
        decode_control_packet(data)


def test_decode_rejects_oversized_packet() -> None:
    data = _pack_raw() + b"\x00"
    with pytest.raises(ProtocolError):
        decode_control_packet(data)


def test_decode_rejects_nan_landmark_coordinate() -> None:
    landmarks = [0.0] * (NUM_LANDMARKS * 2)
    landmarks[0] = math.nan
    data = _pack_raw(landmarks=tuple(landmarks))
    with pytest.raises(ProtocolError):
        decode_control_packet(data)


def test_decode_rejects_nan_score() -> None:
    data = _pack_raw(score=math.nan)
    with pytest.raises(ProtocolError):
        decode_control_packet(data)


def test_construction_rejects_nan_landmark_coordinate() -> None:
    landmarks = list(_sample_landmarks())
    landmarks[3] = (landmarks[3][0], math.nan)
    with pytest.raises(ProtocolError):
        _sample_packet(landmarks_px=tuple(landmarks))


def test_construction_rejects_nan_score() -> None:
    with pytest.raises(ProtocolError):
        _sample_packet(score=math.nan)


def test_construction_rejects_wrong_landmark_count() -> None:
    with pytest.raises(ProtocolError):
        _sample_packet(landmarks_px=_sample_landmarks()[:-1])


def test_landmark_indices_match_head_pose_estimator() -> None:
    """The wire order must never silently drift from the PnP estimator's order."""

    from headcoupled_display.tracking import HeadPoseEstimator

    assert tuple(int(index) for index in HeadPoseEstimator.LANDMARK_INDICES) == LANDMARK_INDICES
    assert len(LANDMARK_INDICES) == NUM_LANDMARKS
