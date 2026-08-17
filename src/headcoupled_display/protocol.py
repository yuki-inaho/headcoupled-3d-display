"""Fixed-length binary control-lane packet: latest FaceMesh landmark subset + timing.

This module is intentionally dependency-free (``struct`` and the standard library
only, no numpy/pydantic/etc.) because it is imported from both the display
controller (Python 3.13) and the FaceMesh producer process (Python 3.10) over a
plain ``PYTHONPATH`` import, without pulling in either side's heavier tracking
stack. It must not use syntax newer than Python 3.10.

Packets are ``struct``-packed with the ``"!"`` prefix (network byte order,
standard sizes, no native alignment padding), never pickled -- pickle allows
arbitrary code execution on unpickling and must not be used for data crossing a
process boundary.

Landmarks and score travel as float32, so a decoded packet is generally *not*
equal to the ``ControlPacket`` that produced it: 0.9 and most other Python floats
have no exact float32 representation. Compare decoded values with a tolerance;
never use packet equality to decide whether two samples are the same frame (use
``sequence`` for that).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

#: 4-byte ASCII tag identifying a Head-Coupled control-lane packet on the wire.
#: Picked to be unlikely to collide with JSON (``{``), JPEG (``\xff\xd8``), or
#: other payloads that might land on the same socket/queue.
PROTOCOL_MAGIC: bytes = b"HC3D"

#: Current wire-format version. decode_control_packet() rejects anything else,
#: including *future* versions -- this format never pretends to be forward
#: compatible with a version it has not been updated to understand.
PROTOCOL_VERSION: int = 1

#: Number of 2D landmarks carried in a control packet.
NUM_LANDMARKS: int = 12

#: FaceMesh landmark indices carried in the packet, in the fixed order their
#: (u, v) pairs are packed. Mirrors ``HeadPoseEstimator.LANDMARK_INDICES`` in
#: tracking.py; test_protocol.py asserts the two stay in sync. Duplicated here
#: (rather than imported) to keep this module free of the tracking stack's
#: cv2/numpy dependencies.
LANDMARK_INDICES: tuple[int, ...] = (1, 6, 33, 133, 362, 263, 61, 291, 199, 168, 94, 4)

_LANDMARK_FLOAT_COUNT = NUM_LANDMARKS * 2

# Wire layout, in order:
#   magic                     4s   ASCII tag, see PROTOCOL_MAGIC
#   version                   H    uint16, see PROTOCOL_VERSION
#   landmarks[NUM_LANDMARKS]  24f  float32 u0,v0, u1,v1, ..., u11,v11 (image px)
#   score                     f    float32
#   sequence                  Q    uint64, monotonically increasing
#   capture_monotonic_ns      Q    uint64, time.perf_counter_ns() clock domain
#   capture_unix_ns           Q    uint64, time.time_ns() clock domain
#   inference_monotonic_ns    Q    uint64, time.perf_counter_ns() clock domain
#   inference_unix_ns         Q    uint64, time.time_ns() clock domain
PACKET_FORMAT: str = f"!4sH{_LANDMARK_FLOAT_COUNT}ffQQQQQ"
PACKET_SIZE: int = struct.calcsize(PACKET_FORMAT)

# Slice boundaries into the tuple returned by struct.unpack(PACKET_FORMAT, ...).
_LANDMARKS_START = 2
_LANDMARKS_END = _LANDMARKS_START + _LANDMARK_FLOAT_COUNT


class ProtocolError(ValueError):
    """Raised when a control packet fails validation during construction or decode."""


@dataclass(frozen=True, slots=True)
class ControlPacket:
    """One control-lane sample: 12 image-space landmarks plus capture/inference timing.

    Each timing pair spans two independent clock domains that must never be mixed:

    * ``*_monotonic_ns`` -- a ``time.perf_counter_ns()``-compatible count. Monotonic
      and immune to wall-clock adjustments, but its epoch is arbitrary and only
      meaningful *within a single process's lifetime*.
    * ``*_unix_ns`` -- a ``time.time_ns()``-compatible Unix epoch timestamp.
      Comparable across processes/machines (modulo clock sync), but is NOT
      monotonic -- NTP adjustments can step it backwards.

    Latency must be computed within one domain, e.g.
    ``inference_monotonic_ns - capture_monotonic_ns``. Do not subtract a
    ``*_monotonic_ns`` value from a ``*_unix_ns`` value (or vice versa); the two
    domains have no fixed relationship to each other, even within the same process.
    """

    landmarks_px: tuple[tuple[float, float], ...]
    score: float
    sequence: int
    capture_monotonic_ns: int
    capture_unix_ns: int
    inference_monotonic_ns: int
    inference_unix_ns: int

    def __post_init__(self) -> None:
        if len(self.landmarks_px) != NUM_LANDMARKS:
            raise ProtocolError(f"expected {NUM_LANDMARKS} landmarks, got {len(self.landmarks_px)}")
        coordinates = [value for pair in self.landmarks_px for value in pair]
        if not all(math.isfinite(value) for value in (*coordinates, self.score)):
            raise ProtocolError("landmark coordinates and score must be finite (no NaN/Inf)")


def encode_control_packet(packet: ControlPacket) -> bytes:
    """Serialize a validated ``ControlPacket`` into a fixed ``PACKET_SIZE``-byte payload."""

    flat_landmarks = [value for pair in packet.landmarks_px for value in pair]
    return struct.pack(
        PACKET_FORMAT,
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        *flat_landmarks,
        packet.score,
        packet.sequence,
        packet.capture_monotonic_ns,
        packet.capture_unix_ns,
        packet.inference_monotonic_ns,
        packet.inference_unix_ns,
    )


def decode_control_packet(data: bytes) -> ControlPacket:
    """Deserialize and validate a wire-format control packet.

    Raises ``ProtocolError`` if ``data`` is truncated or oversized (the packet is
    fixed-length), the magic tag does not match ``PROTOCOL_MAGIC``, the version is
    not exactly ``PROTOCOL_VERSION`` (future versions are rejected outright, not
    treated as forward compatible), or any decoded coordinate/score is NaN/Inf.
    """

    if len(data) != PACKET_SIZE:
        raise ProtocolError(f"expected a {PACKET_SIZE}-byte packet, got {len(data)} bytes")

    fields = struct.unpack(PACKET_FORMAT, data)
    magic, version = fields[0], fields[1]
    if magic != PROTOCOL_MAGIC:
        raise ProtocolError(f"bad magic {magic!r}, expected {PROTOCOL_MAGIC!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {version}; only {PROTOCOL_VERSION} is accepted"
        )

    landmark_values = fields[_LANDMARKS_START:_LANDMARKS_END]
    landmarks_px = tuple(
        (landmark_values[i], landmark_values[i + 1]) for i in range(0, len(landmark_values), 2)
    )
    (
        score,
        sequence,
        capture_monotonic_ns,
        capture_unix_ns,
        inference_monotonic_ns,
        inference_unix_ns,
    ) = fields[_LANDMARKS_END:]

    return ControlPacket(
        landmarks_px=landmarks_px,
        score=score,
        sequence=sequence,
        capture_monotonic_ns=capture_monotonic_ns,
        capture_unix_ns=capture_unix_ns,
        inference_monotonic_ns=inference_monotonic_ns,
        inference_unix_ns=inference_unix_ns,
    )
