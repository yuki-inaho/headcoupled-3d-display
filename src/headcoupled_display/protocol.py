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


#: Tag for a dense-mesh packet. Distinct from PROTOCOL_MAGIC so a mesh payload delivered
#: to the control endpoint by mistake is rejected rather than half-parsed.
MESH_MAGIC: bytes = b"HC3M"

#: Wire-format version for the mesh lane, versioned independently of the control lane:
#: the two carry different things and there is no reason to force them to move together.
MESH_VERSION: int = 1

#: MediaPipe ships a 468-point mesh and a 478-point one that appends 10 iris points. The
#: first 468 indices mean the same thing in both, so both are accepted -- and nothing
#: else is, because any other count means the producer is not running the mesh this
#: display was calibrated against.
MESH_POINT_COUNTS: frozenset[int] = frozenset({468, 478})

#: magic 4s | version H | point_count H | sequence Q, then point_count * (x, y) float32.
#: Variable length on purpose: the packet says how many points it carries and the length
#: is checked against that, so a truncated payload cannot be read as a shorter mesh.
MESH_HEADER_FORMAT: str = "!4sHHQ"
MESH_HEADER_SIZE: int = struct.calcsize(MESH_HEADER_FORMAT)


@dataclass(frozen=True, slots=True)
class MeshPacket:
    """One frame of dense face landmarks in source-image pixel coordinates.

    These are the *image* coordinates the recogniser produced, at the recognition
    resolution -- not the preview resolution and not display metres. A viewer scales them
    into whatever box it draws in; nothing downstream may treat them as metric.
    """

    points_px: tuple[tuple[float, float], ...]
    sequence: int

    def __post_init__(self) -> None:
        if len(self.points_px) not in MESH_POINT_COUNTS:
            raise ProtocolError(
                f"mesh must carry one of {sorted(MESH_POINT_COUNTS)} points, "
                f"got {len(self.points_px)}"
            )
        for x, y in self.points_px:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ProtocolError("mesh coordinates must be finite (no NaN/Inf)")


def encode_mesh_packet(packet: MeshPacket) -> bytes:
    count = len(packet.points_px)
    header = struct.pack(MESH_HEADER_FORMAT, MESH_MAGIC, MESH_VERSION, count, packet.sequence)
    flat = [value for point in packet.points_px for value in point]
    return header + struct.pack(f"!{2 * count}f", *flat)


def decode_mesh_packet(data: bytes) -> MeshPacket:
    """Deserialize and validate a mesh packet.

    Rejects a bad magic, a version other than ``MESH_VERSION`` (future versions are not
    treated as forward compatible), a point count outside ``MESH_POINT_COUNTS``, a length
    that disagrees with the declared count, and any non-finite coordinate.
    """

    if len(data) < MESH_HEADER_SIZE:
        raise ProtocolError(f"mesh packet is shorter than its {MESH_HEADER_SIZE}-byte header")
    magic, version, count, sequence = struct.unpack(MESH_HEADER_FORMAT, data[:MESH_HEADER_SIZE])
    if magic != MESH_MAGIC:
        raise ProtocolError(f"bad mesh magic {magic!r}, expected {MESH_MAGIC!r}")
    if version != MESH_VERSION:
        raise ProtocolError(
            f"unsupported mesh protocol version {version}; only {MESH_VERSION} is accepted"
        )
    if count not in MESH_POINT_COUNTS:
        raise ProtocolError(
            f"mesh declares {count} points; only {sorted(MESH_POINT_COUNTS)} are accepted"
        )
    expected = MESH_HEADER_SIZE + 8 * count
    if len(data) != expected:
        raise ProtocolError(
            f"mesh declares {count} points ({expected} bytes) but the payload is {len(data)}"
        )
    values = struct.unpack(f"!{2 * count}f", data[MESH_HEADER_SIZE:])
    points = tuple((values[i], values[i + 1]) for i in range(0, len(values), 2))
    return MeshPacket(points_px=points, sequence=sequence)
