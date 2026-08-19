"""Tracking provider contracts and the optional FaceMesh live-camera adapter."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Condition
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import protocol
from .face_model import HEAD_TO_OPENCV, FaceModel, canonical_face_model, load_personal_face_model
from .filtering import (
    DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    DEFAULT_MAX_LINEAR_SPEED_M_S,
    DEFAULT_PREDICTION_HORIZON_S,
    PoseFilter,
    Se3EKF,
    Se3PoseFilter,
)
from .geometry import normalize
from .models import HardwareProfile, TrackingSource, TrackingState, UserProfile
from .profiles import resolved_camera_to_display


@runtime_checkable
class TrackingProvider(Protocol):
    #: ``None`` only when neither a raw frame nor a producer-compressed preview is
    #: available yet (the IPC control lane before its first preview arrives); see
    #: ``FaceMeshPoseProvider.sample`` and ``FaceMeshInputFrame``.
    def sample(self) -> tuple[TrackingState, bytes | None]: ...

    def close(self) -> None: ...


class FaceMeshUnavailableError(RuntimeError):
    pass


class _LiveFaceObservation(Protocol):
    score: float
    xy: np.ndarray
    points: np.ndarray


class _FaceMeshResult(Protocol):
    faces: Sequence[_LiveFaceObservation]


@dataclass(frozen=True)
class HeadPoseEstimate:
    """Metric head pose in display coordinates plus PnP quality (plan §3.3).

    ``T_S_H`` is the full SE(3) head pose; the eye/forward vectors are derived from
    it via :meth:`HeadPoseEstimator.eyes_from_pose` and kept here for compatibility.
    """

    T_S_H: np.ndarray
    left_eye_display_m: np.ndarray
    right_eye_display_m: np.ndarray
    cyclopean_eye_display_m: np.ndarray
    head_forward_display: np.ndarray
    reprojection_rms_px: float
    inlier_count: int
    timestamp_unix_ns: int


#: Default PnP reprojection-RMS gate. Stage 1 value; the plan (§10 D8) calls for
#: measuring the real RMS distribution before treating this as a hard threshold.
DEFAULT_REPROJECTION_RMS_LIMIT_PX = 4.0
#: Consecutive invalid frames before the provider raises so the runtime declares the
#: input stale (plan §10 B3). At 30 fps this is ~0.17 s of held last pose.
DEFAULT_INVALID_LIMIT = 5


def _optional_tuple(values: np.ndarray | None) -> tuple[float, float, float] | None:
    """Convert a 3-vector to a JSON-safe tuple, or ``None`` when absent."""

    if values is None:
        return None
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class _FilterStep:
    """Outcome of one pose-filter update for :meth:`FaceMeshPoseProvider.sample`."""

    tracking_valid: bool
    rejection: str | None
    reprojection_rms_px: float | None
    inlier_count: int | None
    pose: np.ndarray | None
    eyes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    head_position_display_m: np.ndarray | None
    head_orientation_xyzw: tuple[float, float, float, float] | None
    linear_velocity_display_m_s: np.ndarray | None
    angular_velocity_display_rad_s: np.ndarray | None
    pose_timestamp_unix_ns: int | None
    predicted_to_unix_ns: int | None


@dataclass(frozen=True)
class FaceMeshObservation:
    score: float
    landmarks_xy: np.ndarray


@dataclass(frozen=True)
class _RecordedFrame:
    frame_index: int
    faces: tuple[FaceMeshObservation, ...]


@dataclass(frozen=True)
class FaceMeshInputFrame:
    """One BGR frame already paired with its FaceMesh observations.

    The pose/WebSocket code owns this contract, not the camera, JSON, or future IPC
    transport. A recording is therefore a deterministic test double for a live source.
    """

    #: ``None`` when the producer already compressed the preview itself. Holding a
    #: decoded frame on the display machine only to draw on it and re-encode it is the
    #: cost this input port exists to avoid, so the pixels are simply not carried.
    frame_bgr: np.ndarray | None
    faces: tuple[FaceMeshObservation, ...]
    label: str
    frame_index: int | None = None
    #: A preview already compressed by the producer. When set, the server forwards these
    #: bytes untouched: decoding a JPEG only to draw on it and re-encode it costs a
    #: decode plus an encode per frame on the display machine and gains nothing the
    #: producer could not have drawn itself. ``None`` keeps the local encode path, which
    #: recorded replay and synthetic sources still use.
    preview_jpeg: bytes | None = None
    #: Upstream-producer clock readings to surface in ``TrackingState.diagnostics``, or
    #: ``None`` when there is no separate producer process (replay/synthetic/live camera
    #: all run PnP in this process, so ``started``/``time.perf_counter()`` already cover
    #: them). Keys must self-describe both *whose* clock and *which* domain, since the
    #: producer and this server are separate processes: a ``*_monotonic_ns`` value is
    #: only comparable to another ``*_monotonic_ns`` value from the *same* process, never
    #: across processes, and never against a ``*_unix_ns`` value (see protocol.py's own
    #: ``ControlPacket`` docstring). See ``IpcFaceMeshInput.next_frame`` for the actual
    #: keys this port populates.
    source_timestamps: dict[str, int] | None = None


#: JPEG start-of-frame markers. SOF4/SOF8/SOF12 are not frame headers despite being in
#: the 0xC0-0xCF block, so they are excluded rather than mis-parsed.
_JPEG_SOF_EXCLUDED = frozenset({0xC4, 0xC8, 0xCC})

#: Preview lane resolution contract (workdoc steps 37-38). The server never resizes or
#: re-encodes a preview -- it only checks these dimensions via ``jpeg_dimensions`` and
#: rejects anything else with 422. Recognition/PnP always stay at the camera's full
#: resolution; this is a separate, display-only contract, never derived from it.
#: Mirrored (not imported) in ``scripts/facemesh_ipc_producer.py``'s own
#: ``PREVIEW_WIDTH_PX``/``PREVIEW_HEIGHT_PX``, since that script runs in a separate
#: Python 3.10 process and only imports the dependency-free ``protocol`` submodule.
PREVIEW_WIDTH_PX = 640
PREVIEW_HEIGHT_PX = 360


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read a JPEG's width and height from its SOF marker, without decoding it.

    The server polices the preview resolution contract on every frame. Decoding the image
    to read its shape would reintroduce exactly the per-frame decode this lane exists to
    remove, so the header is parsed directly instead.
    """

    if data[:2] != b"\xff\xd8":
        raise ValueError("preview payload is not a JPEG")
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            raise ValueError("malformed JPEG marker segment")
        marker = data[offset + 1]
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in _JPEG_SOF_EXCLUDED:
            if offset + 9 > len(data):
                raise ValueError("truncated JPEG start-of-frame segment")
            height = int.from_bytes(data[offset + 5 : offset + 7], "big")
            width = int.from_bytes(data[offset + 7 : offset + 9], "big")
            return width, height
        if length < 2:
            raise ValueError("malformed JPEG segment length")
        offset += 2 + length
    raise ValueError("JPEG has no start-of-frame marker")


@runtime_checkable
class FaceMeshFrameSource(Protocol):
    """Port for a live, recorded, or IPC-provided sequence of FaceMesh frames."""

    def next_frame(self) -> FaceMeshInputFrame: ...

    def close(self) -> None: ...


def _apply_forward_axis_corrections(
    forward_axis_head: np.ndarray, *, yaw_deg: float, pitch_deg: float
) -> np.ndarray:
    """Apply per-user yaw (about head +Y) then pitch (about head +X) corrections.

    Both default to 0deg, so an uncalibrated profile leaves the forward axis untouched.
    """

    forward = np.asarray(forward_axis_head, dtype=np.float64).reshape(3)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    rotation_yaw = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]], dtype=np.float64
    )
    rotation_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]], dtype=np.float64
    )
    corrected = rotation_pitch @ rotation_yaw @ forward
    return normalize(corrected)


class HeadPoseEstimator:
    """Solve a metric dense-landmark PnP pose and derive pupil positions.

    ``SOLVEPNP_SQPNP`` has no fragile initial estimate; cheirality is then enforced so a
    numerically successful mirrored solution cannot put the face behind the camera.
    """

    LANDMARK_INDICES = np.array([1, 6, 33, 133, 362, 263, 61, 291, 199, 168, 94, 4], dtype=np.int32)

    def __init__(self, hardware: HardwareProfile, user: UserProfile) -> None:
        self.camera_matrix = np.asarray(hardware.camera.camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(
            hardware.camera.distortion_coefficients, dtype=np.float64
        ).reshape(-1, 1)
        camera_to_display = resolved_camera_to_display(hardware)
        self._display_rotation = np.ascontiguousarray(camera_to_display[:3, :3])
        self._display_translation = np.ascontiguousarray(camera_to_display[:3, 3])
        self._head_to_opencv = np.diag(HEAD_TO_OPENCV)
        self.face_model: FaceModel = (
            load_personal_face_model(Path(user.face_model_path))
            if user.face_model_path is not None
            else canonical_face_model()
        )
        if self.face_model.is_personal:
            self.left_eye_head_m = self.face_model.left_iris_head_m
            self.right_eye_head_m = self.face_model.right_iris_head_m
            # A personal ``shape.pcd`` is the metric ground truth, so ``ipd_m`` is not
            # applied here (plan §2.2 d): scaling a measured mesh would distort it.
        else:
            user_left = np.asarray(user.left_eye_center_head_m, dtype=np.float64)
            user_right = np.asarray(user.right_eye_center_head_m, dtype=np.float64)
            user_eye_ipd = float(np.linalg.norm(user_left - user_right))
            # Scale the canonical prior so its inter-ocular distance matches the user's
            # measured ``ipd_m``, and apply the SAME factor to the PnP model so the PnP
            # distance estimate stays consistent with the eye offsets (plan §2.2 d /
            # geometry_6dof_review §4.2: scaling eyes alone breaks rigidity).
            ipd_scale = user.ipd_m / user_eye_ipd if user_eye_ipd > 1e-9 else 1.0
            self.left_eye_head_m = user_left * ipd_scale
            self.right_eye_head_m = user_right * ipd_scale
            if abs(ipd_scale - 1.0) > 1e-12:
                self.face_model = FaceModel(
                    points_head_m=self.face_model.points_head_m * ipd_scale,
                    source=f"{self.face_model.source}-scaled-by-ipd",
                    is_personal=False,
                )
        self.cyclopean_eye_head_m = 0.5 * (self.left_eye_head_m + self.right_eye_head_m)
        self._pnp_object_points = np.ascontiguousarray(
            self.face_model.pnp_points_opencv_m[self.LANDMARK_INDICES]
        )
        self._left_eye_opencv_m = self.left_eye_head_m * HEAD_TO_OPENCV
        self._right_eye_opencv_m = self.right_eye_head_m * HEAD_TO_OPENCV
        self._cyclopean_eye_opencv_m = self.cyclopean_eye_head_m * HEAD_TO_OPENCV
        # Forward-axis yaw/pitch corrections were previously defined but unused; apply
        # them now so a per-user head-forward calibration actually reaches the renderer.
        self.neutral_forward_axis_head = _apply_forward_axis_corrections(
            np.asarray(user.neutral_forward_axis_head, dtype=np.float64),
            yaw_deg=user.forward_axis_yaw_correction_deg,
            pitch_deg=user.forward_axis_pitch_correction_deg,
        )
        self._forward_axis_opencv = self.neutral_forward_axis_head * HEAD_TO_OPENCV

    def eyes_from_pose(
        self, t_s_h: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Derive (left, right, cyclopean, forward) in display coords from ``T_S_H``.

        Because the transform is rigid, the inter-ocular distance is preserved
        exactly -- smoothing the pose and then deriving eyes here is what keeps IPD
        stable, instead of low-passing each eye independently (plan §4).
        """

        transform = np.asarray(t_s_h, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("T_S_H must be a 4x4 transform")
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        left = rotation @ self.left_eye_head_m + translation
        right = rotation @ self.right_eye_head_m + translation
        cyclopean = rotation @ self.cyclopean_eye_head_m + translation
        forward = normalize(rotation @ self.neutral_forward_axis_head)
        return left, right, cyclopean, forward

    def estimate_pose(
        self,
        landmarks_xy: np.ndarray,
        *,
        timestamp_unix_ns: int | None = None,
        reprojection_inlier_px: float = 3.0,
    ) -> HeadPoseEstimate:
        """Solve the metric PnP pose, build ``T_S_H``, and score reprojection quality."""

        landmarks = np.asarray(landmarks_xy, dtype=np.float64)
        if (
            landmarks.ndim != 2
            or landmarks.shape[1] < 2
            or len(landmarks) <= int(self.LANDMARK_INDICES.max())
        ):
            raise ValueError("FaceMesh landmarks must contain the 468 base landmarks")
        image_points = np.ascontiguousarray(landmarks[self.LANDMARK_INDICES, :2])
        if not np.isfinite(image_points).all():
            raise ValueError("FaceMesh landmarks contain non-finite image coordinates")
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self._pnp_object_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success or translation_vector.reshape(3)[2] <= 0.0:
            raise RuntimeError("head pose PnP did not converge")
        rotation_camera_object, _ = cv2.Rodrigues(rotation_vector)
        translation_camera_head = translation_vector.reshape(3)

        # Head-frame H -> camera C. The object points are head coords with the
        # HEAD_TO_OPENCV axis flip, so R_C_H = R_C_O @ diag(HEAD_TO_OPENCV).
        rotation_camera_head = rotation_camera_object @ self._head_to_opencv
        rotation_display_head = self._display_rotation @ rotation_camera_head
        translation_display_head = (
            self._display_rotation @ translation_camera_head + self._display_translation
        )
        t_s_h = np.eye(4, dtype=np.float64)
        t_s_h[:3, :3] = rotation_display_head
        t_s_h[:3, 3] = translation_display_head

        projected, _ = cv2.projectPoints(
            self._pnp_object_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.distortion,
        )
        projected = projected.reshape(-1, 2)
        per_point_error = np.linalg.norm(projected - image_points, axis=1)
        rms_px = float(np.sqrt(np.mean(per_point_error**2)))
        inlier_count = int(np.count_nonzero(per_point_error <= reprojection_inlier_px))

        left, right, cyclopean, forward = self.eyes_from_pose(t_s_h)
        return HeadPoseEstimate(
            T_S_H=t_s_h,
            left_eye_display_m=left,
            right_eye_display_m=right,
            cyclopean_eye_display_m=cyclopean,
            head_forward_display=forward,
            reprojection_rms_px=rms_px,
            inlier_count=inlier_count,
            timestamp_unix_ns=int(
                time.time_ns() if timestamp_unix_ns is None else timestamp_unix_ns
            ),
        )

    def estimate(
        self,
        landmarks_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Legacy 4-tuple eye/forward accessor; delegates to :meth:`estimate_pose`."""

        estimate = self.estimate_pose(landmarks_xy)
        return (
            estimate.left_eye_display_m,
            estimate.right_eye_display_m,
            estimate.cyclopean_eye_display_m,
            estimate.head_forward_display,
        )


def _parse_replay_face(value: object, *, frame_index: int, face_index: int) -> FaceMeshObservation:
    if not isinstance(value, dict):
        raise ValueError(f"replay frame {frame_index}, face {face_index} must be an object")
    try:
        score = float(value["score"])
        landmarks = np.asarray(value["landmarks"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"replay frame {frame_index}, face {face_index} is malformed") from exc
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"replay frame {frame_index}, face {face_index} has an invalid score")
    if landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] < 2:
        raise ValueError(f"replay frame {frame_index}, face {face_index} lacks 468 image landmarks")
    if not np.isfinite(landmarks[:, :2]).all():
        raise ValueError(f"replay frame {frame_index}, face {face_index} has non-finite landmarks")
    return FaceMeshObservation(score=score, landmarks_xy=np.ascontiguousarray(landmarks[:, :2]))


def _load_replay_frames(path: Path) -> tuple[_RecordedFrame, ...]:
    """Load the stable JSON schema emitted by ``facemesh run --save-json``."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load FaceMesh replay JSON {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("FaceMesh replay JSON must be a non-empty frame list")
    frames: list[_RecordedFrame] = []
    for expected_index, value in enumerate(data):
        if not isinstance(value, dict) or value.get("frame") != expected_index:
            raise ValueError("FaceMesh replay frames must have contiguous zero-based frame indices")
        faces = value.get("faces")
        if not isinstance(faces, list):
            raise ValueError(f"replay frame {expected_index} must contain a faces list")
        frames.append(
            _RecordedFrame(
                frame_index=expected_index,
                faces=tuple(
                    _parse_replay_face(face, frame_index=expected_index, face_index=face_index)
                    for face_index, face in enumerate(faces)
                ),
            )
        )
    return tuple(frames)


def _open_replay_capture(
    video_path: Path, hardware: HardwareProfile, expected_frames: int
) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open replay video {video_path}")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (hardware.camera.image_width_px, hardware.camera.image_height_px):
        capture.release()
        raise ValueError(
            "replay video dimensions must match the intrinsics "
            f"({width}x{height} != {hardware.camera.image_width_px}x{hardware.camera.image_height_px})"
        )
    decoded_frames = _count_decodable_frames(capture)
    if decoded_frames != expected_frames:
        capture.release()
        raise ValueError(
            "replay video has "
            f"{decoded_frames} decodable frames but landmarks JSON has {expected_frames}"
        )
    return capture


def _count_decodable_frames(capture: cv2.VideoCapture) -> int:
    """Count decoded frames instead of trusting unreliable container frame metadata."""

    count = 0
    while capture.grab():
        count += 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


class RecordedFaceMeshInput:
    """Video-backed implementation of :class:`FaceMeshFrameSource`.

    It is intentionally a transport adapter: the exact JSON emitted by
    ``facemesh run --save-json`` is replayed as if an upstream live service had emitted
    those observations. It lets browser E2E tests exercise the production metric-PnP and
    WebSocket path without importing the Python/CUDA inference environment.
    """

    def __init__(
        self, hardware: HardwareProfile, *, landmarks_path: Path, video_path: Path
    ) -> None:
        self._records = _load_replay_frames(landmarks_path)
        self._capture = _open_replay_capture(video_path, hardware, len(self._records))
        self._record_index = 0

    def next_frame(self) -> FaceMeshInputFrame:
        record = self._records[self._record_index]
        frame = self._read_video_frame()
        self._record_index = (self._record_index + 1) % len(self._records)
        return FaceMeshInputFrame(
            frame_bgr=frame,
            faces=record.faces,
            label="FaceMesh replay",
            frame_index=record.frame_index,
        )

    def _read_video_frame(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if ok:
            return frame
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("replay video cannot be read")
        return frame

    def close(self) -> None:
        self._capture.release()


def _sparse_landmarks_from_control(packet: protocol.ControlPacket) -> np.ndarray:
    """Expand a control packet's 12 image points into a ``(478, 2)`` landmark array.

    ``HeadPoseEstimator.estimate`` expects an array shaped like a full FaceMesh result
    and gathers ``LANDMARK_INDICES`` (identical to ``protocol.LANDMARK_INDICES``, see
    protocol.py's own cross-check test) out of it. The control lane carries only those
    12 points, so every other row is filled with NaN rather than some plausible-looking
    placeholder: if anything ever reads a row this transport does not carry, it must
    fail loudly (NaN fails ``estimate``'s finiteness check) instead of silently acting
    on stale or zeroed-out data.

    Known limitation -- personal-mesh iris eye positions: ``HeadPoseEstimator`` never
    reads a *live* iris landmark (index 468/473) per frame either way -- when a personal
    face model is loaded, ``self.left_eye_head_m``/``right_eye_head_m`` are read once
    from the model's own fixed mesh (``FaceModel.left_iris_head_m``/``right_iris_head_m``,
    reconstructed offline) at ``HeadPoseEstimator.__init__`` time, not from any per-frame
    observation. So this control packet's omission of live 468/473 does not break that
    path today. It does mean no *future* code path can read a live per-frame iris
    position over this transport (it would only ever see NaN here); such code must
    explicitly fall back to ``UserProfile.left_eye_center_head_m``/
    ``right_eye_center_head_m`` -- the same fallback already used whenever no personal
    face model is loaded at all -- rather than silently treating NaN as a real position.
    """

    landmarks = np.full((478, 2), np.nan, dtype=np.float64)
    landmarks[list(protocol.LANDMARK_INDICES), :] = np.asarray(
        packet.landmarks_px, dtype=np.float64
    )
    return landmarks


def _control_source_timestamps(packet: protocol.ControlPacket) -> dict[str, int]:
    """Producer-side clock readings for ``FaceMeshInputFrame.source_timestamps``.

    Keys are named ``producer_*`` (whose clock) and ``*_monotonic_ns``/``*_unix_ns``
    (which domain) so a consumer never has to guess before comparing: the two
    ``producer_*_monotonic_ns`` values are only comparable to each other (both come from
    the producer's own ``time.perf_counter_ns()``), never to this server's own clock. The
    two ``producer_*_unix_ns`` values share a Unix epoch with this server's
    ``time.time_ns()`` and with a browser's
    ``performance.timeOrigin + performance.now()`` converted to ns (modulo ordinary
    clock-sync error), so those are the pair success condition 10 (recognition-to-WebGL
    latency) is measured against.
    """

    return {
        "producer_capture_monotonic_ns": packet.capture_monotonic_ns,
        "producer_capture_unix_ns": packet.capture_unix_ns,
        "producer_inference_monotonic_ns": packet.inference_monotonic_ns,
        "producer_inference_unix_ns": packet.inference_unix_ns,
    }


class IpcFaceMeshInput:
    """Two-lane latest-value localhost input port for a separate FaceMesh process.

    Control and preview are independent lanes (workdoc steps 36-38); each retains only
    its newest published value, so neither lane ever accumulates a backlog.
    ``next_frame`` blocks on new *control* data only: a control packet advances the pose
    whether or not a preview has ever been published, and publishing (or not
    republishing) a preview never blocks or skips a control update. A rejected preview
    never affects control, and a rejected control packet never touches the stored
    preview -- ``publish_control``/``publish_preview`` validate independently.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        # `hardware` is accepted only for constructor-signature stability with existing
        # callers (api.py). Unlike the old single-lane JSON+JPEG payload, neither lane's
        # geometry depends on it any more: the control lane is a fixed 12-point wire
        # format (protocol.py) independent of camera resolution, and the preview lane's
        # resolution is the fixed PREVIEW_WIDTH_PX/PREVIEW_HEIGHT_PX contract instead of
        # the camera's own intrinsics.
        del hardware
        self._condition = Condition()
        self._latest_control: protocol.ControlPacket | None = None
        self._latest_preview: bytes | None = None
        self._published_version = 0
        self._consumed_version = 0
        self._closed = False

    def publish_control(self, packet: bytes) -> int:
        """Decode and accept one control packet, returning its wire sequence number.

        Raises ``protocol.ProtocolError`` (a ``ValueError``) for anything malformed:
        bad magic/version, wrong length, or non-finite coordinates/score.
        """

        decoded = protocol.decode_control_packet(packet)
        with self._condition:
            if self._closed:
                raise RuntimeError("FaceMesh IPC input is closed")
            self._latest_control = decoded
            self._published_version += 1
            self._condition.notify_all()
        return decoded.sequence

    def publish_preview(self, jpeg: bytes) -> None:
        """Accept one already-compressed preview frame, checked only for dimensions.

        Never decoded back into pixels anywhere in this module (workdoc step 36-38 DoD)
        -- the bytes are stored as-is and later forwarded byte-for-byte by
        ``FaceMeshPoseProvider.sample``.
        """

        width, height = jpeg_dimensions(jpeg)
        if (width, height) != (PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX):
            raise ValueError(
                "FaceMesh IPC preview must be "
                f"{PREVIEW_WIDTH_PX}x{PREVIEW_HEIGHT_PX}, got {width}x{height}"
            )
        with self._condition:
            if self._closed:
                raise RuntimeError("FaceMesh IPC input is closed")
            self._latest_preview = jpeg

    def next_frame(self) -> FaceMeshInputFrame:
        """Block for the next *control* update only; preview never gates this call.

        Returns ``frame_bgr=None`` always (the IPC control lane never carries raw
        pixels) and ``preview_jpeg`` set to whatever preview was most recently
        published -- possibly ``None`` if none has arrived yet, possibly the same bytes
        as last time if the preview lane simply has not been updated since.
        """

        deadline = time.monotonic() + 3.0
        with self._condition:
            while not self._closed and (
                self._latest_control is None or self._published_version == self._consumed_version
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("waiting for a FaceMesh IPC control packet timed out")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("FaceMesh IPC input is closed")
            assert self._latest_control is not None
            self._consumed_version = self._published_version
            control = self._latest_control
            preview = self._latest_preview
        return FaceMeshInputFrame(
            frame_bgr=None,
            faces=(
                FaceMeshObservation(
                    score=control.score, landmarks_xy=_sparse_landmarks_from_control(control)
                ),
            ),
            label="FaceMesh IPC",
            frame_index=control.sequence,
            preview_jpeg=preview,
            source_timestamps=_control_source_timestamps(control),
        )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _LiveFaceMeshInput:
    """Live implementation of the same frame-source port, kept behind optional imports."""

    def __init__(
        self, *, camera_device: str, backend: str, width: int, height: int
    ) -> None:  # pragma: no cover - hardware dependent
        source_path = os.getenv("FACEMESH_TRACKING_SOURCE")
        if source_path:
            sys.path.insert(0, str(Path(source_path).expanduser().resolve()))
        try:
            from facemesh_tracking.pipeline import FaceMeshPipeline
            from facemesh_tracking.runtime import Backend
        except Exception as exc:
            raise FaceMeshUnavailableError(
                "facemesh_tracking is unavailable. Use replay mode or install the "
                "tracking project and set FACEMESH_TRACKING_SOURCE to its src/."
            ) from exc
        self._pipeline = FaceMeshPipeline.create(backend=Backend(backend))
        source = int(camera_device) if camera_device.isdecimal() else camera_device
        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError(f"unable to open camera device {camera_device!r}")

    def next_frame(self) -> FaceMeshInputFrame:  # pragma: no cover - hardware dependent
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("camera frame capture failed")
        result = self._pipeline.process(frame)
        faces = tuple(
            FaceMeshObservation(
                score=float(face.score), landmarks_xy=np.ascontiguousarray(face.xy[:, :2])
            )
            for face in result.faces
        )
        return FaceMeshInputFrame(frame_bgr=frame, faces=faces, label="FaceMesh")

    def close(self) -> None:  # pragma: no cover - hardware dependent
        self._capture.release()


class FaceMeshPoseProvider:
    """Metric PnP/WebSocket provider over any :class:`FaceMeshFrameSource` input port."""

    def __init__(
        self,
        hardware: HardwareProfile,
        user: UserProfile,
        *,
        source: TrackingSource,
        frame_source: FaceMeshFrameSource,
    ) -> None:
        self._source = source
        self._frame_source = frame_source
        self._estimator = HeadPoseEstimator(hardware, user)
        self._sequence = 0
        self._last_timestamp = time.perf_counter()
        self._fps_ema = 0.0
        self._last_eye = np.array([0.0, 0.0, 0.65], dtype=np.float64)
        # Counted so a test can assert the server re-encoded nothing, rather than
        # inferring it from timings.
        self._forwarded_previews = 0
        self._encoded_previews = 0
        # SE(3) pose filter (plan §3.5/§10 C4: manifold EMA + velocity prediction,
        # or the stage-2 on-manifold EKF when quality_metrics["pose_filter"]="ekf").
        # Default is the EKF: it tracks a constant velocity exactly and has lower
        # lag than the EMA at similar smoothing (measured in the stage-2 workdoc).
        pose_filter = str(hardware.quality_metrics.get("pose_filter", "ekf"))
        if pose_filter not in ("ema", "ekf"):
            raise ValueError(f"pose_filter must be 'ema' or 'ekf', got {pose_filter!r}")
        self._prediction_horizon_s = float(
            hardware.quality_metrics.get("prediction_horizon_s", DEFAULT_PREDICTION_HORIZON_S)
        )
        common = {
            "prediction_horizon_s": self._prediction_horizon_s,
            "max_linear_speed_m_s": float(
                hardware.quality_metrics.get("max_linear_speed_m_s", DEFAULT_MAX_LINEAR_SPEED_M_S)
            ),
            "max_angular_speed_rad_s": float(
                hardware.quality_metrics.get(
                    "max_angular_speed_rad_s", DEFAULT_MAX_ANGULAR_SPEED_RAD_S
                )
            ),
        }
        if pose_filter == "ekf":
            self._filter: PoseFilter = Se3EKF(**common)
        else:
            self._filter = Se3PoseFilter(**common)
        self._last_estimate: HeadPoseEstimate | None = None
        self._consecutive_invalid = 0
        self._invalid_limit = DEFAULT_INVALID_LIMIT
        self._reprojection_rms_limit = float(
            hardware.quality_metrics.get(
                "reprojection_rms_limit_px", DEFAULT_REPROJECTION_RMS_LIMIT_PX
            )
        )

    @property
    def preview_encode_count(self) -> int:
        """How many previews this server encoded itself. Must stay 0 for IPC input."""

        return self._encoded_previews

    @property
    def preview_forward_count(self) -> int:
        return self._forwarded_previews

    def sample(self) -> tuple[TrackingState, bytes | None]:
        started = time.perf_counter()
        input_frame = self._frame_source.next_frame()
        pose_ts_ns = self._pose_timestamp_ns(input_frame)
        face_score, landmark_count = self._estimate_for_frame(input_frame, pose_ts_ns)
        step = self._step_filter(face_score, pose_ts_ns)
        self._advance_invalid_counter(step)
        movement_m = 0.0
        if step.eyes is not None:
            movement_m = float(np.linalg.norm(step.eyes[2] - self._last_eye))
            self._last_eye = step.eyes[2]
        diagnostics: dict[str, object] | None = None
        if input_frame.frame_index is not None:
            diagnostics = {"input_frame": input_frame.frame_index}
        if input_frame.source_timestamps is not None:
            diagnostics = {**(diagnostics or {}), **input_frame.source_timestamps}
        state = self._build_state(
            step,
            face_score,
            len(input_frame.faces),
            landmark_count,
            movement_m,
            started,
            diagnostics=diagnostics,
        )
        return state, self._forward_preview(input_frame, state, face_score)

    def close(self) -> None:
        self._frame_source.close()

    def _estimate_for_frame(
        self, input_frame: FaceMeshInputFrame, pose_ts_ns: int
    ) -> tuple[float, int]:
        """Run PnP on the best face, returning (score, landmark_count)."""

        if not input_frame.faces:
            self._last_estimate = None
            return 0.0, 0
        face = max(input_frame.faces, key=lambda value: value.score)
        self._last_estimate = self._estimator.estimate_pose(
            face.landmarks_xy, timestamp_unix_ns=pose_ts_ns
        )
        self._draw_landmarks(input_frame.frame_bgr, face.landmarks_xy)
        return float(face.score), int(face.landmarks_xy.shape[0])

    def _advance_invalid_counter(self, step: _FilterStep) -> None:
        if step.tracking_valid:
            self._consecutive_invalid = 0
        elif step.rejection != "no_face":
            # A detected-but-rejected pose (bad reprojection or velocity) counts toward
            # stale. An empty faces tuple is a missed detection, not a dead producer:
            # the IPC control lane would have raised TimeoutError in next_frame if the
            # producer truly stopped, so do not accumulate that here.
            self._consecutive_invalid += 1
        if self._consecutive_invalid >= self._invalid_limit:
            # Holding the last good pose forever would render a dead producer as a
            # perfectly still observer (ONBOARDING §2). Raise so the runtime's stale
            # path fires after stale_after_s (plan §10 B3).
            raise RuntimeError(
                f"tracking invalid for {self._consecutive_invalid} consecutive frames"
            )

    def _forward_preview(
        self,
        input_frame: FaceMeshInputFrame,
        state: TrackingState,
        face_score: float,
    ) -> bytes | None:
        if input_frame.preview_jpeg is not None:
            self._forwarded_previews += 1
            return input_frame.preview_jpeg
        if input_frame.frame_bgr is None:
            # Neither a producer-compressed preview nor raw pixels are available yet --
            # the IPC control lane before its first preview publish. There is nothing to
            # send on /ws/camera this tick; that is fine, subscribers simply keep
            # waiting. The control-lane pose above still advances regardless (workdoc
            # steps 36-38): a preview lane that has never been used must never block it.
            return None
        self._encoded_previews += 1
        return self._encode_preview(input_frame.frame_bgr, face_score, input_frame.label)

    @staticmethod
    def _pose_timestamp_ns(input_frame: FaceMeshInputFrame) -> int:
        timestamps = input_frame.source_timestamps
        if timestamps is not None:
            capture = timestamps.get("producer_capture_unix_ns")
            if isinstance(capture, int):
                return capture
        return time.time_ns()

    def _step_filter(
        self,
        face_score: float,
        pose_ts_ns: int,
    ) -> _FilterStep:
        """Run the reprojection + velocity gates and the SE(3) filter (plan §3.4-3.5)."""

        estimate = self._last_estimate
        if estimate is None:
            return self._hold_step("no_face", pose_ts_ns)

        rms = estimate.reprojection_rms_px
        if rms > self._reprojection_rms_limit:
            return self._hold_step("reprojection_rms", pose_ts_ns, estimate=estimate)

        accepted = self._filter.correct(estimate.T_S_H, rms, timestamp=time.perf_counter())
        if not accepted:
            return self._hold_step("velocity_gate", pose_ts_ns, estimate=estimate)

        pose = self._filter.predict_to(self._prediction_horizon_s)
        eyes = self._estimator.eyes_from_pose(pose)
        derived = self._pose_derived(pose, pose_ts_ns)
        return _FilterStep(
            tracking_valid=True,
            rejection=None,
            reprojection_rms_px=rms,
            inlier_count=estimate.inlier_count,
            pose=pose,
            eyes=eyes,
            head_position_display_m=derived[0],
            head_orientation_xyzw=derived[1],
            linear_velocity_display_m_s=derived[2],
            angular_velocity_display_rad_s=derived[3],
            pose_timestamp_unix_ns=pose_ts_ns,
            predicted_to_unix_ns=pose_ts_ns + int(self._prediction_horizon_s * 1e9),
        )

    def _hold_step(
        self,
        reason: str,
        pose_ts_ns: int,
        *,
        estimate: HeadPoseEstimate | None = None,
    ) -> _FilterStep:
        """Build an invalid step that holds the last good pose (plan §10 B3)."""

        if self._filter.initialised:
            pose = self._filter.pose
            eyes = self._estimator.eyes_from_pose(pose)
            derived = self._pose_derived(pose, pose_ts_ns)
            return _FilterStep(
                tracking_valid=False,
                rejection=reason,
                reprojection_rms_px=estimate.reprojection_rms_px if estimate else None,
                inlier_count=estimate.inlier_count if estimate else None,
                pose=pose,
                eyes=eyes,
                head_position_display_m=derived[0],
                head_orientation_xyzw=derived[1],
                linear_velocity_display_m_s=derived[2],
                angular_velocity_display_rad_s=derived[3],
                pose_timestamp_unix_ns=pose_ts_ns,
                predicted_to_unix_ns=None,
            )
        # No good pose yet: emit a deterministic fallback so the browser still renders
        # *something* before the first accepted frame, flagged invalid.
        eye = self._last_eye
        fallback_eyes = (
            eye + np.array([-0.032, 0.0, 0.0]),
            eye + np.array([0.032, 0.0, 0.0]),
            eye,
            np.array([0.0, 0.0, -1.0]),
        )
        return _FilterStep(
            tracking_valid=False,
            rejection=reason,
            reprojection_rms_px=estimate.reprojection_rms_px if estimate else None,
            inlier_count=estimate.inlier_count if estimate else None,
            pose=None,
            eyes=fallback_eyes,
            head_position_display_m=None,
            head_orientation_xyzw=None,
            linear_velocity_display_m_s=None,
            angular_velocity_display_rad_s=None,
            pose_timestamp_unix_ns=pose_ts_ns,
            predicted_to_unix_ns=None,
        )

    def _pose_derived(
        self, pose: np.ndarray, pose_ts_ns: int
    ) -> tuple[
        np.ndarray,
        tuple[float, float, float, float],
        np.ndarray | None,
        np.ndarray | None,
    ]:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        quaternion = Rotation.from_matrix(rotation).as_quat()
        # scipy returns [x, y, z, w]; normalise to guard against drift.
        quaternion = quaternion / (np.linalg.norm(quaternion) + 1e-12)
        orientation = (
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        )
        velocity_body = self._filter.velocity_body
        # Body twist -> spatial (display-frame) velocities via rotation only, since the
        # body-frame linear velocity of the head origin rotates into the display frame.
        linear = rotation @ velocity_body[3:]
        angular = rotation @ velocity_body[:3]
        return translation, orientation, linear, angular

    @staticmethod
    def _draw_landmarks(frame: np.ndarray | None, landmarks_xy: np.ndarray) -> None:
        if frame is None:
            # The producer draws its own overlay before compressing; there are no pixels
            # here to draw on, and materialising some would defeat the point.
            return
        for point in landmarks_xy[::8, :2]:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 1, (90, 220, 170), -1)

    def _build_state(
        self,
        step: _FilterStep,
        face_score: float,
        face_count: int,
        landmark_count: int,
        movement_m: float,
        started: float,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> TrackingState:
        now = time.perf_counter()
        self._fps_ema = self._smoothed_fps(now)
        self._sequence += 1
        assert step.eyes is not None
        left, right, cyclopean, forward = step.eyes
        confidence = face_score if step.tracking_valid else 0.0
        return TrackingState(
            sequence=self._sequence,
            timestamp_unix_s=time.time(),
            source=self._source,
            confidence=confidence,
            cyclopean_eye_display_m=tuple(float(value) for value in cyclopean),
            left_eye_display_m=tuple(float(value) for value in left),
            right_eye_display_m=tuple(float(value) for value in right),
            head_forward_display=tuple(float(value) for value in forward),
            tracking_fps=self._fps_ema,
            inference_ms=(now - started) * 1000.0,
            stable=step.tracking_valid and movement_m < 0.004 and face_score >= 0.75,
            diagnostics=self._collect_diagnostics(
                step, face_count, landmark_count, movement_m, diagnostics
            ),
            pose_timestamp_unix_ns=step.pose_timestamp_unix_ns,
            head_position_display_m=_optional_tuple(step.head_position_display_m),
            head_orientation_display_xyzw=step.head_orientation_xyzw,
            linear_velocity_display_m_s=_optional_tuple(step.linear_velocity_display_m_s),
            angular_velocity_display_rad_s=_optional_tuple(step.angular_velocity_display_rad_s),
            reprojection_rms_px=step.reprojection_rms_px,
            inlier_count=step.inlier_count,
            tracking_valid=step.tracking_valid,
            predicted_to_unix_ns=step.predicted_to_unix_ns,
        )

    @staticmethod
    def _collect_diagnostics(
        step: _FilterStep,
        face_count: int,
        landmark_count: int,
        movement_m: float,
        extra: dict[str, object] | None,
    ) -> dict[str, object]:
        state_diagnostics: dict[str, object] = {"face_count": face_count}
        if landmark_count:
            state_diagnostics.update(
                landmark_count=landmark_count,
                eye_movement_m=movement_m,
            )
        if step.rejection is not None:
            state_diagnostics["rejection"] = step.rejection
        if step.reprojection_rms_px is not None:
            state_diagnostics["reprojection_rms_px"] = step.reprojection_rms_px
        if step.inlier_count is not None:
            state_diagnostics["inlier_count"] = step.inlier_count
        if extra is not None:
            state_diagnostics.update(extra)
        return state_diagnostics

    def _smoothed_fps(self, now: float) -> float:
        interval = max(now - self._last_timestamp, 1e-6)
        self._last_timestamp = now
        instant_fps = 1.0 / interval
        self._fps_ema = (
            instant_fps if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * instant_fps
        )
        return self._fps_ema

    def _encode_preview(self, frame: np.ndarray | None, confidence: float, label: str) -> bytes:
        if frame is None:
            # No silent placeholder: an input that carries neither pixels nor a
            # producer-compressed preview is a configuration error, not a blank frame.
            raise RuntimeError(
                "this frame source supplied neither frame_bgr nor preview_jpeg; "
                "there is nothing to send on /ws/camera"
            )
        cv2.putText(
            frame,
            f"{label} {self._fps_ema:.1f} fps / {confidence:.2f}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (90, 220, 170),
            2,
            cv2.LINE_AA,
        )
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not encoded_ok:
            raise RuntimeError("failed to encode camera preview")
        return encoded.tobytes()


class FaceMeshTrackingProvider(FaceMeshPoseProvider):
    """Live composition of the shared pose provider and a camera-backed input port."""

    def __init__(
        self,
        hardware: HardwareProfile,
        user: UserProfile,
        *,
        camera_device: str = "/dev/video0",
        backend: str = "cpu",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        super().__init__(
            hardware,
            user,
            source="facemesh",
            frame_source=_LiveFaceMeshInput(
                camera_device=camera_device, backend=backend, width=width, height=height
            ),
        )


class FaceMeshReplayProvider(FaceMeshPoseProvider):
    """Recorded-input composition of the same production metric-PnP provider."""

    def __init__(
        self,
        hardware: HardwareProfile,
        user: UserProfile,
        *,
        landmarks_path: Path,
        video_path: Path,
    ) -> None:
        super().__init__(
            hardware,
            user,
            source="replay",
            frame_source=RecordedFaceMeshInput(
                hardware, landmarks_path=landmarks_path, video_path=video_path
            ),
        )


class FaceMeshIpcProvider(FaceMeshPoseProvider):
    """Compose the metric pose path with a producer-driven local IPC input port."""

    def __init__(
        self, hardware: HardwareProfile, user: UserProfile, *, frame_source: IpcFaceMeshInput
    ) -> None:
        super().__init__(hardware, user, source="ipc", frame_source=frame_source)
