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

from . import protocol
from .face_model import HEAD_TO_OPENCV, FaceModel, canonical_face_model, load_personal_face_model
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
class _PoseMeasurement:
    left: np.ndarray
    right: np.ndarray
    cyclopean: np.ndarray
    forward: np.ndarray
    confidence: float
    face_count: int
    landmark_count: int
    movement_m: float


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
        self.face_model: FaceModel = (
            load_personal_face_model(Path(user.face_model_path))
            if user.face_model_path is not None
            else canonical_face_model()
        )
        if self.face_model.is_personal:
            self.left_eye_head_m = self.face_model.left_iris_head_m
            self.right_eye_head_m = self.face_model.right_iris_head_m
        else:
            self.left_eye_head_m = np.asarray(user.left_eye_center_head_m, dtype=np.float64)
            self.right_eye_head_m = np.asarray(user.right_eye_center_head_m, dtype=np.float64)
        self.cyclopean_eye_head_m = 0.5 * (self.left_eye_head_m + self.right_eye_head_m)
        self._pnp_object_points = np.ascontiguousarray(
            self.face_model.pnp_points_opencv_m[self.LANDMARK_INDICES]
        )
        self._left_eye_opencv_m = self.left_eye_head_m * HEAD_TO_OPENCV
        self._right_eye_opencv_m = self.right_eye_head_m * HEAD_TO_OPENCV
        self._cyclopean_eye_opencv_m = self.cyclopean_eye_head_m * HEAD_TO_OPENCV
        self._forward_axis_opencv = np.asarray(user.neutral_forward_axis_head) * HEAD_TO_OPENCV

    def estimate(
        self,
        landmarks_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        rotation_camera_head, _ = cv2.Rodrigues(rotation_vector)
        translation_camera_head = translation_vector.reshape(3)

        left_camera = rotation_camera_head @ self._left_eye_opencv_m + translation_camera_head
        right_camera = rotation_camera_head @ self._right_eye_opencv_m + translation_camera_head
        cyclopean_camera = (
            rotation_camera_head @ self._cyclopean_eye_opencv_m + translation_camera_head
        )
        forward_camera = rotation_camera_head @ self._forward_axis_opencv

        left_display = self._display_rotation @ left_camera + self._display_translation
        right_display = self._display_rotation @ right_camera + self._display_translation
        cyclopean_display = self._display_rotation @ cyclopean_camera + self._display_translation
        forward_display = normalize(self._display_rotation @ forward_camera)
        return left_display, right_display, cyclopean_display, forward_display


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
        if input_frame.faces:
            face = max(input_frame.faces, key=lambda value: value.score)
            measurement = self._measurement_from_landmarks(
                face.landmarks_xy, face.score, len(input_frame.faces), input_frame.frame_bgr
            )
        else:
            measurement = self._fallback_measurement(0)
        diagnostics: dict[str, object] | None = None
        if input_frame.frame_index is not None:
            diagnostics = {"input_frame": input_frame.frame_index}
        if input_frame.source_timestamps is not None:
            # Replay/synthetic/live-camera frames leave source_timestamps at its default
            # None and so never add these keys (workdoc step 36 follow-up, item 4).
            diagnostics = {**(diagnostics or {}), **input_frame.source_timestamps}
        state = self._build_state(measurement, started, diagnostics=diagnostics)
        if input_frame.preview_jpeg is not None:
            self._forwarded_previews += 1
            return state, input_frame.preview_jpeg
        if input_frame.frame_bgr is None:
            # Neither a producer-compressed preview nor raw pixels are available yet --
            # the IPC control lane before its first preview publish. There is nothing to
            # send on /ws/camera this tick; that is fine, subscribers simply keep
            # waiting. The control-lane pose above still advances regardless (workdoc
            # steps 36-38): a preview lane that has never been used must never block it.
            return state, None
        self._encoded_previews += 1
        return state, self._encode_preview(input_frame.frame_bgr, measurement, input_frame.label)

    def close(self) -> None:
        self._frame_source.close()

    def _measurement_from_landmarks(
        self, landmarks: np.ndarray, confidence: float, face_count: int, frame: np.ndarray | None
    ) -> _PoseMeasurement:
        left, right, eye, forward = self._estimator.estimate(landmarks)
        movement = float(np.linalg.norm(eye - self._last_eye))
        self._last_eye = eye
        self._draw_landmarks(frame, landmarks)
        return _PoseMeasurement(
            left=left,
            right=right,
            cyclopean=eye,
            forward=forward,
            confidence=confidence,
            face_count=face_count,
            landmark_count=int(landmarks.shape[0]),
            movement_m=movement,
        )

    def _fallback_measurement(self, face_count: int) -> _PoseMeasurement:
        eye = self._last_eye
        return _PoseMeasurement(
            left=eye + np.array([-0.032, 0.0, 0.0]),
            right=eye + np.array([0.032, 0.0, 0.0]),
            cyclopean=eye,
            forward=np.array([0.0, 0.0, -1.0]),
            confidence=0.0,
            face_count=face_count,
            landmark_count=0,
            movement_m=0.0,
        )

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
        measurement: _PoseMeasurement,
        started: float,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> TrackingState:
        now = time.perf_counter()
        self._fps_ema = self._smoothed_fps(now)
        self._sequence += 1
        state_diagnostics: dict[str, object] = {"face_count": measurement.face_count}
        if measurement.landmark_count:
            state_diagnostics.update(
                landmark_count=measurement.landmark_count,
                eye_movement_m=measurement.movement_m,
            )
        if diagnostics is not None:
            state_diagnostics.update(diagnostics)
        return TrackingState(
            sequence=self._sequence,
            timestamp_unix_s=time.time(),
            source=self._source,
            confidence=measurement.confidence,
            cyclopean_eye_display_m=tuple(float(value) for value in measurement.cyclopean),
            left_eye_display_m=tuple(float(value) for value in measurement.left),
            right_eye_display_m=tuple(float(value) for value in measurement.right),
            head_forward_display=tuple(float(value) for value in measurement.forward),
            tracking_fps=self._fps_ema,
            inference_ms=(now - started) * 1000.0,
            stable=measurement.movement_m < 0.004 and measurement.confidence >= 0.75,
            diagnostics=state_diagnostics,
        )

    def _smoothed_fps(self, now: float) -> float:
        interval = max(now - self._last_timestamp, 1e-6)
        self._last_timestamp = now
        instant_fps = 1.0 / interval
        self._fps_ema = (
            instant_fps if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * instant_fps
        )
        return self._fps_ema

    def _encode_preview(
        self, frame: np.ndarray | None, measurement: _PoseMeasurement, label: str
    ) -> bytes:
        if frame is None:
            # No silent placeholder: an input that carries neither pixels nor a
            # producer-compressed preview is a configuration error, not a blank frame.
            raise RuntimeError(
                "this frame source supplied neither frame_bgr nor preview_jpeg; "
                "there is nothing to send on /ws/camera"
            )
        cv2.putText(
            frame,
            f"{label} {self._fps_ema:.1f} fps / {measurement.confidence:.2f}",
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
