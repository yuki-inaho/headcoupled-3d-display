"""Tracking provider contracts and the optional FaceMesh live-camera adapter."""

from __future__ import annotations

import base64
import binascii
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

from .face_model import HEAD_TO_OPENCV, FaceModel, canonical_face_model, load_personal_face_model
from .geometry import normalize
from .models import HardwareProfile, TrackingSource, TrackingState, UserProfile
from .profiles import resolved_camera_to_display


@runtime_checkable
class TrackingProvider(Protocol):
    def sample(self) -> tuple[TrackingState, bytes]: ...

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

    frame_bgr: np.ndarray
    faces: tuple[FaceMeshObservation, ...]
    label: str
    frame_index: int | None = None


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


class IpcFaceMeshInput:
    """Latest-value localhost input port for a separate FaceMesh Python environment.

    Producers submit one JSON object containing an MJPEG frame and the corresponding
    FaceMesh landmarks. Only the newest complete pair is retained, matching the browser
    runtime's no-backlog policy.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self._expected_size = (hardware.camera.image_width_px, hardware.camera.image_height_px)
        self._condition = Condition()
        self._latest: FaceMeshInputFrame | None = None
        self._published_version = 0
        self._consumed_version = 0
        self._closed = False

    def publish_payload(self, payload: object) -> int:
        if not isinstance(payload, dict):
            raise ValueError("FaceMesh IPC payload must be a JSON object")
        frame_index = payload.get("frame_index")
        faces = payload.get("faces")
        encoded_jpeg = payload.get("frame_jpeg_base64")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("FaceMesh IPC payload has an invalid frame_index")
        if not isinstance(faces, list):
            raise ValueError("FaceMesh IPC payload must contain a faces list")
        if not isinstance(encoded_jpeg, str):
            raise ValueError("FaceMesh IPC payload must contain frame_jpeg_base64")
        observations = tuple(
            _parse_replay_face(face, frame_index=frame_index, face_index=face_index)
            for face_index, face in enumerate(faces)
        )
        frame = self._decode_frame(encoded_jpeg)
        with self._condition:
            if self._closed:
                raise RuntimeError("FaceMesh IPC input is closed")
            self._latest = FaceMeshInputFrame(
                frame_bgr=frame,
                faces=observations,
                label="FaceMesh IPC",
                frame_index=frame_index,
            )
            self._published_version += 1
            self._condition.notify_all()
            return self._published_version

    def _decode_frame(self, encoded_jpeg: str) -> np.ndarray:
        try:
            encoded = base64.b64decode(encoded_jpeg, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("FaceMesh IPC frame_jpeg_base64 is invalid") from exc
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("FaceMesh IPC frame is not a decodable JPEG")
        size = (frame.shape[1], frame.shape[0])
        if size != self._expected_size:
            raise ValueError(
                "FaceMesh IPC frame dimensions must match the intrinsics "
                f"({size[0]}x{size[1]} != {self._expected_size[0]}x{self._expected_size[1]})"
            )
        return frame

    def next_frame(self) -> FaceMeshInputFrame:
        deadline = time.monotonic() + 3.0
        with self._condition:
            while not self._closed and (
                self._latest is None or self._published_version == self._consumed_version
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("waiting for a FaceMesh IPC frame timed out")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("FaceMesh IPC input is closed")
            assert self._latest is not None
            self._consumed_version = self._published_version
            return self._latest

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

    def sample(self) -> tuple[TrackingState, bytes]:
        started = time.perf_counter()
        input_frame = self._frame_source.next_frame()
        if input_frame.faces:
            face = max(input_frame.faces, key=lambda value: value.score)
            measurement = self._measurement_from_landmarks(
                face.landmarks_xy, face.score, len(input_frame.faces), input_frame.frame_bgr
            )
        else:
            measurement = self._fallback_measurement(0)
        diagnostics = (
            None if input_frame.frame_index is None else {"input_frame": input_frame.frame_index}
        )
        state = self._build_state(measurement, started, diagnostics=diagnostics)
        return state, self._encode_preview(input_frame.frame_bgr, measurement, input_frame.label)

    def close(self) -> None:
        self._frame_source.close()

    def _measurement_from_landmarks(
        self, landmarks: np.ndarray, confidence: float, face_count: int, frame: np.ndarray
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
    def _draw_landmarks(frame: np.ndarray, landmarks_xy: np.ndarray) -> None:
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
        self, frame: np.ndarray, measurement: _PoseMeasurement, label: str
    ) -> bytes:
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
