from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector, DetectionObservation
from tagcal.models import CaptureSpec

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CameraDevice:
    index: int
    width: int
    height: int
    fps: float
    backend: str
    fourcc: str


def fourcc_text(value: float) -> str:
    """Decode a FOURCC property value into its four characters."""
    code = int(value)
    if code <= 0:
        return "unknown"
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4)).strip()


def describe_capture(capture: cv2.VideoCapture) -> str:
    """Summarise what the driver actually granted, which is often not what was asked."""
    return (
        f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"{capture.get(cv2.CAP_PROP_FPS):.2f}fps "
        f"{fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))}"
    )


def actual_fps(capture: cv2.VideoCapture, fallback: float) -> float:
    """Frame rate the device reports, ignoring values it clearly cannot mean."""
    reported = float(capture.get(cv2.CAP_PROP_FPS))
    return reported if 1.0 < reported <= 240.0 else fallback


@dataclass(frozen=True, slots=True)
class RecordingResult:
    video_path: Path
    frames_written: int
    width: int
    height: int
    fps: float
    duration_seconds: float


def preferred_backend_name() -> str:
    """Name of the platform's native capture backend, as `getBackendName` reports it."""
    return {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
        cv2.CAP_V4L2: "V4L2",
    }.get(preferred_backend(), "ANY")


def preferred_backend() -> int:
    system = platform.system()
    if system == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if system == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    if system == "Linux" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(spec: CaptureSpec) -> cv2.VideoCapture:
    backends = [preferred_backend(), cv2.CAP_ANY]
    attempted: set[int] = set()
    capture: cv2.VideoCapture | None = None
    for backend in backends:
        if backend in attempted:
            continue
        attempted.add(backend)
        candidate = cv2.VideoCapture(spec.camera_index, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()

    if capture is None:
        raise RuntimeError(f"Unable to open camera index {spec.camera_index}")

    # The pixel format has to be selected before the frame size: V4L2 validates the
    # requested size against the format that is currently active.
    if spec.input_fourcc is not None:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*spec.input_fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(spec.width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(spec.height))
    capture.set(cv2.CAP_PROP_FPS, float(spec.fps))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2.0)

    ok, _ = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"Camera {spec.camera_index} opened but did not return a frame")
    return capture


def mode_mismatch(capture: cv2.VideoCapture, spec: CaptureSpec) -> str | None:
    """Describe how the granted capture mode differs from the requested one.

    A device node that is not the real capture node accepts the request and then
    ignores it, so this is usually the first sign the wrong index was picked.
    """
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    granted = fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
    problems: list[str] = []
    if (width, height) != (spec.width, spec.height):
        problems.append(f"解像度 {spec.width}x{spec.height} -> {width}x{height}")
    if spec.input_fourcc is not None and granted.upper() != spec.input_fourcc.upper():
        problems.append(f"フォーマット {spec.input_fourcc} -> {granted}")
    if not problems:
        return None
    return (
        "要求が反映されませんでした（" + " / ".join(problems) + "）。"
        "別のカメラindexを選ぶか、対応する解像度を指定してください。"
    )


def probe_cameras(max_index: int = 10, input_fourcc: str | None = "MJPG") -> list[CameraDevice]:
    """List openable camera indices, reporting the format actually granted.

    The format is requested here too, because the frame rate a device advertises
    depends on it: the same camera reports 30fps as MJPG and 5fps as YUYV.
    """
    if max_index < 1:
        raise ValueError("max_index must be at least 1")
    devices: list[CameraDevice] = []
    backend = preferred_backend()
    for index in range(max_index):
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not capture.isOpened():
            capture.release()
            continue
        if input_fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*input_fourcc))
        ok, frame = capture.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            backend_name = capture.getBackendName() if hasattr(capture, "getBackendName") else ""
            devices.append(
                CameraDevice(
                    index=index,
                    width=int(width),
                    height=int(height),
                    fps=fps if fps > 0 else 0.0,
                    backend=backend_name,
                    fourcc=fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)),
                )
            )
        capture.release()
    return devices


def timestamped_video_path(output_dir: Path, stem: str = "capture", suffix: str = ".mp4") -> Path:
    """Name a recording by local start time.

    Recordings are expensive to repeat -- the operator has to stand up and move the
    camera again -- so a session must never overwrite the one before it.
    """
    return output_dir / f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{suffix}"


def create_video_writer(
    path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*codec),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Unable to create video writer for {path} using codec {codec}. "
            "Try --codec MJPG with an .avi output path."
        )
    return writer


def record_video(
    output_path: Path,
    spec: CaptureSpec,
    *,
    detector: AprilTagDetector | None = None,
    progress: ProgressCallback | None = None,
) -> RecordingResult:
    notify = progress or (lambda _: None)
    capture = open_camera(spec)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = actual_fps(capture, spec.fps)

    writer = create_video_writer(output_path, width, height, fps, spec.codec)
    started = time.monotonic()
    frames_written = 0
    latest_observation: DetectionObservation | None = None
    preview_enabled = spec.preview
    mismatch = mode_mismatch(capture, spec)
    if mismatch is not None:
        notify(f"警告: {mismatch}")
    notify(
        f"Recording camera {spec.camera_index} at {describe_capture(capture)}. "
        "Press Q or Esc to stop."
    )

    try:
        while True:
            ok, captured = capture.read()
            if not ok or captured is None:
                raise RuntimeError("Camera frame acquisition failed during recording")
            frame = as_uint8(captured)
            writer.write(frame)
            frames_written += 1

            if detector is not None and frames_written % 3 == 1:
                latest_observation = detector.detect(frame)

            if preview_enabled:
                elapsed = time.monotonic() - started
                status = f"REC {elapsed:6.1f}s  frames={frames_written}"
                preview_frame: NDArray[np.uint8]
                if detector is None:
                    preview_frame = frame.copy()
                    cv2.putText(
                        preview_frame,
                        status,
                        (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    preview_frame = detector.draw_overlay(frame, latest_observation, status)
                try:
                    cv2.imshow("tagcal recording", preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break
                except cv2.error:
                    preview_enabled = False
                    notify("OpenCV preview is unavailable; recording continues without a window.")

            elapsed = time.monotonic() - started
            if spec.duration_seconds is not None and elapsed >= spec.duration_seconds:
                break
            if frames_written % max(1, round(fps * 5.0)) == 0:
                notify(f"Recorded {elapsed:.1f} seconds ({frames_written} frames)")
    except KeyboardInterrupt:
        notify("Recording interrupted by user.")
    finally:
        duration = time.monotonic() - started
        writer.release()
        capture.release()
        if preview_enabled:
            cv2.destroyWindow("tagcal recording")

    if frames_written == 0:
        raise RuntimeError("No frames were recorded")
    notify(f"Video saved: {output_path}")
    return RecordingResult(
        video_path=output_path,
        frames_written=frames_written,
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
    )
