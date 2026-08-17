"""Publish live FaceMesh observations to a local head-coupled display server.

Run this script with the ``facemesh_tracking`` Python environment.  It deliberately uses
only Python's standard-library HTTP client for transport, so the producer does not import
the Python 3.13 display package or inherit its dependency constraints.
"""

from __future__ import annotations

import argparse
import http.client
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

#: This script normally runs in a separate Python 3.10/CUDA environment via
#: `just facemesh-ipc`, which does not set PYTHONPATH -- so `headcoupled_display.protocol`
#: (the only headcoupled_display module this script needs; deliberately dependency-free,
#: see protocol.py's own module docstring) is reached by inserting this repo's `src/`
#: directly, computed from this file's own location rather than an environment variable.
_PROTOCOL_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_PROTOCOL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTOCOL_SRC_DIR))

from headcoupled_display.protocol import (  # noqa: E402
    LANDMARK_INDICES,
    ControlPacket,
    encode_control_packet,
)


@dataclass
class IpcPublisher:
    """Persistent HTTP publisher with one reconnect attempt per request.

    Generic over a single endpoint URL and raw payload bytes. The producer keeps two
    independent instances -- control lane and preview lane (workdoc steps 36-38) -- each
    with its own connection, so a broken preview-lane socket can never affect the
    control lane's.
    """

    endpoint: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "http" or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http:// URL")
        self._host = parsed.netloc
        self._path = parsed.path or "/"
        if parsed.query:
            self._path = f"{self._path}?{parsed.query}"
        self._connection: http.client.HTTPConnection | None = None

    def publish_bytes(self, body: bytes, *, content_type: str) -> None:
        for attempt in range(2):
            try:
                connection = self._connection or http.client.HTTPConnection(self._host, timeout=3.0)
                self._connection = connection
                connection.request(
                    "POST",
                    self._path,
                    body=body,
                    headers={"Content-Type": content_type, "Content-Length": str(len(body))},
                )
                response = connection.getresponse()
                response_body = response.read().decode("utf-8", errors="replace")
                if 200 <= response.status < 300:
                    return
                raise RuntimeError(f"IPC server returned HTTP {response.status}: {response_body}")
            except OSError:
                self.close()
                if attempt:
                    raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None


def build_control_packet(
    face: Any,
    *,
    sequence: int,
    capture_monotonic_ns: int,
    capture_unix_ns: int,
    inference_monotonic_ns: int,
    inference_unix_ns: int,
) -> ControlPacket:
    """Build one control-lane packet from a detected face's dense landmarks.

    Only `LANDMARK_INDICES` -- the wire format's fixed 12-point subset, see protocol.py
    -- ever leaves this process on the control lane; the other ~466 dense points never
    cross the process boundary.
    """

    landmarks_px = tuple(
        (float(face.points[index, 0]), float(face.points[index, 1])) for index in LANDMARK_INDICES
    )
    return ControlPacket(
        landmarks_px=landmarks_px,
        score=float(face.score),
        sequence=sequence,
        capture_monotonic_ns=capture_monotonic_ns,
        capture_unix_ns=capture_unix_ns,
        inference_monotonic_ns=inference_monotonic_ns,
        inference_unix_ns=inference_unix_ns,
    )


#: Preview lane resolution (workdoc steps 37-38). Must exactly match the server's
#: `/api/input/facemesh/preview` contract (`jpeg_dimensions()` in tracking.py rejects
#: anything else with 422 rather than resizing it server-side). Recognition/PnP always
#: stay at the full `--width`/`--height` (1280x720 by default); this is a *separate*,
#: display-only contract, never derived from or substituted for it.
PREVIEW_WIDTH_PX = 640
PREVIEW_HEIGHT_PX = 360

#: Preview publish rate cap (workdoc steps 37-38). Control packets go out on every
#: processed frame; previews are throttled well below the ~27-30 FPS recognition rate
#: because a compressed thumbnail refreshing faster than it is ever displayed is wasted
#: encode/network work competing with the control lane for the same producer process.
PREVIEW_MAX_FPS = 10.0
PREVIEW_MIN_INTERVAL_S = 1.0 / PREVIEW_MAX_FPS


def _should_publish_preview(now_s: float, last_sent_s: float | None) -> bool:
    """True once at least `PREVIEW_MIN_INTERVAL_S` has passed since the last send.

    A separate, pure predicate (rather than inline in the main loop) so the 10 FPS cap
    can be checked against a virtual clock in tests, with no real `sleep`.
    """

    return last_sent_s is None or (now_s - last_sent_s) >= PREVIEW_MIN_INTERVAL_S


def encode_preview_frame(
    frame_bgr: np.ndarray,
    landmarks_xy: np.ndarray | None,
    label: str,
    *,
    jpeg_quality: int,
) -> bytes:
    """Resize a full-resolution BGR frame to the preview contract, draw the overlay, and
    JPEG-encode it. The server forwards these bytes untouched (workdoc steps 37-38), so
    any overlay has to be baked in here -- there is nowhere left downstream to draw it.
    """

    preview = cv2.resize(
        frame_bgr, (PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX), interpolation=cv2.INTER_AREA
    )
    if landmarks_xy is not None:
        scale_x = PREVIEW_WIDTH_PX / frame_bgr.shape[1]
        scale_y = PREVIEW_HEIGHT_PX / frame_bgr.shape[0]
        for x, y in landmarks_xy[::8, :2]:
            center = (round(float(x) * scale_x), round(float(y) * scale_y))
            cv2.circle(preview, center, 1, (90, 220, 170), -1)
    cv2.putText(
        preview, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 220, 170), 1, cv2.LINE_AA
    )
    encoded_ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded_ok:
        raise RuntimeError("failed to encode preview frame")
    return encoded.tobytes()


def publish_preview_best_effort(
    publisher: IpcPublisher, body: bytes, *, log: Callable[[str], None] = print
) -> bool:
    """Publish preview bytes, swallowing any failure so it can never stop the control
    lane (workdoc step 38). The return value is for logging/metrics only; callers must
    not branch production behavior on it.
    """

    try:
        publisher.publish_bytes(body, content_type="image/jpeg")
    except (OSError, RuntimeError) as exc:
        log(f"preview publish failed (control lane unaffected): {exc}")
        return False
    return True


#: Alignment landmark indices used to synthesize a detector-equivalent 5-point template
#: (iris left, iris right, nose tip, mouth-corner left, mouth-corner right) from the
#: previous frame's dense V2_478 mesh, in the same order as the detector's own keypoints.
#: This mapping was confirmed by comparing the detector's 5 points against V2_478
#: landmarks on test10.avi frame 41: point index 4 matched the detector's nose point
#: (d=1.8px) far better than index 1 (d=13.1px), which the earlier workdoc draft assumed.
ROI_ALIGNMENT_LANDMARK_INDICES: tuple[int, int, int, int, int] = (468, 473, 4, 61, 291)

#: The dense mesh needs at least this many points for ROI_ALIGNMENT_LANDMARK_INDICES to
#: be valid (i.e. it must include the two iris points from the V2_478 model).
_MIN_POINTS_FOR_ROI_ALIGNMENT = max(ROI_ALIGNMENT_LANDMARK_INDICES) + 1

#: A landmark-only ROI is trusted only while the previous frame's score stayed at or
#: above this. Matches UnifaceFaceMesh's / FaceMeshPipeline.create's own default
#: `landmark_threshold=0.5`, so this does not introduce a second, drifting notion of
#: "good enough" alongside the estimator's own acceptance threshold.
DEFAULT_MIN_LANDMARK_SCORE: float = 0.5

#: Conservative default: refresh (full-detect) every single frame, i.e. the temporal ROI
#: shortcut is off by default. Workdoc step 27 explores 1/2/3/5/8/10 on test10.avi and
#: picks a value from measured accuracy/latency, so this default is deliberately *not* a
#: performance guess -- it is the safe fallback if step 27 is never run.
DEFAULT_DETECTOR_REFRESH_INTERVAL = 1


@dataclass
class _RoiEstimateResult:
    """Landmark-only counterpart of ``facemesh_tracking.pipeline.FaceMeshResult``.

    Deliberately duck-typed (``boxes``, ``faces``, ``len()``) instead of importing the
    real class: the only facemesh_tracking symbol this module needs at runtime is
    ``geometry.BBox`` (imported lazily inside ``TemporalRoiRunner``, matching how
    ``main()`` below defers the whole ``facemesh_tracking`` import), so unit tests can
    exercise the full state machine with plain protocol fakes and no facemesh_tracking
    installation at all.
    """

    boxes: list[Any]
    faces: list[Any]

    def __len__(self) -> int:
        return len(self.faces)


class TemporalRoiRunner:
    """Single-person temporal ROI state machine (workdoc steps 25-26).

    Runs ``pipeline``'s full detector only when necessary; every other frame reuses the
    previous frame's dense landmarks to build a ROI and calls ``pipeline.estimator.
    estimate`` directly, skipping detection entirely. A full detector pass runs when
    any of:

    1. this is the first frame processed (no previous landmarks yet);
    2. the previous frame produced no face at all -- a miss is never carried forward as
       a "last known" ROI, so recovery always goes through the full detector;
    3. the previous frame's landmark score fell below ``min_landmark_score``;
    4. the ROI built from the previous frame's landmarks would extend past the frame
       edges before clipping, or collapse to zero area after clipping;
    5. ``detector_refresh_interval`` frames have elapsed since the last full detect.

    A missed frame is reported as an empty ``faces`` list on the very frame it happens,
    and clears the internal "last landmarks" state -- this class never re-publishes a
    stale, high-confidence pose from an earlier frame once the face is actually gone.

    ROI construction: propagating the anchor detector box, not the landmark box
    -----------------------------------------------------------------------------
    Every full detect ("refresh") caches an *anchor*: the detector's box (and its
    5-point alignment keypoints, if any) plus the landmark bounding box that same frame
    produced. Each subsequent landmark-only frame maps the anchor's detector box (and
    keypoints) through the per-axis scale+translate that takes the anchor's landmark box
    to the *current* (most recently available) landmark box, and uses the mapped result
    as the ROI. This is deliberate, not incidental: on test10.avi the detector's box and
    the plain landmark-point bounding box are measurably different shapes (mean
    (x1,y1,x2,y2) offset roughly (+2.5, -36, -3.6, -9.6) px, height ratio ~1.07, width
    ratio ~0.98) -- UniFace crops+aligns using the *detector's* box, so feeding the
    estimator a differently-shaped crop measurably drifts the resulting pose. An
    interval=2 sweep confirmed this: mean landmark error vs. a full-detect baseline was
    2.53px with a plain landmark-derived box, but 0.60px propagating the detector box
    instead. If the anchor's detector box had no keypoints, this falls back to
    synthesizing them from the current frame's own landmarks at
    ``ROI_ALIGNMENT_LANDMARK_INDICES``.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        detector_refresh_interval: int = DEFAULT_DETECTOR_REFRESH_INTERVAL,
        min_landmark_score: float = DEFAULT_MIN_LANDMARK_SCORE,
    ) -> None:
        if detector_refresh_interval < 1:
            raise ValueError(
                f"detector_refresh_interval must be >= 1, got {detector_refresh_interval!r}"
            )
        self._pipeline = pipeline
        self._refresh_interval = detector_refresh_interval
        self._min_score = min_landmark_score
        self._last_landmarks: Any | None = None
        self._frames_since_refresh = 0
        #: The most recent full detect's detector box/keypoints and landmark bounding
        #: box; None until the first full detect. See the class docstring.
        self._anchor_det_box: Any | None = None
        self._anchor_landmark_box: tuple[float, float, float, float] | None = None

    def process(self, frame: np.ndarray) -> Any:
        """Process one BGR frame, returning a ``FaceMeshResult``-shaped object."""
        height, width = frame.shape[:2]
        roi = (
            self._prepare_roi(self._last_landmarks, width, height)
            if self._last_landmarks is not None
            else None
        )
        result = (
            self._run_full_detect(frame)
            if self._needs_full_detect(roi)
            else self._run_landmark_only(frame, roi)
        )
        self._last_landmarks = result.faces[0] if result.faces else None
        return result

    def _needs_full_detect(self, roi: Any | None) -> bool:
        """Decide detect-vs-reuse. Kept separate from ROI/inference so it stays testable
        in isolation and under the project's cyclomatic-complexity budget on its own."""
        if self._last_landmarks is None:
            return True
        if self._last_landmarks.score < self._min_score:
            return True
        if roi is None:
            return True
        return self._frames_since_refresh >= self._refresh_interval

    def _run_full_detect(self, frame: np.ndarray) -> Any:
        result = self._pipeline.process(frame)
        self._frames_since_refresh = 1
        self._update_anchor(result.faces[0] if result.faces else None)
        return result

    def _run_landmark_only(self, frame: np.ndarray, roi: Any) -> _RoiEstimateResult:
        faces = self._pipeline.estimator.estimate(frame, [roi])
        self._frames_since_refresh += 1
        return _RoiEstimateResult(boxes=[roi], faces=faces)

    def _update_anchor(self, face: Any | None) -> None:
        """Cache a fresh full-detect's detector box/keypoints and landmark bounding box
        as the propagation reference for landmark-only frames, until the next full
        detect. Cleared to None on a miss -- harmless, since the very next frame is
        forced back to full-detect by ``_needs_full_detect`` (``_last_landmarks is
        None``), which refreshes the anchor before it would ever be read.
        """
        if face is None:
            self._anchor_det_box = None
            self._anchor_landmark_box = None
            return
        self._anchor_det_box = face.bbox
        x1, y1 = face.points[:, :2].min(axis=0)
        x2, y2 = face.points[:, :2].max(axis=0)
        self._anchor_landmark_box = (float(x1), float(y1), float(x2), float(y2))

    def _landmark_to_anchor_transform(
        self, xy: np.ndarray
    ) -> tuple[float, float, float, float] | None:
        """Per-axis ``scale, offset`` mapping an anchor-frame x/y to this frame's x/y:
        ``mapped = value * scale + offset``. Derived from how far the landmark bounding
        box's extent, on each axis independently, has moved/resized since the anchor.
        None if the current landmarks are degenerate on either axis (would divide by
        zero) or the anchor itself somehow was (defensive; a real detected face never
        produces a zero-size landmark box).
        """
        cur_x1, cur_y1 = xy.min(axis=0)
        cur_x2, cur_y2 = xy.max(axis=0)
        ax1, ay1, ax2, ay2 = self._anchor_landmark_box
        anchor_w, anchor_h = ax2 - ax1, ay2 - ay1
        if anchor_w <= 0 or anchor_h <= 0:
            return None
        scale_x = (cur_x2 - cur_x1) / anchor_w
        scale_y = (cur_y2 - cur_y1) / anchor_h
        return scale_x, scale_y, cur_x1 - ax1 * scale_x, cur_y1 - ay1 * scale_y

    def _map_anchor_box(
        self, transform: tuple[float, float, float, float], width: int, height: int
    ) -> tuple[float, float, float, float] | None:
        """Map the anchor detector box through ``transform``; None if it would extend
        past the frame before clipping (the face has drifted towards/out of an edge)."""
        scale_x, scale_y, offset_x, offset_y = transform
        det = self._anchor_det_box
        x1, y1 = det.x1 * scale_x + offset_x, det.y1 * scale_y + offset_y
        x2, y2 = det.x2 * scale_x + offset_x, det.y2 * scale_y + offset_y
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return None
        return x1, y1, x2, y2

    def _propagate_keypoints(
        self, transform: tuple[float, float, float, float], xy: np.ndarray
    ) -> np.ndarray | None:
        """Map the anchor's detector keypoints through ``transform``, or -- if the
        detector didn't provide any (``BBox.keypoints`` is None) -- fall back to
        synthesizing them from the current frame's own landmarks at
        ``ROI_ALIGNMENT_LANDMARK_INDICES``. None if that fallback is needed but ``xy``
        lacks the iris points it requires (not a 478-point/V2_478 mesh).
        """
        det_keypoints = self._anchor_det_box.keypoints
        if det_keypoints is not None:
            scale_x, scale_y, offset_x, offset_y = transform
            mapped = np.empty_like(det_keypoints, dtype=np.float32)
            mapped[:, 0] = det_keypoints[:, 0] * scale_x + offset_x
            mapped[:, 1] = det_keypoints[:, 1] * scale_y + offset_y
            return mapped
        if xy.shape[0] < _MIN_POINTS_FOR_ROI_ALIGNMENT:
            return None
        return xy[list(ROI_ALIGNMENT_LANDMARK_INDICES)].astype(np.float32)

    def _prepare_roi(self, landmarks: Any, width: int, height: int) -> Any | None:
        """Build this frame's ROI by propagating the anchor detector box (see the class
        docstring), or None to force a full detect. Deliberately touches the 478-point
        array only via cheap numpy views/reductions (min/max over an existing view, a
        5-row gather for the fallback keypoints) -- no full-array copy and no JSON
        serialisation happen on this hot path.
        """
        from facemesh_tracking.geometry import BBox

        if self._anchor_det_box is None or self._anchor_landmark_box is None:
            return None
        xy = landmarks.points[:, :2]  # view, not a copy
        transform = self._landmark_to_anchor_transform(xy)
        if transform is None:
            return None
        mapped = self._map_anchor_box(transform, width, height)
        if mapped is None:
            return None
        keypoints = self._propagate_keypoints(transform, xy)
        if keypoints is None:
            return None
        x1, y1, x2, y2 = mapped
        box = BBox(
            x1=round(x1),
            y1=round(y1),
            x2=round(x2),
            y2=round(y2),
            score=landmarks.score,
            keypoints=keypoints,
        ).clipped(width, height)
        return box if box.is_valid else None


class Pacing(str, Enum):  # noqa: UP042 - enum.StrEnum is Python 3.11+; this module targets 3.10
    """How a recorded-video ``FrameSource`` releases frames (workdoc step 28-29)."""

    #: Decode as fast as possible and stop at EOF. Never blocks on a wall-clock wait, so
    #: this is the safer choice for automated/CI/benchmark runs.
    ONESHOT = "oneshot"
    #: Sleep between frames to match the container's reported FPS, reproducing live
    #: camera timing. Requires the container to report a valid (>0) FPS.
    REALTIME = "realtime"


def _check_resolution(
    frame: np.ndarray, expected_width: int, expected_height: int, *, source: str
) -> None:
    """Raise unless a *decoded* frame's actual size matches ``--width``/``--height``.

    Checked against the frame OpenCV actually decoded, never against container/camera
    metadata: this producer must not trust header claims about resolution any more than
    it trusts header frame counts (test10.avi's header lies about both frame count and
    FPS -- 602 frames / 60 FPS claimed, 294 frames / unknown-true-FPS actually decode).
    """
    actual_height, actual_width = frame.shape[:2]
    if actual_width != expected_width or actual_height != expected_height:
        raise ValueError(
            f"{source}: decoded frame resolution {actual_width}x{actual_height} does not "
            f"match expected {expected_width}x{expected_height} "
            "(pass --width/--height matching this source; no implicit resize happens here)"
        )


@dataclass
class CameraFrameSource:
    """Live camera input via OpenCV/V4L2. ``device`` is a V4L2 path or numeric index."""

    device: int | str
    width: int
    height: int

    def __post_init__(self) -> None:
        self._capture = cv2.VideoCapture(self.device)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._capture.isOpened():
            raise RuntimeError(f"unable to open camera {self.device!r}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame. Unlike a file source, this never returns ``False``:

        a live camera failing to deliver a frame is an unexpected hardware fault, not an
        expected end-of-stream, so it is raised instead of being handed to the caller as
        an ordinary loop-termination signal.
        """
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError(f"camera frame capture failed for {self.device!r}")
        _check_resolution(frame, self.width, self.height, source=str(self.device))
        return True, frame

    def close(self) -> None:
        self._capture.release()


@dataclass
class VideoFileFrameSource:
    """Frames from a recorded video file (workdoc step 28-29, e.g. ``test10.avi``).

    End-of-stream is decided solely by ``cv2.VideoCapture.read()`` returning ``False``.
    The container header's frame count / FPS are read only for realtime pacing's actual
    interval, never as a substitute for "how many frames does this file have" -- e.g.
    test10.avi's header claims 602 frames / 60 FPS but only 294 frames actually decode.

    ``now_fn``/``sleep_fn`` are injectable (default ``time.perf_counter``/``time.sleep``)
    so unit tests can verify realtime pacing without a real wall-clock wait.
    """

    path: str
    width: int
    height: int
    pacing: Pacing = Pacing.ONESHOT
    now_fn: Callable[[], float] = time.perf_counter
    sleep_fn: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise RuntimeError(f"unable to open video file {self.path!r}")
        self._frame_interval = 0.0
        if self.pacing is Pacing.REALTIME:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError(
                    f"{self.path!r}: --pacing realtime requires a valid FPS, but the "
                    f"container reports {fps!r}"
                )
            self._frame_interval = 1.0 / fps
        self._next_due: float | None = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame, or ``(False, None)`` at EOF (expected, not an error)."""
        if self.pacing is Pacing.REALTIME and self._next_due is not None:
            remaining = self._next_due - self.now_fn()
            if remaining > 0:
                self.sleep_fn(remaining)
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        _check_resolution(frame, self.width, self.height, source=self.path)
        if self.pacing is Pacing.REALTIME:
            # Simple fixed-interval pacing relative to this read, not drift-compensated
            # against an absolute start time -- adequate for the short recordings this
            # producer targets (test10.avi replay, not long-running unattended capture).
            self._next_due = self.now_fn() + self._frame_interval
        return True, frame

    def close(self) -> None:
        self._capture.release()


def build_frame_source(
    source: str, *, width: int, height: int, pacing: Pacing
) -> CameraFrameSource | VideoFileFrameSource:
    """Dispatch ``--source`` to a camera or recorded-video input.

    Three explicit, non-overlapping rules -- never an implicit fallback between them:

    1. a purely decimal string (e.g. ``"0"``) is a numeric V4L2 device index;
    2. an existing regular file on disk is a recorded video (``--pacing`` applies only
       to this case; a camera already paces itself in real time);
    3. anything else (e.g. ``"/dev/video0"``) is a V4L2 device path.
    """
    if source.isdecimal():
        return CameraFrameSource(int(source), width=width, height=height)
    if Path(source).is_file():
        return VideoFileFrameSource(source, width=width, height=height, pacing=pacing)
    return CameraFrameSource(source, width=width, height=height)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Split out so tests can inspect it without invoking main()."""
    parser = argparse.ArgumentParser(description="Publish live FaceMesh frames to headcoupled IPC")
    parser.add_argument(
        "--source",
        default="/dev/video0",
        help=(
            "Frame input (workdoc steps 28-29): a V4L2 path (e.g. /dev/video0, the "
            "default), a numeric camera index (e.g. 0), or the path to an existing "
            "recorded video file (e.g. recordings/test10.avi). Dispatch is explicit: a "
            "purely decimal string is always a camera index, an existing file is always "
            "a recorded video, anything else is a camera device path -- never a guess."
        ),
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Deprecated alias for --source, kept for backward compatibility with "
        "existing callers. If given (non-None), it overrides --source.",
    )
    parser.add_argument(
        "--pacing",
        choices=tuple(p.value for p in Pacing),
        default=Pacing.ONESHOT.value,
        help=(
            "Playback pacing for a recorded-video --source; has no effect on a camera "
            "source (live hardware already paces itself). 'oneshot' (default) decodes "
            "as fast as possible and stops at EOF, e.g. for benchmarks/smoke tests. "
            "'realtime' sleeps between frames to match the file's reported FPS, "
            "reproducing live timing. The default is 'oneshot' because it never blocks "
            "on a wall-clock wait, which is the safer choice for unattended runs."
        ),
    )
    parser.add_argument(
        "--detector-refresh-interval",
        type=int,
        default=DEFAULT_DETECTOR_REFRESH_INTERVAL,
        help=(
            "Run the full face detector once every N frames; frames in between reuse "
            "the previous frame's landmarks for a landmark-only ROI estimate instead "
            "(temporal ROI, workdoc steps 25-26). Default 1 means 'every frame' (the "
            "ROI shortcut is off) -- this is a conservative default, not a performance "
            "recommendation; workdoc step 27 sweeps 1/2/3/5/8/10 on test10.avi and "
            "picks a value from measured accuracy/latency."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/api/input/facemesh",
        help=(
            "Base URL for the two independent IPC lanes (workdoc steps 36-38), not an "
            "endpoint by itself: control packets POST to '<endpoint>/control' "
            "(application/octet-stream, see headcoupled_display.protocol) and previews "
            "POST to '<endpoint>/preview' (image/jpeg)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "Execution backend. TensorRT is a non-goal here (R-PERF-2): "
            "'libnvinfer.so.10' is absent, so it would silently fall back to CUDA "
            "instead of failing, which is exactly the kind of fallback this producer "
            "must not hide. Use --backend cuda (the default) for production runs; "
            "--backend cpu is only for debugging without a GPU and skips the CUDA "
            "provider check below."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        choices=range(1, 101),
        help="JPEG quality for the preview lane only (workdoc step 37-38). The control "
        "lane never carries pixels.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = until Ctrl-C)"
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _actual_providers(component: Any, label: str) -> list[str]:
    """Read the *actual* ONNX Runtime providers active for one pipeline stage.

    ``component.providers`` (set by facemesh_tracking's UniFace wrappers) is only the
    provider list ONNX Runtime was *asked* to try. onnxruntime silently keeps walking that
    list (e.g. down to CPUExecutionProvider) when an earlier entry fails to load, so the
    requested list is not evidence that CUDA is actually running.

    facemesh_tracking has no public accessor for the live session, so this reaches through
    UniFace's own ``_model.session`` attribute layout instead (UniFace's ``session`` is
    itself public; only the wrapper's ``_model`` is internal). If that internal layout
    ever changes, this raises rather than silently trusting the requested provider list as
    a stand-in for reality.
    """
    session = getattr(getattr(component, "_model", None), "session", None)
    get_providers = getattr(session, "get_providers", None)
    if get_providers is None:
        raise RuntimeError(
            f"{label}: cannot reach an ONNX Runtime session via '_model.session' on "
            f"{type(component).__name__!r}; refusing to trust the requested provider list "
            "as evidence of the actual execution provider"
        )
    return list(get_providers())


def assert_cuda_providers(pipeline: Any) -> dict[str, list[str]]:
    """Fail startup unless both the detector and estimator sessions actually run on CUDA.

    ``--backend cuda`` only requests CUDAExecutionProvider; onnxruntime falls back to
    CPUExecutionProvider without raising when CUDA cannot load (see
    ``facemesh_tracking.runtime.providers_for``). Treating that fallback as success would
    silently violate R-PERF-2 / TR-4, so this inspects the real session providers instead
    of the requested list and names every stage that did not lead with CUDA.

    Returns the actual provider list per stage so the caller can log it once.
    """
    actual: dict[str, list[str]] = {}
    failed: list[str] = []
    for label, component in (("detector", pipeline.detector), ("estimator", pipeline.estimator)):
        providers = _actual_providers(component, label)
        actual[label] = providers
        if not providers or providers[0] != "CUDAExecutionProvider":
            failed.append(f"{label} (actual={providers!r})")
    if failed:
        raise RuntimeError(
            "CUDA execution provider is not active for: "
            + ", ".join(failed)
            + " - CPU fallback is not treated as success (R-PERF-2)"
        )
    return actual


def main() -> None:
    args = parse_args()
    from facemesh_tracking.pipeline import FaceMeshPipeline
    from facemesh_tracking.runtime import Backend

    pipeline = FaceMeshPipeline.create(backend=Backend(args.backend))
    if args.backend == "cuda":
        # Print the actual (not merely requested) providers exactly once, so a CPU
        # fallback is both fatal and visible in the log leading up to the failure.
        print(f"providers (actual): {assert_cuda_providers(pipeline)}")
    else:
        print(f"providers (requested, backend={args.backend}): {pipeline.detector.providers}")
    # --camera is a deprecated alias for --source; only override when explicitly given.
    source_value = args.camera if args.camera is not None else args.source
    pacing = Pacing(args.pacing)
    frame_source = build_frame_source(
        source_value, width=args.width, height=args.height, pacing=pacing
    )
    runner = TemporalRoiRunner(pipeline, detector_refresh_interval=args.detector_refresh_interval)
    endpoint_base = args.endpoint.rstrip("/")
    control_publisher = IpcPublisher(f"{endpoint_base}/control")
    preview_publisher = IpcPublisher(f"{endpoint_base}/preview")
    print(
        f"source: {source_value} (pacing={pacing.value}) -> "
        f"control={control_publisher.endpoint} preview={preview_publisher.endpoint}"
    )
    frame_index = 0
    last_preview_sent_s: float | None = None
    started = time.perf_counter()
    try:
        while not args.max_frames or frame_index < args.max_frames:
            ok, frame = frame_source.read()
            if not ok:
                print(f"frame source exhausted (EOF) after {frame_index} frames")
                break
            capture_monotonic_ns = time.perf_counter_ns()
            capture_unix_ns = time.time_ns()
            result = runner.process(frame)
            inference_monotonic_ns = time.perf_counter_ns()
            inference_unix_ns = time.time_ns()
            face = result.faces[0] if result.faces else None
            if face is not None:
                # A miss (face is None) is simply not published this frame: the control
                # wire format only carries finite landmark coordinates (protocol.py
                # rejects NaN/Inf), so there is nothing valid to send. A *sustained* miss
                # is surfaced by the server's own stale-input timeout
                # (RuntimeCoordinator.stale_after_s); a single skipped frame here is
                # invisible at video frame rate.
                packet = build_control_packet(
                    face,
                    sequence=frame_index,
                    capture_monotonic_ns=capture_monotonic_ns,
                    capture_unix_ns=capture_unix_ns,
                    inference_monotonic_ns=inference_monotonic_ns,
                    inference_unix_ns=inference_unix_ns,
                )
                control_publisher.publish_bytes(
                    encode_control_packet(packet), content_type="application/octet-stream"
                )
            now_s = time.perf_counter()
            if _should_publish_preview(now_s, last_preview_sent_s):
                last_preview_sent_s = now_s
                running_fps = frame_index / max(now_s - started, 1e-6)
                landmarks_xy = face.points[:, :2] if face is not None else None
                preview_bytes = encode_preview_frame(
                    frame,
                    landmarks_xy,
                    f"IPC {running_fps:.1f} FPS",
                    jpeg_quality=args.jpeg_quality,
                )
                # Preview publish failures must never stop the control lane above.
                publish_preview_best_effort(preview_publisher, preview_bytes)
            frame_index += 1
            if frame_index % 30 == 0:
                fps = frame_index / max(time.perf_counter() - started, 1e-6)
                print(f"published={frame_index}  faces={len(result.faces)}  {fps:.1f} FPS")
    finally:
        frame_source.close()
        control_publisher.close()
        preview_publisher.close()


if __name__ == "__main__":
    main()
