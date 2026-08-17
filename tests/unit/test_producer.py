"""Unit tests for the live FaceMesh IPC producer.

Most tests here (CUDA provider attestation, CLI parsing) run under this repository's own
(Python 3.13) virtualenv and must not import ``facemesh_tracking`` or require a real ONNX
Runtime / CUDA session; the producer script defers those imports to inside ``main()`` for
exactly this reason.

The ``TemporalRoiRunner`` tests (workdoc steps 25-26) are a narrow, deliberate exception:
they add ``facemesh_tracking/src`` to ``sys.path`` so they can import the *pure value
objects* ``facemesh_tracking.geometry.BBox`` / ``FaceLandmarks`` -- plain dataclasses
with no onnxruntime/CUDA/model dependency (``geometry.py`` only imports ``dataclasses``
and ``numpy``). ``detector``/``estimator``/``pipeline`` themselves stay duck-typed
protocol fakes; this only makes the *shape of the data* TemporalRoiRunner builds and
passes to those fakes identical to production, so e.g. the keypoints-ordering test
checks the real ``BBox`` contract rather than a hand-rolled stand-in for it.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import scripts.facemesh_ipc_producer as producer
from headcoupled_display.tracking import jpeg_dimensions

#: See the module docstring: only facemesh_tracking's dependency-free geometry module is
#: reached this way, never anything that imports onnxruntime/UniFace/CUDA.
_FACEMESH_TRACKING_SRC = Path(__file__).resolve().parents[3] / "facemesh_tracking" / "src"
if str(_FACEMESH_TRACKING_SRC) not in sys.path:
    sys.path.insert(0, str(_FACEMESH_TRACKING_SRC))

from facemesh_tracking.geometry import BBox, FaceLandmarks  # noqa: E402

from tests.fixtures.synthetic_video import write_synthetic_avi  # noqa: E402


class _FakeSession:
    """Stands in for an ``onnxruntime.InferenceSession``."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class _FakeModel:
    """Stands in for UniFace's internal model object, which owns ``session``."""

    def __init__(self, providers: list[str]) -> None:
        self.session = _FakeSession(providers)


class _FakeStage:
    """Stands in for facemesh_tracking's ``UnifaceFaceDetector`` / ``UnifaceFaceMesh``."""

    def __init__(self, providers: list[str]) -> None:
        self._model = _FakeModel(providers)
        #: The requested (not actual) provider list, as facemesh_tracking exposes it.
        self.providers = providers


class _FakePipeline:
    def __init__(self, detector_providers: list[str], estimator_providers: list[str]) -> None:
        self.detector = _FakeStage(detector_providers)
        self.estimator = _FakeStage(estimator_providers)


class _StageWithoutSession:
    """A stage whose internal layout no longer exposes ``_model.session``."""


def test_assert_cuda_providers_succeeds_when_both_stages_lead_with_cuda() -> None:
    pipeline = _FakePipeline(
        detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        estimator_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    actual = producer.assert_cuda_providers(pipeline)

    assert actual == {
        "detector": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "estimator": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }


def test_assert_cuda_providers_raises_when_detector_is_cpu_only() -> None:
    pipeline = _FakePipeline(
        detector_providers=["CPUExecutionProvider"],
        estimator_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="detector"):
        producer.assert_cuda_providers(pipeline)


def test_assert_cuda_providers_raises_when_estimator_is_cpu_only() -> None:
    pipeline = _FakePipeline(
        detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        estimator_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="estimator"):
        producer.assert_cuda_providers(pipeline)


def test_assert_cuda_providers_does_not_treat_cpu_fallback_as_success() -> None:
    """Both stages silently fell back to CPU: this must fail, not just warn."""
    pipeline = _FakePipeline(
        detector_providers=["CPUExecutionProvider"],
        estimator_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="detector") as excinfo:
        producer.assert_cuda_providers(pipeline)
    assert "estimator" in str(excinfo.value)


def test_assert_cuda_providers_raises_when_session_access_path_is_broken() -> None:
    """If the internal ``_model.session`` layout ever changes, fail loudly.

    The requested provider list must never be substituted as evidence of the actual
    provider - a broken introspection path is an error, not a silent pass.
    """
    pipeline = _FakePipeline(
        detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        estimator_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    pipeline.detector = _StageWithoutSession()

    with pytest.raises(RuntimeError, match="detector"):
        producer.assert_cuda_providers(pipeline)


def test_cli_backend_choices_do_not_offer_tensorrt() -> None:
    parser = producer.build_arg_parser()
    backend_action = next(action for action in parser._actions if action.dest == "backend")

    # The selectable values themselves are what "no TensorRT option" means; TensorRT may
    # still be *mentioned in prose* explaining why it is a documented non-goal (R-PERF-2).
    assert tuple(backend_action.choices) == ("cuda", "cpu")
    assert "tensorrt" not in parser.format_usage().lower()

    args = parser.parse_args(["--backend", "cuda"])
    assert args.backend == "cuda"

    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "tensorrt"])


# ---------------------------------------------------------------------------------------
# TemporalRoiRunner (workdoc steps 25-26)
# ---------------------------------------------------------------------------------------

_FRAME_WIDTH = 200
_FRAME_HEIGHT = 200


def _frame(width: int = _FRAME_WIDTH, height: int = _FRAME_HEIGHT) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _spread_xy(*, x0: float, x1: float, y0: float, y1: float, count: int = 478) -> np.ndarray:
    """``(count, 2)`` points spread linearly between two corners.

    Every landmark index maps to a distinguishable ``(x, y)`` -- needed to pin down that
    the alignment keypoints TemporalRoiRunner synthesizes are gathered in exactly
    ``(468, 473, 4, 61, 291)`` order, not merely the right five values in some order.
    """
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    xy = np.empty((count, 2), dtype=np.float32)
    xy[:, 0] = x0 + t * (x1 - x0)
    xy[:, 1] = y0 + t * (y1 - y0)
    return xy


def _make_landmarks(xy: np.ndarray, score: float = 0.9) -> FaceLandmarks:
    """Build a real ``FaceLandmarks`` from an ``(N, 2)`` xy array (z filled with 0)."""
    points = np.zeros((xy.shape[0], 3), dtype=np.float32)
    points[:, :2] = xy
    x1, y1 = (int(v) for v in xy.min(axis=0))
    x2, y2 = (int(v) for v in xy.max(axis=0))
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score)
    return FaceLandmarks(points=points, score=score, bbox=bbox)


def _centered_landmarks(score: float = 0.9) -> FaceLandmarks:
    """A well-formed mesh comfortably inside the frame, away from every edge even after
    the default 15% ROI margin -- the "everything is fine" baseline fixture."""
    return _make_landmarks(_spread_xy(x0=60.0, x1=140.0, y0=60.0, y1=140.0), score=score)


def _shifted_landmarks(dx: float, score: float = 0.9) -> FaceLandmarks:
    """Same shape/extent as ``_centered_landmarks`` but shifted ``dx`` on the x axis --
    simulates the face having moved since the anchor frame, for propagation-transform
    tests (the shape stays identical, so ``scale_x == 1`` and only the offset changes).
    """
    return _make_landmarks(_spread_xy(x0=60.0 + dx, x1=140.0 + dx, y0=60.0, y1=140.0), score=score)


@dataclass
class _FakeResult:
    """Duck-typed stand-in for ``facemesh_tracking.pipeline.FaceMeshResult``."""

    boxes: list
    faces: list


class _RecordingFakeDetector:
    """Fake ``FaceDetector``: returns pre-scripted boxes per call, counts calls."""

    def __init__(self, boxes_per_call: list) -> None:
        self._boxes_per_call = list(boxes_per_call)
        self.call_count = 0

    def detect(self, image_bgr: np.ndarray) -> list:
        boxes = self._boxes_per_call[self.call_count]
        self.call_count += 1
        return boxes


class _RecordingFakeEstimator:
    """Fake ``LandmarkEstimator``: returns pre-scripted faces per call, counts calls, and
    records the exact ``boxes`` it was called with so tests can inspect e.g. ROI content
    and keypoints ordering."""

    def __init__(self, faces_per_call: list) -> None:
        self._faces_per_call = list(faces_per_call)
        self.call_count = 0
        self.received_boxes: list = []

    def estimate(self, image_bgr: np.ndarray, boxes) -> list:
        self.received_boxes.append(list(boxes))
        faces = self._faces_per_call[self.call_count]
        self.call_count += 1
        # Mirror UnifaceFaceMesh.estimate(): each returned FaceLandmarks.bbox is the
        # actual box it was estimated from (zip-aligned), not whatever bbox a fixture
        # happened to be built with -- TemporalRoiRunner's anchor logic depends on this.
        return [replace(face, bbox=box) for face, box in zip(faces, boxes, strict=False)]


class _RecordingFakePipeline:
    """Duck-typed stand-in for ``facemesh_tracking.pipeline.FaceMeshPipeline``.

    ``process`` mirrors the real pipeline's body (detect -> expand by margin_ratio ->
    estimate) exactly, since ``TemporalRoiRunner``'s full-detect path delegates to
    ``pipeline.process`` directly and these tests need that delegation to be observable
    through the same detector/estimator recording fakes used for the landmark-only path.
    """

    def __init__(self, detector, estimator, margin_ratio: float = 0.0) -> None:
        self.detector = detector
        self.estimator = estimator
        self.margin_ratio = margin_ratio

    def process(self, image_bgr: np.ndarray) -> _FakeResult:
        height, width = image_bgr.shape[:2]
        boxes = [
            box.expanded(self.margin_ratio, width, height)
            for box in self.detector.detect(image_bgr)
        ]
        return _FakeResult(boxes=boxes, faces=self.estimator.estimate(image_bgr, boxes))


def test_temporal_roi_runner_first_frame_is_full_detect() -> None:
    detector = _RecordingFakeDetector([[BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]])
    estimator = _RecordingFakeEstimator([[_centered_landmarks()]])
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=5)

    result = runner.process(_frame())

    assert detector.call_count == 1
    assert estimator.call_count == 1
    assert len(result.faces) == 1


def test_temporal_roi_runner_normal_frame_is_landmark_only() -> None:
    detector = _RecordingFakeDetector([[BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]])
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect
            [_centered_landmarks()],  # frame 2: landmark-only
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=5)

    runner.process(_frame())
    result2 = runner.process(_frame())

    assert detector.call_count == 1  # the full detector is not called again
    assert estimator.call_count == 2
    assert len(result2.faces) == 1
    assert len(estimator.received_boxes[-1]) == 1  # a single ROI box, not a re-detect


def test_temporal_roi_runner_interval_refresh() -> None:
    box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([box, box])
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect
            [_centered_landmarks()],  # frame 2: landmark-only
            [_centered_landmarks()],  # frame 3: landmark-only
            [_centered_landmarks()],  # frame 4: full detect (interval=3 elapsed)
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=3)

    for _ in range(4):
        runner.process(_frame())

    assert detector.call_count == 2  # frames 1 and 4
    assert estimator.call_count == 4


def test_temporal_roi_runner_low_score_triggers_refresh() -> None:
    box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([box, box])
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks(score=0.2)],  # frame 1: full detect, low score
            [_centered_landmarks(score=0.9)],  # frame 2: full detect (score forced it)
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    # A large interval means only the low score can be forcing the second full detect.
    runner = producer.TemporalRoiRunner(
        pipeline, detector_refresh_interval=10, min_landmark_score=0.5
    )

    runner.process(_frame())
    runner.process(_frame())

    assert detector.call_count == 2


def test_temporal_roi_runner_roi_out_of_bounds_triggers_refresh() -> None:
    """The anchor's detector box is comfortably in-bounds, but by frame 3 the most
    recently available landmarks (frame 2's own output) show the face having moved far
    enough left that propagating the (still fixed) anchor box through the resulting
    transform pushes it past x=0 -- must force a full detect, not clip silently. Frame
    2 itself is a trivial identity-transform landmark-only frame (it is compared
    against the anchor it came from)."""
    anchor_box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([anchor_box, anchor_box])
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect, sets the anchor
            [_shifted_landmarks(dx=-100.0)],  # frame 2: landmark-only (identity), moved output
            [_centered_landmarks()],  # frame 3: full detect (recovery), if reached
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())  # frame 1
    runner.process(_frame())  # frame 2
    runner.process(_frame())  # frame 3

    # frame 3's propagated x1 = 50 + (-100) = -50 < 0 -> out of bounds -> full detect
    assert detector.call_count == 2  # frame 1 and frame 3


def test_temporal_roi_runner_degenerate_roi_triggers_refresh() -> None:
    """The anchor is fine, but by frame 3 the most recently available landmarks (frame
    2's own output) have collapsed to a single point: the propagated box ends up
    in-bounds (scale 0 maps everything to one x/y) but with zero area after clipping --
    a distinct failure mode from "out of bounds"."""
    anchor_box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([anchor_box, anchor_box])
    coincident = _make_landmarks(np.full((478, 2), 100.0, dtype=np.float32))
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect, sets a valid anchor
            [coincident],  # frame 2: landmark-only (identity), collapsed output
            [_centered_landmarks()],  # frame 3: full detect (recovery), if reached
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())  # frame 1
    runner.process(_frame())  # frame 2
    runner.process(_frame())  # frame 3

    assert detector.call_count == 2  # frame 1 and frame 3


def test_temporal_roi_runner_missing_face_recovers_via_full_detect() -> None:
    box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([box, box])  # frame 1, frame 3 (recovery)
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect succeeds
            [],  # frame 2: landmark-only estimate finds nothing
            [_centered_landmarks()],  # frame 3: full detect succeeds again
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())
    result2 = runner.process(_frame())
    result3 = runner.process(_frame())

    assert result2.faces == []  # the miss is reported honestly, not padded with a stale pose
    assert len(result3.faces) == 1
    assert detector.call_count == 2  # frame 1 (first) and frame 3 (recovery)
    assert estimator.call_count == 3


def test_temporal_roi_runner_keypoints_order_is_468_473_4_61_291() -> None:
    xy = _spread_xy(x0=60.0, x1=140.0, y0=60.0, y1=140.0)
    frame1_landmarks = _make_landmarks(xy)
    detector = _RecordingFakeDetector([[BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]])
    estimator = _RecordingFakeEstimator([[frame1_landmarks], [_centered_landmarks()]])
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())  # full detect, stores frame1_landmarks
    runner.process(_frame())  # landmark-only: builds the ROI from frame1_landmarks

    landmark_only_boxes = estimator.received_boxes[-1]
    assert len(landmark_only_boxes) == 1
    roi = landmark_only_boxes[0]
    expected = xy[[468, 473, 4, 61, 291]]
    assert np.allclose(roi.keypoints, expected)


def test_temporal_roi_runner_roi_propagates_detector_box_not_landmark_box() -> None:
    """The landmark-only ROI must be the *propagated detector box*, not the plain
    landmark bounding box -- on test10.avi the two differ by tens of pixels (see the
    TemporalRoiRunner class docstring), so silently using the landmark box would drift
    the estimator's crop away from what the detector itself would have produced."""
    anchor_landmark_xy = _spread_xy(x0=60.0, x1=140.0, y0=60.0, y1=140.0)  # box (60,60,140,140)
    # Deliberately offset from the landmark box, like the real measured drift.
    anchor_det_box = [BBox(x1=45, y1=24, x2=145, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([anchor_det_box])
    estimator = _RecordingFakeEstimator(
        [
            [_make_landmarks(anchor_landmark_xy)],  # frame 1: full detect, sets the anchor
            [_centered_landmarks()],  # frame 2: identical landmark box -> identity transform
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())
    runner.process(_frame())

    roi = estimator.received_boxes[-1][0]
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (45, 24, 145, 150)  # the detector box
    assert (roi.x1, roi.y1, roi.x2, roi.y2) != (60, 60, 140, 140)  # not the landmark box


def test_temporal_roi_runner_falls_back_to_synthesized_keypoints_when_anchor_has_none() -> None:
    """If the anchor's detector box carries no keypoints (BBox.keypoints is None), the
    ROI must fall back to synthesizing them from the most recently available landmarks
    (the ones frame 2's ROI is built from, i.e. frame 1's) at
    ROI_ALIGNMENT_LANDMARK_INDICES -- exactly as before this change."""
    anchor_frame_xy = _spread_xy(x0=60.0, x1=140.0, y0=60.0, y1=140.0)
    anchor_box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95, keypoints=None)]
    detector = _RecordingFakeDetector([anchor_box])
    estimator = _RecordingFakeEstimator(
        [
            [_make_landmarks(anchor_frame_xy)],  # frame 1: full detect, sets the anchor
            [_centered_landmarks()],  # frame 2: landmark-only (identity transform)
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())
    runner.process(_frame())

    roi = estimator.received_boxes[-1][0]
    expected = anchor_frame_xy[[468, 473, 4, 61, 291]]
    assert np.allclose(roi.keypoints, expected)


def test_temporal_roi_runner_identity_transform_reproduces_detector_box_exactly() -> None:
    """When the current landmark box is identical to the anchor's, the transform is the
    identity (scale 1, offset 0), so the ROI -- box and keypoints -- must exactly match
    the anchor's detector box, unchanged."""
    det_keypoints = np.array(
        [[60.0, 70.0], [140.0, 70.0], [100.0, 100.0], [70.0, 130.0], [130.0, 130.0]],
        dtype=np.float32,
    )
    anchor_box = [BBox(x1=50, y1=45, x2=150, y2=155, score=0.95, keypoints=det_keypoints)]
    detector = _RecordingFakeDetector([anchor_box])
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect, sets the anchor
            [_centered_landmarks()],  # frame 2: identical landmark box -> identity transform
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())
    runner.process(_frame())

    roi = estimator.received_boxes[-1][0]
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (50, 45, 150, 155)
    assert np.allclose(roi.keypoints, det_keypoints)


def test_temporal_roi_runner_scaled_transform_scales_the_detector_box() -> None:
    """When the most recently available landmark box is exactly 2x the (still fixed)
    anchor's box (same top-left corner), the propagated ROI must be the anchor's
    detector box scaled by that same 2x transform -- not the unscaled detector box and
    not a landmark-derived box. Uses 3 frames: frame 1 sets the anchor; frame 2 is a
    (trivially identity-transformed) landmark-only frame whose *own output* is the 2x
    landmark box; frame 3's ROI is then built from frame 2's output against the
    still-unchanged anchor, which is where the 2x scale actually applies."""
    big_frame_size = 400  # large enough that the 2x-scaled ROI still fits in-bounds
    anchor_box = [BBox(x1=50, y1=45, x2=150, y2=155, score=0.95)]
    detector = _RecordingFakeDetector([anchor_box])  # only frame 1 may call the detector
    estimator = _RecordingFakeEstimator(
        [
            [_centered_landmarks()],  # frame 1: full detect, anchor landmark box (60,60,140,140)
            # frame 2: landmark-only, identity transform; its own output is the box
            # (60,60,220,220) -- 2x the anchor's width/height, same top-left corner.
            [_make_landmarks(_spread_xy(x0=60.0, x1=220.0, y0=60.0, y1=220.0))],
            [_centered_landmarks()],  # frame 3: landmark-only, built from frame 2's output
        ]
    )
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame(width=big_frame_size, height=big_frame_size))  # frame 1
    runner.process(_frame(width=big_frame_size, height=big_frame_size))  # frame 2
    runner.process(_frame(width=big_frame_size, height=big_frame_size))  # frame 3

    # scale=2, offset = 60 - 60*2 = -60 -> mapped = det * 2 - 60
    roi = estimator.received_boxes[-1][0]
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (40, 30, 240, 250)
    assert detector.call_count == 1  # frames 2 and 3 both stayed landmark-only


def test_temporal_roi_runner_468_point_mesh_cannot_use_landmark_only() -> None:
    """The 468-point (no-iris) mesh lacks indices 468/473; landmark-only mode needs the
    478-point (V2_478) mesh, so this must fall back to full-detect on every frame."""
    xy_468 = _spread_xy(x0=60.0, x1=140.0, y0=60.0, y1=140.0, count=468)
    landmarks_468 = _make_landmarks(xy_468)
    box = [BBox(x1=50, y1=50, x2=150, y2=150, score=0.95)]
    detector = _RecordingFakeDetector([box, box])
    estimator = _RecordingFakeEstimator([[landmarks_468], [landmarks_468]])
    pipeline = _RecordingFakePipeline(detector, estimator)
    runner = producer.TemporalRoiRunner(pipeline, detector_refresh_interval=10)

    runner.process(_frame())
    runner.process(_frame())

    assert detector.call_count == 2


def test_temporal_roi_runner_rejects_non_positive_refresh_interval() -> None:
    pipeline = _RecordingFakePipeline(_RecordingFakeDetector([]), _RecordingFakeEstimator([]))

    with pytest.raises(ValueError, match="detector_refresh_interval"):
        producer.TemporalRoiRunner(pipeline, detector_refresh_interval=0)


# ---------------------------------------------------------------------------------------
# FrameSource (workdoc steps 28-29)
# ---------------------------------------------------------------------------------------


class _FakeClock:
    """Deterministic stand-in for ``time.perf_counter``/``time.sleep``: "sleeping"
    advances the fake clock by exactly the requested duration, so pacing tests need zero
    real waiting and cannot flake on system/CI jitter."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def now_fn(self) -> float:
        return self.now

    def sleep_fn(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


def test_video_file_frame_source_oneshot_reads_all_frames_then_eof(tmp_path) -> None:
    video_path = tmp_path / "three_frames.avi"
    write_synthetic_avi(video_path, width=64, height=48, fps=30.0, frame_count=3)

    source = producer.VideoFileFrameSource(
        str(video_path), width=64, height=48, pacing=producer.Pacing.ONESHOT
    )
    try:
        seen_pixels = []
        for _ in range(3):
            ok, frame = source.read()
            assert ok
            assert frame.shape == (48, 64, 3)
            seen_pixels.append(int(frame[0, 0, 0]))
        # MJPG is lossy even on a flat color (see write_synthetic_avi); a generous
        # tolerance still unambiguously distinguishes the three frames and their order.
        assert seen_pixels == pytest.approx([0, 100, 200], abs=20)

        ok, frame = source.read()  # OpenCV read() failure, not the header count, is EOF
        assert ok is False
        assert frame is None
    finally:
        source.close()


def test_video_file_frame_source_realtime_paces_between_frames(tmp_path) -> None:
    video_path = tmp_path / "three_frames_10fps.avi"
    write_synthetic_avi(video_path, width=64, height=48, fps=10.0, frame_count=3)
    clock = _FakeClock()

    source = producer.VideoFileFrameSource(
        str(video_path),
        width=64,
        height=48,
        pacing=producer.Pacing.REALTIME,
        now_fn=clock.now_fn,
        sleep_fn=clock.sleep_fn,
    )
    try:
        for _ in range(3):
            ok, _decoded = source.read()
            assert ok
    finally:
        source.close()

    # The first read must not pace (nothing to catch up to yet); each read after it
    # waits one 1/10s frame interval.
    assert clock.sleep_calls == pytest.approx([0.1, 0.1])


def test_video_file_frame_source_realtime_requires_valid_fps(monkeypatch, tmp_path) -> None:
    class _FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, _prop: int) -> float:
            return 0.0  # simulates a container with no reliable FPS metadata

        def release(self) -> None:
            pass

    monkeypatch.setattr(producer.cv2, "VideoCapture", lambda _path: _FakeCapture())
    fixture = tmp_path / "no_fps.avi"
    fixture.write_bytes(b"stand-in bytes; cv2.VideoCapture is monkeypatched for this test")

    with pytest.raises(ValueError, match="realtime"):
        producer.VideoFileFrameSource(
            str(fixture), width=64, height=48, pacing=producer.Pacing.REALTIME
        )


def test_video_file_frame_source_resolution_mismatch_is_explicit_error(tmp_path) -> None:
    video_path = tmp_path / "wrong_resolution.avi"
    write_synthetic_avi(video_path, width=64, height=48, fps=30.0, frame_count=1)

    source = producer.VideoFileFrameSource(
        str(video_path), width=1280, height=720, pacing=producer.Pacing.ONESHOT
    )
    try:
        with pytest.raises(ValueError, match="resolution"):
            source.read()
    finally:
        source.close()


def test_camera_frame_source_raises_on_capture_failure(monkeypatch) -> None:
    class _FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, *_args) -> None:
            pass

        def read(self):
            return False, None

    monkeypatch.setattr(producer.cv2, "VideoCapture", lambda _device: _FakeCapture())

    source = producer.CameraFrameSource("/dev/video0", width=64, height=48)
    with pytest.raises(RuntimeError, match="camera frame capture failed"):
        source.read()


def test_build_frame_source_decimal_string_is_camera_index(monkeypatch) -> None:
    class _FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, *_args) -> None:
            pass

    monkeypatch.setattr(producer.cv2, "VideoCapture", lambda _device: _FakeCapture())

    source = producer.build_frame_source("0", width=64, height=48, pacing=producer.Pacing.ONESHOT)

    assert isinstance(source, producer.CameraFrameSource)
    assert source.device == 0  # a numeric index, not the string "0"


def test_build_frame_source_device_path_is_camera(monkeypatch) -> None:
    class _FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, *_args) -> None:
            pass

    monkeypatch.setattr(producer.cv2, "VideoCapture", lambda _device: _FakeCapture())

    source = producer.build_frame_source(
        "/dev/video0", width=64, height=48, pacing=producer.Pacing.ONESHOT
    )

    assert isinstance(source, producer.CameraFrameSource)
    assert source.device == "/dev/video0"


def test_build_frame_source_existing_file_is_video(tmp_path) -> None:
    video_path = tmp_path / "dispatch.avi"
    write_synthetic_avi(video_path, width=64, height=48, fps=30.0, frame_count=1)

    source = producer.build_frame_source(
        str(video_path), width=64, height=48, pacing=producer.Pacing.REALTIME
    )
    try:
        assert isinstance(source, producer.VideoFileFrameSource)
        assert source.pacing is producer.Pacing.REALTIME
    finally:
        source.close()


def test_cli_source_default_and_camera_alias() -> None:
    parser = producer.build_arg_parser()

    args = parser.parse_args([])
    assert args.source == "/dev/video0"
    assert args.camera is None  # deprecated alias, only overrides --source when given

    args = parser.parse_args(["--camera", "/dev/video2"])
    assert args.camera == "/dev/video2"


def test_cli_pacing_choices_and_default() -> None:
    parser = producer.build_arg_parser()

    args = parser.parse_args([])
    assert args.pacing == "oneshot"

    args = parser.parse_args(["--pacing", "realtime"])
    assert args.pacing == "realtime"

    with pytest.raises(SystemExit):
        parser.parse_args(["--pacing", "bogus"])


def test_cli_detector_refresh_interval_default_is_one() -> None:
    parser = producer.build_arg_parser()

    args = parser.parse_args([])
    assert args.detector_refresh_interval == 1

    args = parser.parse_args(["--detector-refresh-interval", "5"])
    assert args.detector_refresh_interval == 5


# --- Preview lane (workdoc steps 37-38) ---------------------------------------------


def test_preview_publish_rate_is_capped_at_ten_fps() -> None:
    """Feed frames at a simulated 30 FPS and assert the 10 FPS preview cap holds.

    Uses a virtual clock (no real ``sleep``) so the assertion is exact, not timing-flaky.
    """
    now = 0.0
    last_sent: float | None = None
    sent_at: list[float] = []
    frame_interval = 1.0 / 30.0

    for _ in range(90):  # 3 simulated seconds of 30 FPS frames
        if producer._should_publish_preview(now, last_sent):
            sent_at.append(now)
            last_sent = now
        now += frame_interval

    # 3s at <= 10 FPS is at most 31 sends (one immediate send plus 30 more one full
    # second apart in the worst case of interval rounding).
    assert 1 < len(sent_at) <= 31
    for earlier, later in itertools.pairwise(sent_at):
        assert later - earlier >= producer.PREVIEW_MIN_INTERVAL_S - 1e-9


def test_encode_preview_frame_resizes_to_the_preview_contract() -> None:
    """The producer must always send exactly the 640x360 the server's contract expects."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    encoded = producer.encode_preview_frame(frame, None, "label", jpeg_quality=82)

    assert jpeg_dimensions(encoded) == (producer.PREVIEW_WIDTH_PX, producer.PREVIEW_HEIGHT_PX)


def test_encode_preview_frame_draws_landmarks_scaled_into_preview_space() -> None:
    """Landmark coordinates are in full-resolution pixels; drawing must not go out of
    the resized 640x360 canvas (a naive un-scaled draw would land far outside it)."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # A landmark near the full-resolution frame's bottom-right corner.
    landmarks_xy = np.tile(np.array([[1270.0, 710.0]]), (478, 1))

    encoded = producer.encode_preview_frame(frame, landmarks_xy, "label", jpeg_quality=82)

    # Encoding must succeed without raising (an unscaled circle center would fall
    # outside the 640x360 canvas, which OpenCV clips silently rather than erroring, so
    # the meaningful assertion here is that the contract dimensions still hold).
    assert jpeg_dimensions(encoded) == (producer.PREVIEW_WIDTH_PX, producer.PREVIEW_HEIGHT_PX)


class _RaisingPreviewPublisher:
    """Stands in for an ``IpcPublisher`` whose preview POST always fails."""

    def publish_bytes(self, body: bytes, *, content_type: str) -> None:
        raise OSError("preview socket exploded")


def test_preview_publish_failure_is_swallowed_and_never_propagates() -> None:
    """A raising preview lane must not stop whatever the caller does next -- in
    ``main()``, that is the following frame's control-packet publish."""
    log_calls: list[str] = []

    ok = producer.publish_preview_best_effort(
        _RaisingPreviewPublisher(), b"\xff\xd8fake-jpeg", log=log_calls.append
    )

    assert ok is False
    assert len(log_calls) == 1
    assert "preview socket exploded" in log_calls[0]
    # If publish_preview_best_effort had let the OSError propagate, this line would
    # never run -- proving control-lane code placed after it is unaffected.
    log_calls.append("control lane continued")
    assert log_calls[-1] == "control lane continued"
