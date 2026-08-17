from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

import cv2
import numpy as np
from numpy.typing import NDArray

from tagcal.board import AprilGridBoard
from tagcal.cvtypes import as_uint8
from tagcal.detection import AprilTagDetector, DetectionObservation
from tagcal.models import KeyframeRecord, SelectionReport, SelectionSpec

ProgressCallback = Callable[[str], None]
_DESCRIPTOR_WEIGHTS = np.asarray([1.25, 1.25, 1.00, 0.90, 0.90, 0.45, 0.50])


def local_sharpness_maxima(
    frame_indices: Sequence[int],
    sharpness: Sequence[float],
    *,
    window_frames: int,
    relative_floor: float,
) -> list[int]:
    """Positions of frames that are the sharpest within `window_frames` of themselves.

    An operator sweeping a board through poses produces the same pattern every
    time: the image is sharp where they paused and blurred while they moved. A
    fixed sharpness threshold cannot separate the two, because the measure's scale
    depends on the scene, the exposure and how large the board appears -- all of
    which change during the sweep. Over a fraction of a second those factors are
    effectively constant, so the neighbourhood maximum is the frame the operator
    actually meant to capture.

    `relative_floor` additionally compares each winner against the median of the
    whole recording, so a stretch where every frame is blurred does not contribute
    its least-blurred frame. It is deliberately loose: raising it turns blur
    rejection into pose selection, which collapses the pose diversity calibration
    depends on.

    Inputs must be ordered by frame index.
    """
    if window_frames < 1:
        raise ValueError("window_frames must be at least 1")
    if len(frame_indices) != len(sharpness):
        raise ValueError("frame_indices and sharpness must have the same length")
    if not frame_indices:
        return []

    values = np.asarray(sharpness, dtype=np.float64)
    floor = float(np.median(values)) * relative_floor
    kept: list[int] = []
    for position, frame_index in enumerate(frame_indices):
        start, stop = position, position
        while start > 0 and frame_index - frame_indices[start - 1] <= window_frames:
            start -= 1
        while stop + 1 < len(frame_indices) and frame_indices[stop + 1] - frame_index <= window_frames:
            stop += 1
        if values[position] >= float(np.max(values[start : stop + 1])) and values[position] >= floor:
            kept.append(position)
    return kept


@dataclass(slots=True)
class _Candidate:
    frame_index: int
    timestamp_seconds: float
    observation: DetectionObservation
    jpeg: bytes

    @property
    def descriptor(self) -> NDArray[np.float64]:
        return np.asarray(self.observation.metrics.descriptor, dtype=np.float64)

    @property
    def quality(self) -> float:
        return self.observation.metrics.quality_score


class VideoKeyframeSelector:
    """Sample a recorded video, reject weak frames, then greedily maximize pose diversity."""

    def __init__(self, board: AprilGridBoard, spec: SelectionSpec) -> None:
        self._board = board
        self._spec = spec
        self._detector = AprilTagDetector(board)

    def select(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> SelectionReport:
        notify = progress or (lambda _: None)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0.0 or source_fps > 1000.0:
            source_fps = 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sample_step = max(1, round(source_fps / self._spec.sample_fps))

        candidates: list[_Candidate] = []
        rejected = {
            "not_detected": 0,
            "too_few_tags": 0,
            "low_coverage": 0,
            "blurred": 0,
            "near_duplicate": 0,
            "jpeg_failure": 0,
            "reservoir_replaced": 0,
        }
        sampled_frames = 0
        valid_seen = 0
        frame_index = 0
        previous_descriptor: NDArray[np.float64] | None = None
        previous_quality = 0.0
        rng = Random(0)
        required_tags = min(self._spec.min_detected_tags, self._board.marker_count)

        notify(
            f"Analyzing {video_path.name}: {width}x{height}, {source_fps:.2f} fps, "
            f"sampling every {sample_step} frame(s)."
        )
        try:
            while True:
                ok, decoded = capture.read()
                if not ok or decoded is None:
                    break
                if frame_index % sample_step != 0:
                    frame_index += 1
                    continue

                sampled_frames += 1
                frame = as_uint8(decoded)
                observation = self._detector.detect(frame)
                if observation is None:
                    rejected["not_detected"] += 1
                    frame_index += 1
                    continue

                metrics = observation.metrics
                if metrics.detected_tags < required_tags:
                    rejected["too_few_tags"] += 1
                    frame_index += 1
                    continue
                if metrics.board_coverage < self._spec.min_board_coverage:
                    rejected["low_coverage"] += 1
                    frame_index += 1
                    continue
                if metrics.sharpness < self._spec.min_sharpness:
                    rejected["blurred"] += 1
                    frame_index += 1
                    continue

                descriptor = np.asarray(metrics.descriptor, dtype=np.float64)
                if previous_descriptor is not None:
                    distance = self._descriptor_distance(descriptor, previous_descriptor)
                    if (
                        distance < self._spec.min_candidate_distance
                        and metrics.quality_score <= previous_quality * 1.08
                    ):
                        rejected["near_duplicate"] += 1
                        frame_index += 1
                        continue

                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._spec.jpeg_quality],
                )
                if not encoded_ok:
                    rejected["jpeg_failure"] += 1
                    frame_index += 1
                    continue

                candidate = _Candidate(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / source_fps,
                    observation=observation,
                    jpeg=encoded.tobytes(),
                )
                valid_seen += 1
                if len(candidates) < self._spec.max_candidates:
                    candidates.append(candidate)
                else:
                    replacement_index = rng.randrange(valid_seen)
                    if replacement_index < self._spec.max_candidates:
                        candidates[replacement_index] = candidate
                        rejected["reservoir_replaced"] += 1

                previous_descriptor = descriptor
                previous_quality = metrics.quality_score
                if sampled_frames % 50 == 0:
                    notify(
                        f"Sampled {sampled_frames} frames; "
                        f"retained {len(candidates)} valid candidates."
                    )
                frame_index += 1
        finally:
            capture.release()

        if not candidates:
            raise RuntimeError(
                "No usable frames were found. Increase board visibility, lighting, focus, or "
                "relax selection thresholds."
            )

        before_blur = len(candidates)
        candidates = self._suppress_blurred(candidates, source_fps)
        rejected["blurred_vs_neighbours"] = before_blur - len(candidates)
        notify(
            f"Sharpness suppression kept {len(candidates)} of {before_blur} candidates "
            f"(local maxima within {self._spec.blur_window_seconds:.2f}s)."
        )

        selected_candidates = self._greedy_select(candidates, self._spec.target_frames)
        if len(selected_candidates) < self._spec.min_keyframes:
            raise RuntimeError(
                f"Only {len(selected_candidates)} usable keyframes were selected; "
                f"at least {self._spec.min_keyframes} are required."
            )

        keyframe_dir = output_dir / "keyframes"
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        for stale_image in keyframe_dir.glob("keyframe_*.jpg"):
            stale_image.unlink()
        records: list[KeyframeRecord] = []
        for order, candidate in enumerate(sorted(selected_candidates, key=lambda item: item.frame_index)):
            filename = f"keyframe_{order:03d}_frame_{candidate.frame_index:08d}.jpg"
            image_path = keyframe_dir / filename
            image_path.write_bytes(candidate.jpeg)
            observation = candidate.observation
            records.append(
                KeyframeRecord(
                    image_path=str(Path("keyframes") / filename),
                    frame_index=candidate.frame_index,
                    timestamp_seconds=candidate.timestamp_seconds,
                    marker_ids=[int(value) for value in observation.marker_ids.reshape(-1)],
                    object_points=observation.object_points.astype(float).tolist(),
                    image_points=observation.image_points.astype(float).tolist(),
                    metrics=observation.metrics,
                )
            )

        report = SelectionReport(
            video_path=str(video_path),
            image_width=width,
            image_height=height,
            source_fps=source_fps,
            total_frames=total_frames if total_frames > 0 else frame_index,
            sampled_frames=sampled_frames,
            valid_candidates=min(valid_seen, self._spec.max_candidates),
            selected=records,
            rejected_summary=rejected,
        )
        report_path = output_dir / "selection_report.json"
        report.save(report_path)
        notify(
            f"Selected {len(records)} keyframes from {sampled_frames} sampled frames. "
            f"Report: {report_path}"
        )
        return report

    def _suppress_blurred(
        self,
        candidates: list[_Candidate],
        source_fps: float,
    ) -> list[_Candidate]:
        if len(candidates) < 3:
            return candidates

        ordered = sorted(candidates, key=lambda item: item.frame_index)
        kept_positions = local_sharpness_maxima(
            [item.frame_index for item in ordered],
            [item.observation.metrics.sharpness for item in ordered],
            window_frames=max(1, round(self._spec.blur_window_seconds * source_fps)),
            relative_floor=self._spec.blur_relative_floor,
        )
        if len(kept_positions) >= self._spec.min_keyframes:
            return [ordered[position] for position in kept_positions]

        # The recording may simply be short; fall back to the sharpest candidates.
        ranked = sorted(candidates, key=lambda item: -item.observation.metrics.sharpness)
        return ranked[: max(self._spec.min_keyframes, len(kept_positions))]

    def _greedy_select(self, candidates: list[_Candidate], limit: int) -> list[_Candidate]:
        if len(candidates) <= limit:
            return list(candidates)

        first_index = max(range(len(candidates)), key=lambda index: candidates[index].quality)
        selected_indices = [first_index]
        remaining = set(range(len(candidates)))
        remaining.remove(first_index)
        occupied_cells = {self._spatial_cell(candidates[first_index])}

        while remaining and len(selected_indices) < limit:
            best_index = -1
            best_score = float("-inf")
            for candidate_index in remaining:
                candidate = candidates[candidate_index]
                minimum_distance = min(
                    self._descriptor_distance(
                        candidate.descriptor,
                        candidates[selected_index].descriptor,
                    )
                    for selected_index in selected_indices
                )
                diversity_score = min(1.0, minimum_distance / 0.85)
                cell_bonus = 1.0 if self._spatial_cell(candidate) not in occupied_cells else 0.0
                score = 0.56 * diversity_score + 0.34 * candidate.quality + 0.10 * cell_bonus
                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            selected_indices.append(best_index)
            remaining.remove(best_index)
            occupied_cells.add(self._spatial_cell(candidates[best_index]))

        return [candidates[index] for index in selected_indices]

    @staticmethod
    def _spatial_cell(candidate: _Candidate) -> tuple[int, int]:
        metrics = candidate.observation.metrics
        x = min(2, max(0, int(metrics.center_x * 3.0)))
        y = min(2, max(0, int(metrics.center_y * 3.0)))
        return x, y

    @staticmethod
    def _descriptor_distance(
        left: NDArray[np.float64],
        right: NDArray[np.float64],
    ) -> float:
        if left.shape != _DESCRIPTOR_WEIGHTS.shape or right.shape != _DESCRIPTOR_WEIGHTS.shape:
            raise ValueError("Unexpected pose descriptor shape")
        return float(np.linalg.norm((left - right) * _DESCRIPTOR_WEIGHTS))
