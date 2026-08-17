"""Sweep detector-refresh intervals over a recording and record accuracy + latency.

Runs in the FaceMesh project's Python 3.10 / CUDA environment, so it depends only on the
standard library, numpy, cv2 and facemesh_tracking. Neither pydantic nor beartype is
installed there; the schema-checked aggregation is a separate 3.13 step
(``scripts/analyze_refresh_sweep.py``).

The interval is not guessed. Every candidate is run over the whole recording, the
landmarks needed to reconstruct the metric pose are written out, and the 3.13 step picks
the fastest candidate that still meets the accuracy thresholds -- or reports that none
does.

``--interval 1`` refreshes the detector on every frame and is therefore the full-detect
reference the other candidates are compared against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facemesh_ipc_producer import (
    TemporalRoiRunner,
    _actual_providers,
)

#: The 12 PnP landmarks plus both iris centres. Only these are needed to rebuild the
#: metric pose downstream, so the sweep does not carry 478 points per frame per candidate.
PNP_INDICES = (1, 6, 33, 133, 362, 263, 61, 291, 199, 168, 94, 4)
LEFT_IRIS_CENTRE = 468
RIGHT_IRIS_CENTRE = 473
EXPORTED_INDICES = (*PNP_INDICES, LEFT_IRIS_CENTRE, RIGHT_IRIS_CENTRE)

DEFAULT_INTERVALS = (1, 2, 3, 5, 8, 10)
DEFAULT_WARMUP = 5


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=list(DEFAULT_INTERVALS),
        help="detector_refresh_interval candidates; 1 is the full-detect reference",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--device-id", type=int, default=0)
    return parser


def _decode_all_frames(video_path: Path) -> list[np.ndarray]:
    """Decode the whole recording once so every candidate sees identical input.

    The AVI header is not consulted: this file reports 602 frames at 60 fps while only
    294 decode. ``read()`` returning False is the only end-of-file signal used.
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"{video_path} decoded zero frames")
    return frames


def _build_pipeline(device_id: int) -> Any:
    from facemesh_tracking.pipeline import FaceMeshPipeline
    from facemesh_tracking.runtime import Backend

    return FaceMeshPipeline.create(backend=Backend("cuda"), device_id=device_id)


def _exported_points(face: Any) -> list[list[float]]:
    points = np.asarray(face.points, dtype=np.float64)
    return [[float(points[index, 0]), float(points[index, 1])] for index in EXPORTED_INDICES]


def run_candidate(
    pipeline: Any, frames: list[np.ndarray], interval: int, warmup: int
) -> dict[str, Any]:
    runner = TemporalRoiRunner(pipeline, detector_refresh_interval=interval)
    durations_ms: list[float] = []
    landmarks: list[list[list[float]]] = []
    scores: list[float] = []
    missing = 0

    for index, frame in enumerate(frames):
        started = time.perf_counter_ns()
        result = runner.process(frame)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if index >= warmup:
            durations_ms.append(elapsed_ms)
        if result.faces:
            face = max(result.faces, key=lambda value: float(value.score))
            landmarks.append(_exported_points(face))
            scores.append(float(face.score))
        else:
            missing += 1
            landmarks.append([])
            scores.append(0.0)

    return {
        "interval": interval,
        "recognition_ms": durations_ms,
        "landmarks_px": landmarks,
        "scores": scores,
        "missing_face_frame_count": missing,
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    frames = _decode_all_frames(args.video)
    if len(frames) <= args.warmup:
        raise RuntimeError(f"decoded only {len(frames)} frames but --warmup={args.warmup}")
    pipeline = _build_pipeline(args.device_id)
    providers = {
        label: _actual_providers(component, label)
        for label, component in (("detector", pipeline.detector), ("estimator", pipeline.estimator))
    }
    for label, names in providers.items():
        if not names or names[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{label} is not running on CUDA (actual={names!r})")

    height, width = frames[0].shape[:2]
    candidates = [run_candidate(pipeline, frames, interval, args.warmup) for interval in args.intervals]
    return {
        "schema_version": 1,
        "source": str(args.video.resolve()),
        "frame_count": len(frames),
        "warmup": args.warmup,
        "resolution": {"width_px": width, "height_px": height},
        "provider": "CUDAExecutionProvider",
        "providers_by_stage": providers,
        "clock_domain": "monotonic_ns",
        "exported_landmark_indices": list(EXPORTED_INDICES),
        "pnp_indices": list(PNP_INDICES),
        "iris_indices": [LEFT_IRIS_CENTRE, RIGHT_IRIS_CENTRE],
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for candidate in result["candidates"]:
        samples = sorted(candidate["recognition_ms"])
        p50 = samples[len(samples) // 2]
        p95 = samples[min(len(samples) - 1, int(0.95 * (len(samples) - 1)))]
        print(
            f"  interval={candidate['interval']:>2}  "
            f"p50={p50:7.3f} ms  p95={p95:7.3f} ms  "
            f"missing={candidate['missing_face_frame_count']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
