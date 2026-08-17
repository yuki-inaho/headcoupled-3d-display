"""Pick a detector-refresh interval from measured accuracy and latency, or refuse to.

Reads the raw sweep produced by ``scripts/sweep_detector_refresh.py`` in the Python 3.10
CUDA environment and evaluates every candidate against the workdoc's thresholds. The
candidate with the lowest p95 recognition latency *among those that pass every accuracy
threshold* is selected. If no candidate passes, this says so and selects nothing rather
than adopting the least-bad one.

Accuracy is measured against the full-detect reference (``interval == 1``) on the same
frames, in the metric display frame, using the same ``HeadPoseEstimator`` the product
uses -- not against the landmark pixels, which would hide errors that only matter after
the pose solve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from headcoupled_display.models import HardwareProfile
from headcoupled_display.profiles import (
    load_user_profile,
    profile_with_resolved_matrix,
)
from headcoupled_display.tracking import HeadPoseEstimator

REFERENCE_INTERVAL = 1
LANDMARK_ARRAY_LENGTH = 478

MAX_EYE_POSITION_P95_MM = 5.0
MAX_ANGLE_P95_DEG = 1.0
MAX_RECOGNITION_MEDIAN_MS = 16.7
MAX_RECOGNITION_P95_MS = 33.3
MAX_MISSING_FRAMES = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="sweep_detector_refresh output")
    parser.add_argument("--profile", type=Path, required=True, help="hardware profile JSON")
    parser.add_argument("--user-profile", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, help="tagcal calibration.json for real K, D")
    parser.add_argument("--face-model", type=Path, help="personal 478-point shape.pcd")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _percentiles(samples: list[float]) -> dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, [50, 95, 99], method="linear")
    return {"p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99)}


def _dense_landmarks(points: list[list[float]], indices: list[int]) -> np.ndarray | None:
    """Rebuild the sparse (478, 2) array the estimator indexes into.

    Rows outside ``indices`` are never read by the estimator; filling them with NaN makes
    an accidental read fail loudly instead of silently using the origin.
    """

    if not points:
        return None
    dense = np.full((LANDMARK_ARRAY_LENGTH, 2), np.nan, dtype=np.float64)
    for index, point in zip(indices, points, strict=True):
        dense[index] = point
    return dense


def _estimator(args: argparse.Namespace) -> HeadPoseEstimator:
    hardware = profile_with_resolved_matrix(HardwareProfile.load(args.profile))
    if args.intrinsics is not None:
        from headcoupled_display.profiles import load_tagcal_calibration

        hardware = hardware.model_copy(update={"camera": load_tagcal_calibration(args.intrinsics)})
    user = load_user_profile(args.user_profile)
    if args.face_model is not None:
        user = user.model_copy(update={"face_model_path": str(args.face_model.resolve())})
    return HeadPoseEstimator(hardware, user)


def _poses(
    estimator: HeadPoseEstimator, candidate: dict[str, Any], indices: list[int]
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    poses: list[tuple[np.ndarray, np.ndarray] | None] = []
    for points in candidate["landmarks_px"]:
        dense = _dense_landmarks(points, indices)
        if dense is None:
            poses.append(None)
            continue
        try:
            _left, _right, eye, forward = estimator.estimate(dense)
        except RuntimeError:
            poses.append(None)
            continue
        poses.append((np.asarray(eye, dtype=np.float64), np.asarray(forward, dtype=np.float64)))
    return poses


def _accuracy(
    reference: list[tuple[np.ndarray, np.ndarray] | None],
    candidate: list[tuple[np.ndarray, np.ndarray] | None],
) -> dict[str, Any]:
    eye_errors_mm: list[float] = []
    angle_errors_deg: list[float] = []
    comparable = 0
    for ref, cand in zip(reference, candidate, strict=True):
        if ref is None or cand is None:
            continue
        comparable += 1
        eye_errors_mm.append(float(np.linalg.norm(cand[0] - ref[0]) * 1000.0))
        cosine = float(np.clip(np.dot(cand[1], ref[1]), -1.0, 1.0))
        angle_errors_deg.append(float(np.degrees(np.arccos(cosine))))
    if not comparable:
        return {"comparable_frames": 0, "eye_position_p95_mm": None, "angle_p95_deg": None}
    return {
        "comparable_frames": comparable,
        "eye_position_p95_mm": float(np.percentile(eye_errors_mm, 95)),
        "eye_position_max_mm": float(np.max(eye_errors_mm)),
        "angle_p95_deg": float(np.percentile(angle_errors_deg, 95)),
        "angle_max_deg": float(np.max(angle_errors_deg)),
    }


def _judge(candidate: dict[str, Any]) -> dict[str, Any]:
    accuracy = candidate["accuracy"]
    latency = candidate["recognition"]
    reasons: list[str] = []
    if candidate["missing_face_frame_count"] > MAX_MISSING_FRAMES:
        reasons.append(f"missing {candidate['missing_face_frame_count']} frames")
    eye_p95 = accuracy.get("eye_position_p95_mm")
    angle_p95 = accuracy.get("angle_p95_deg")
    if eye_p95 is None or angle_p95 is None:
        reasons.append("no comparable frames")
    else:
        if eye_p95 > MAX_EYE_POSITION_P95_MM:
            reasons.append(f"eye p95 {eye_p95:.3f} mm > {MAX_EYE_POSITION_P95_MM}")
        if angle_p95 > MAX_ANGLE_P95_DEG:
            reasons.append(f"angle p95 {angle_p95:.3f} deg > {MAX_ANGLE_P95_DEG}")
    accuracy_ok = not reasons
    latency_reasons: list[str] = []
    if latency["p50_ms"] > MAX_RECOGNITION_MEDIAN_MS:
        latency_reasons.append(f"median {latency['p50_ms']:.3f} ms > {MAX_RECOGNITION_MEDIAN_MS}")
    if latency["p95_ms"] > MAX_RECOGNITION_P95_MS:
        latency_reasons.append(f"p95 {latency['p95_ms']:.3f} ms > {MAX_RECOGNITION_P95_MS}")
    return {
        "accuracy_ok": accuracy_ok,
        "accuracy_reasons": reasons,
        "latency_ok": not latency_reasons,
        "latency_reasons": latency_reasons,
    }


def _evaluate_candidates(raw: dict[str, Any], estimator: HeadPoseEstimator) -> list[dict[str, Any]]:
    indices = raw["exported_landmark_indices"]
    by_interval = {int(entry["interval"]): entry for entry in raw["candidates"]}
    if REFERENCE_INTERVAL not in by_interval:
        raise SystemExit(
            f"the sweep has no interval={REFERENCE_INTERVAL} full-detect reference to compare to"
        )
    reference_poses = _poses(estimator, by_interval[REFERENCE_INTERVAL], indices)

    results: list[dict[str, Any]] = []
    for interval in sorted(by_interval):
        entry = by_interval[interval]
        poses = (
            reference_poses if interval == REFERENCE_INTERVAL else _poses(estimator, entry, indices)
        )
        result = {
            "interval": interval,
            "missing_face_frame_count": entry["missing_face_frame_count"],
            "recognition": _percentiles(entry["recognition_ms"]),
            "accuracy": _accuracy(reference_poses, poses),
        }
        result["verdict"] = _judge(result)
        results.append(result)
    return results


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    results = _evaluate_candidates(raw, _estimator(args))

    eligible = [
        entry
        for entry in results
        if entry["verdict"]["accuracy_ok"] and entry["verdict"]["latency_ok"]
    ]
    selected = min(eligible, key=lambda entry: entry["recognition"]["p95_ms"]) if eligible else None
    return {
        "schema_version": 1,
        "source": raw["source"],
        "frame_count": raw["frame_count"],
        "warmup": raw["warmup"],
        "provider": raw["provider"],
        "reference_interval": REFERENCE_INTERVAL,
        "thresholds": {
            "eye_position_p95_mm": MAX_EYE_POSITION_P95_MM,
            "angle_p95_deg": MAX_ANGLE_P95_DEG,
            "recognition_median_ms": MAX_RECOGNITION_MEDIAN_MS,
            "recognition_p95_ms": MAX_RECOGNITION_P95_MS,
            "missing_frames": MAX_MISSING_FRAMES,
        },
        "candidates": results,
        "selected_interval": None if selected is None else selected["interval"],
        "selection_note": (
            "no candidate met every accuracy and latency threshold; nothing was selected"
            if selected is None
            else "lowest p95 recognition latency among candidates meeting every threshold"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for entry in report["candidates"]:
        accuracy = entry["accuracy"]
        eye = accuracy.get("eye_position_p95_mm")
        angle = accuracy.get("angle_p95_deg")
        print(
            f"interval={entry['interval']:>2}  "
            f"p50={entry['recognition']['p50_ms']:7.3f}  "
            f"p95={entry['recognition']['p95_ms']:7.3f}  "
            f"eye_p95={'n/a' if eye is None else f'{eye:6.3f} mm'}  "
            f"angle_p95={'n/a' if angle is None else f'{angle:5.3f} deg'}  "
            f"missing={entry['missing_face_frame_count']}  "
            f"accuracy_ok={entry['verdict']['accuracy_ok']}  "
            f"latency_ok={entry['verdict']['latency_ok']}"
        )
    print(f"selected_interval: {report['selected_interval']} ({report['selection_note']})")
    return 0 if report["selected_interval"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
