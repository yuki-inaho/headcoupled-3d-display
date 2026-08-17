"""Machine-judge the workdoc's latency and accuracy thresholds against measured reports.

Every threshold in the workdoc is encoded here as a named constant so that "did we meet
it?" is answered by running this script, not by reading a table and agreeing with it.

A check whose inputs are absent reports ``not_measured`` and makes the overall verdict
fail. That is deliberate: the failure mode this guards against is a run that looks green
because a stage was never measured at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Thresholds, quoted from the workdoc's 成功条件 -------------------------------------

#: 成功条件 5: the recorded acceptance set must stay complete.
REQUIRED_FRAME_COUNT = 294
MAX_MISSING_FRAMES = 0
MAX_EYE_POSITION_P95_MM = 5.0
MAX_ANGLE_P95_DEG = 1.0

#: 成功条件 6: recognition, measured on the recording.
MAX_RECOGNITION_MEDIAN_MS = 16.7
MAX_RECOGNITION_P95_MS = 33.3

#: 成功条件 7: adopted control transport under overload.
MAX_CONTROL_P95_MS = 2.0
MAX_CATCH_UP_FRAMES = 2

#: 成功条件 8: preview lane.
PREVIEW_WIDTH_PX = 640
PREVIEW_HEIGHT_PX = 360
MAX_PREVIEW_FPS = 10.0
MAX_SERVER_REENCODES = 0

#: 成功条件 9: browser side.
MAX_RECEIVE_TO_DRAW_P95_MS = 16.7
MAX_CPU_DRAW_P95_MS = 4.0

#: 成功条件 10: recognition completion to WebGL, excluding camera exposure/capture.
MAX_END_TO_END_MEDIAN_MS = 33.0
MAX_END_TO_END_P95_MS = 60.0

#: Runs whose two clock domains disagree by more than this are refused outright.
MAX_CLOCK_UNCERTAINTY_MS = 2.0

#: Stages whose sum is "recognition" for 成功条件 6.
RECOGNITION_STAGES = ("detector", "landmarks")


@dataclass
class Check:
    """One threshold decision, carrying the numbers that produced it."""

    condition: str
    name: str
    status: str  # "pass" | "fail" | "not_measured"
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "measured": self.measured,
        }


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _not_measured(condition: str, name: str, what: str) -> Check:
    return Check(
        condition=condition,
        name=name,
        status="not_measured",
        detail=f"{what} was not supplied; an unmeasured threshold is not a passing one",
    )


def _verdict(ok: bool) -> str:
    return "pass" if ok else "fail"


def check_clock_trust(report: dict[str, Any] | None) -> Check:
    if report is None:
        return _not_measured("前提", "clock_uncertainty", "the recognition report")
    uncertainty = float(report.get("clock_uncertainty_ms", float("inf")))
    ok = uncertainty <= MAX_CLOCK_UNCERTAINTY_MS
    return Check(
        condition="前提",
        name="clock_uncertainty",
        status=_verdict(ok),
        detail=f"{uncertainty:.6f} ms <= {MAX_CLOCK_UNCERTAINTY_MS} ms",
        measured={
            "clock_uncertainty_ms": uncertainty,
            "clock_domain": report.get("clock_domain"),
        },
    )


def check_provider_is_cuda(report: dict[str, Any] | None) -> Check:
    if report is None:
        return _not_measured("成功条件4", "cuda_provider", "the recognition report")
    provider = str(report.get("provider", ""))
    ok = provider == "CUDAExecutionProvider"
    return Check(
        condition="成功条件4",
        name="cuda_provider",
        status=_verdict(ok),
        detail=f"actual leading provider {provider!r}; CPU fallback is not success",
        measured={"provider": provider},
    )


def check_frame_completeness(report: dict[str, Any] | None, missing: int | None) -> Check:
    if report is None:
        return _not_measured("成功条件5", "frame_completeness", "the recognition report")
    frame_count = int(report.get("frame_count", 0))
    if missing is None:
        return _not_measured("成功条件5", "frame_completeness", "the missing-face count")
    ok = frame_count == REQUIRED_FRAME_COUNT and missing <= MAX_MISSING_FRAMES
    return Check(
        condition="成功条件5",
        name="frame_completeness",
        status=_verdict(ok),
        detail=(
            f"decoded {frame_count} frames (expected {REQUIRED_FRAME_COUNT}), "
            f"{missing} missing (allowed {MAX_MISSING_FRAMES})"
        ),
        measured={"frame_count": frame_count, "missing_face_frame_count": missing},
    )


def check_accuracy(accuracy: dict[str, Any] | None) -> Check:
    if accuracy is None:
        return _not_measured("成功条件5", "accuracy_vs_full_detect", "the accuracy sweep result")
    eye_p95 = float(accuracy.get("eye_position_p95_mm", float("inf")))
    angle_p95 = float(accuracy.get("angle_p95_deg", float("inf")))
    ok = eye_p95 <= MAX_EYE_POSITION_P95_MM and angle_p95 <= MAX_ANGLE_P95_DEG
    return Check(
        condition="成功条件5",
        name="accuracy_vs_full_detect",
        status=_verdict(ok),
        detail=(
            f"eye p95 {eye_p95:.3f} mm <= {MAX_EYE_POSITION_P95_MM}, "
            f"angle p95 {angle_p95:.3f} deg <= {MAX_ANGLE_P95_DEG}"
        ),
        measured={"eye_position_p95_mm": eye_p95, "angle_p95_deg": angle_p95},
    )


def _recognition_totals(report: dict[str, Any]) -> tuple[float, float] | None:
    stages = report.get("stages", {})
    if not all(stage in stages for stage in RECOGNITION_STAGES):
        return None
    median = sum(float(stages[stage]["p50_ms"]) for stage in RECOGNITION_STAGES)
    p95 = sum(float(stages[stage]["p95_ms"]) for stage in RECOGNITION_STAGES)
    return median, p95


def check_recognition_latency(report: dict[str, Any] | None) -> Check:
    if report is None:
        return _not_measured("成功条件6", "recognition_latency", "the recognition report")
    totals = _recognition_totals(report)
    if totals is None:
        return _not_measured(
            "成功条件6",
            "recognition_latency",
            f"stages {RECOGNITION_STAGES} in the recognition report",
        )
    median, p95 = totals
    ok = median <= MAX_RECOGNITION_MEDIAN_MS and p95 <= MAX_RECOGNITION_P95_MS
    return Check(
        condition="成功条件6",
        name="recognition_latency",
        status=_verdict(ok),
        detail=(
            f"median {median:.3f} ms <= {MAX_RECOGNITION_MEDIAN_MS}, "
            f"p95 {p95:.3f} ms <= {MAX_RECOGNITION_P95_MS} "
            f"(sum of {', '.join(RECOGNITION_STAGES)}; "
            "the p95 sum is a conservative upper bound, not an additive percentile)"
        ),
        measured={"recognition_median_ms": median, "recognition_p95_ms": p95},
    )


def check_transport(transport: dict[str, Any] | None) -> Check:
    if transport is None:
        return _not_measured("成功条件7", "control_transport", "the transport comparison result")
    p95 = float(transport.get("control_p95_ms", float("inf")))
    catch_up = int(transport.get("catch_up_frames", 10**6))
    reversals = int(transport.get("sequence_reversals", 10**6))
    ok = p95 <= MAX_CONTROL_P95_MS and catch_up <= MAX_CATCH_UP_FRAMES and reversals == 0
    return Check(
        condition="成功条件7",
        name="control_transport",
        status=_verdict(ok),
        detail=(
            f"control p95 {p95:.3f} ms <= {MAX_CONTROL_P95_MS}, "
            f"catch-up {catch_up} <= {MAX_CATCH_UP_FRAMES} frames, "
            f"sequence reversals {reversals} == 0"
        ),
        measured={
            "control_p95_ms": p95,
            "catch_up_frames": catch_up,
            "sequence_reversals": reversals,
            "candidate": transport.get("candidate"),
        },
    )


def check_preview(preview: dict[str, Any] | None) -> Check:
    if preview is None:
        return _not_measured("成功条件8", "preview_lane", "the preview measurement")
    width = int(preview.get("width_px", 0))
    height = int(preview.get("height_px", 0))
    fps = float(preview.get("max_fps", float("inf")))
    reencodes = int(preview.get("server_reencode_count", 10**6))
    ok = (
        width == PREVIEW_WIDTH_PX
        and height == PREVIEW_HEIGHT_PX
        and fps <= MAX_PREVIEW_FPS
        and reencodes <= MAX_SERVER_REENCODES
    )
    return Check(
        condition="成功条件8",
        name="preview_lane",
        status=_verdict(ok),
        detail=(
            f"{width}x{height} (expected {PREVIEW_WIDTH_PX}x{PREVIEW_HEIGHT_PX}), "
            f"{fps:.2f} fps <= {MAX_PREVIEW_FPS}, "
            f"server re-encodes {reencodes} <= {MAX_SERVER_REENCODES}"
        ),
        measured={
            "width_px": width,
            "height_px": height,
            "max_fps": fps,
            "server_reencode_count": reencodes,
        },
    )


def check_browser(browser: dict[str, Any] | None) -> Check:
    if browser is None:
        return _not_measured("成功条件9", "browser_draw", "the browser timing measurement")
    receive_to_draw = float(browser.get("receive_to_draw_p95_ms", float("inf")))
    cpu_draw = float(browser.get("cpu_draw_p95_ms", float("inf")))
    reversals = int(browser.get("sequence_reversals", 10**6))
    ok = (
        receive_to_draw <= MAX_RECEIVE_TO_DRAW_P95_MS
        and cpu_draw <= MAX_CPU_DRAW_P95_MS
        and reversals == 0
    )
    return Check(
        condition="成功条件9",
        name="browser_draw",
        status=_verdict(ok),
        detail=(
            f"receive-to-draw p95 {receive_to_draw:.3f} ms <= {MAX_RECEIVE_TO_DRAW_P95_MS}, "
            f"CPU draw p95 {cpu_draw:.3f} ms <= {MAX_CPU_DRAW_P95_MS}, "
            f"sequence reversals {reversals} == 0"
        ),
        measured={
            "receive_to_draw_p95_ms": receive_to_draw,
            "cpu_draw_p95_ms": cpu_draw,
            "sequence_reversals": reversals,
            "gpu_timing_available": browser.get("gpu_timing_available"),
        },
    )


def check_end_to_end(end_to_end: dict[str, Any] | None) -> Check:
    if end_to_end is None:
        return _not_measured("成功条件10", "inference_to_webgl", "the end-to-end measurement")
    median = float(end_to_end.get("median_ms", float("inf")))
    p95 = float(end_to_end.get("p95_ms", float("inf")))
    ok = median <= MAX_END_TO_END_MEDIAN_MS and p95 <= MAX_END_TO_END_P95_MS
    return Check(
        condition="成功条件10",
        name="inference_to_webgl",
        status=_verdict(ok),
        detail=(
            f"median {median:.3f} ms <= {MAX_END_TO_END_MEDIAN_MS}, "
            f"p95 {p95:.3f} ms <= {MAX_END_TO_END_P95_MS}; "
            "camera exposure and capture time are NOT included"
        ),
        measured={"median_ms": median, "p95_ms": p95},
    )


def _stage_delta(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """Per-stage before/after, for stages present in both reports."""

    before = baseline.get("stages", {})
    after = final.get("stages", {})
    delta: dict[str, Any] = {}
    for stage in sorted(set(before) & set(after)):
        for percentile in ("p50_ms", "p95_ms", "p99_ms"):
            old = float(before[stage][percentile])
            new = float(after[stage][percentile])
            delta.setdefault(stage, {})[percentile] = {
                "before": old,
                "after": new,
                "delta": new - old,
                "ratio": (new / old) if old > 0 else None,
            }
    for stage in sorted(set(before) ^ set(after)):
        delta[stage] = {"note": "present in only one report; not comparable"}
    return delta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True, help="PerformanceReport JSON")
    parser.add_argument("--baseline", type=Path, help="Earlier PerformanceReport JSON")
    parser.add_argument("--missing-faces", type=int, help="Frames with no detected face")
    parser.add_argument("--accuracy", type=Path, help="Accuracy-sweep result JSON")
    parser.add_argument("--transport", type=Path, help="Adopted transport result JSON")
    parser.add_argument("--preview", type=Path, help="Preview-lane measurement JSON")
    parser.add_argument("--browser", type=Path, help="Browser timing JSON")
    parser.add_argument("--end-to-end", type=Path, help="Inference-to-WebGL timing JSON")
    parser.add_argument("--output", type=Path, help="Write the verdict JSON here")
    return parser


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    final = _load(args.final)
    baseline = _load(args.baseline)
    checks = [
        check_clock_trust(final),
        check_provider_is_cuda(final),
        check_frame_completeness(final, args.missing_faces),
        check_accuracy(_load(args.accuracy)),
        check_recognition_latency(final),
        check_transport(_load(args.transport)),
        check_preview(_load(args.preview)),
        check_browser(_load(args.browser)),
        check_end_to_end(_load(args.end_to_end)),
    ]
    failed = [check for check in checks if check.status != "pass"]
    return {
        "verdict": "pass" if not failed else "fail",
        "failed_count": len(failed),
        "final_commit": (final or {}).get("commit"),
        "baseline_commit": (baseline or {}).get("commit"),
        "before_after": _stage_delta(baseline, final) if baseline and final else None,
        "checks": [check.as_dict() for check in checks],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verdict = run_checks(args)
    text = json.dumps(verdict, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
