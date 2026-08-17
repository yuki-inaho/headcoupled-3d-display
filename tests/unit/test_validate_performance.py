"""The performance gate must fail loudly on absent inputs, not quietly pass.

The failure mode this guards against is a run reported as green because a stage was
never measured. Every check therefore has an explicit ``not_measured`` state that counts
as a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.validate_performance import (
    MAX_RECOGNITION_MEDIAN_MS,
    REQUIRED_FRAME_COUNT,
    check_accuracy,
    check_end_to_end,
    check_frame_completeness,
    check_preview,
    check_provider_is_cuda,
    check_recognition_latency,
    check_transport,
    run_checks,
)


def report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "commit": "0" * 40,
        "source": "/tmp/test10.avi",
        "provider": "CUDAExecutionProvider",
        "resolution": {"width_px": 1280, "height_px": 720},
        "frame_count": REQUIRED_FRAME_COUNT,
        "warmup": 5,
        "clock_domain": "monotonic_ns",
        "clock_uncertainty_ms": 0.001,
        "stages": {
            "detector": {"sample_count": 289, "p50_ms": 2.0, "p95_ms": 4.0, "p99_ms": 5.0},
            "landmarks": {"sample_count": 289, "p50_ms": 9.0, "p95_ms": 14.0, "p99_ms": 18.0},
        },
    }
    base.update(overrides)
    return base


def write(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_absent_inputs_are_not_measured_rather_than_passing() -> None:
    for check in (
        check_accuracy(None),
        check_transport(None),
        check_preview(None),
        check_end_to_end(None),
    ):
        assert check.status == "not_measured"


def test_cpu_fallback_is_not_a_pass() -> None:
    assert check_provider_is_cuda(report(provider="CPUExecutionProvider")).status == "fail"
    assert check_provider_is_cuda(report()).status == "pass"


def test_recognition_latency_sums_the_detector_and_landmark_stages() -> None:
    passing = check_recognition_latency(report())
    assert passing.status == "pass"
    assert passing.measured["recognition_median_ms"] == pytest.approx(11.0)

    slow = report(
        stages={
            "detector": {"sample_count": 289, "p50_ms": 29.6, "p95_ms": 39.5, "p99_ms": 50.0},
            "landmarks": {"sample_count": 289, "p50_ms": 13.1, "p95_ms": 19.2, "p99_ms": 24.0},
        }
    )
    failing = check_recognition_latency(slow)
    assert failing.status == "fail"
    assert failing.measured["recognition_median_ms"] > MAX_RECOGNITION_MEDIAN_MS


def test_recognition_latency_is_not_measured_when_a_stage_is_missing() -> None:
    partial = report(
        stages={"landmarks": {"sample_count": 289, "p50_ms": 9.0, "p95_ms": 14.0, "p99_ms": 18.0}}
    )
    assert check_recognition_latency(partial).status == "not_measured"


def test_frame_completeness_rejects_a_short_or_lossy_run() -> None:
    assert check_frame_completeness(report(), 0).status == "pass"
    assert check_frame_completeness(report(), 3).status == "fail"
    assert check_frame_completeness(report(frame_count=120), 0).status == "fail"
    assert check_frame_completeness(report(), None).status == "not_measured"


def test_accuracy_uses_p95_not_mean() -> None:
    assert check_accuracy({"eye_position_p95_mm": 4.9, "angle_p95_deg": 0.9}).status == "pass"
    assert check_accuracy({"eye_position_p95_mm": 5.1, "angle_p95_deg": 0.9}).status == "fail"
    assert check_accuracy({"eye_position_p95_mm": 4.9, "angle_p95_deg": 1.1}).status == "fail"


def test_transport_requires_all_three_properties() -> None:
    good = {"control_p95_ms": 1.5, "catch_up_frames": 1, "sequence_reversals": 0}
    assert check_transport(good).status == "pass"
    assert check_transport({**good, "control_p95_ms": 2.5}).status == "fail"
    assert check_transport({**good, "catch_up_frames": 3}).status == "fail"
    assert check_transport({**good, "sequence_reversals": 1}).status == "fail"


def test_preview_rejects_a_full_resolution_or_re_encoded_lane() -> None:
    good = {"width_px": 640, "height_px": 360, "max_fps": 10.0, "server_reencode_count": 0}
    assert check_preview(good).status == "pass"
    assert check_preview({**good, "width_px": 1280, "height_px": 720}).status == "fail"
    assert check_preview({**good, "server_reencode_count": 1}).status == "fail"
    assert check_preview({**good, "max_fps": 30.0}).status == "fail"


def test_end_to_end_states_that_camera_exposure_is_excluded() -> None:
    check = check_end_to_end({"median_ms": 30.0, "p95_ms": 55.0})
    assert check.status == "pass"
    assert "exposure" in check.detail


def test_overall_verdict_fails_while_any_input_is_missing(tmp_path: Path) -> None:
    final = write(tmp_path, "final.json", report())
    args = argparse.Namespace(
        final=final,
        baseline=None,
        missing_faces=0,
        accuracy=None,
        transport=None,
        preview=None,
        browser=None,
        end_to_end=None,
        output=None,
    )
    verdict = run_checks(args)
    assert verdict["verdict"] == "fail"
    assert verdict["failed_count"] > 0
    statuses = {check["name"]: check["status"] for check in verdict["checks"]}
    assert statuses["cuda_provider"] == "pass"
    assert statuses["control_transport"] == "not_measured"


def test_before_after_is_reported_per_stage(tmp_path: Path) -> None:
    baseline = write(
        tmp_path,
        "baseline.json",
        report(
            stages={
                "detector": {"sample_count": 289, "p50_ms": 29.6, "p95_ms": 39.5, "p99_ms": 50.0},
                "landmarks": {"sample_count": 289, "p50_ms": 13.1, "p95_ms": 19.2, "p99_ms": 24.0},
            }
        ),
    )
    final = write(tmp_path, "final.json", report())
    args = argparse.Namespace(
        final=final,
        baseline=baseline,
        missing_faces=0,
        accuracy=None,
        transport=None,
        preview=None,
        browser=None,
        end_to_end=None,
        output=None,
    )
    verdict = run_checks(args)
    detector = verdict["before_after"]["detector"]["p50_ms"]
    assert detector["before"] == pytest.approx(29.6)
    assert detector["after"] == pytest.approx(2.0)
    assert detector["delta"] < 0


def test_a_directly_measured_combined_stage_is_preferred_over_a_sum() -> None:
    """Summing per-stage percentiles understates the median of the combined block.

    Measured on the recording: p50(detector) + p50(landmarks) = 42.7 ms while the true
    p50 of the whole recognition block was 59.4 ms.
    """

    combined = report(
        stages={
            "detector": {"sample_count": 289, "p50_ms": 2.0, "p95_ms": 4.0, "p99_ms": 5.0},
            "landmarks": {"sample_count": 289, "p50_ms": 9.0, "p95_ms": 14.0, "p99_ms": 18.0},
            "recognition_total": {
                "sample_count": 289,
                "p50_ms": 20.0,
                "p95_ms": 30.0,
                "p99_ms": 40.0,
            },
        }
    )
    check = check_recognition_latency(combined)
    assert check.measured["basis"] == "recognition_total"
    assert check.measured["recognition_median_ms"] == pytest.approx(20.0)
    # The sum would have said 11.0 ms and passed; the real measurement fails.
    assert check.status == "fail"


def test_the_summed_fallback_says_so_in_its_detail() -> None:
    check = check_recognition_latency(report())
    assert check.measured["basis"].startswith("sum of ")
    assert "understates the median" in check.detail
