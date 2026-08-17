"""Pins the fixed schema and percentile aggregation of performance reports.

Written first (t-wada style RED) against ``headcoupled_display.performance``,
which does not exist yet.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from headcoupled_display.performance import (
    FrameResolution,
    PerformanceReport,
    StagePercentiles,
    build_performance_report,
    compute_stage_percentiles,
)

RESOLUTION = FrameResolution(width_px=1280, height_px=720)


def _valid_report_kwargs(**overrides: object) -> dict:
    kwargs: dict = {
        "commit": "abc1234",
        "source": "/home/inaho-omen/Project/facemesh_tracking/recordings/test10.avi",
        "provider": "CUDAExecutionProvider",
        "resolution": RESOLUTION,
        "frame_count": 294,
        "warmup": 5,
        "clock_domain": "monotonic_ns",
        "clock_uncertainty_ms": 0.5,
        "stage_samples_ms": {
            "detector": [10.0, 11.0, 12.0, 13.0, 14.0],
            "landmarks": [1.0, 1.5, 2.0, 2.5, 3.0],
        },
    }
    kwargs.update(overrides)
    return kwargs


# --- schema: required fields and stable JSON key order --------------------


def test_report_round_trips_and_exposes_required_fields() -> None:
    report = build_performance_report(**_valid_report_kwargs())

    assert report.commit == "abc1234"
    assert report.source.endswith("test10.avi")
    assert report.provider == "CUDAExecutionProvider"
    assert report.resolution.width_px == 1280
    assert report.resolution.height_px == 720
    assert report.frame_count == 294
    assert report.warmup == 5
    assert report.clock_domain == "monotonic_ns"
    assert report.clock_uncertainty_ms == pytest.approx(0.5)
    for stage in ("detector", "landmarks"):
        percentiles = report.stages[stage]
        assert percentiles.p50_ms <= percentiles.p95_ms <= percentiles.p99_ms

    reloaded = PerformanceReport.model_validate_json(report.model_dump_json())
    assert reloaded == report


def test_report_json_keys_are_in_declared_order() -> None:
    report = build_performance_report(**_valid_report_kwargs())
    keys = list(json.loads(report.model_dump_json()).keys())

    assert keys == [
        "schema_version",
        "commit",
        "source",
        "provider",
        "resolution",
        "frame_count",
        "warmup",
        "clock_domain",
        "clock_uncertainty_ms",
        "stages",
        "created_at",
    ]
    stage_keys = list(json.loads(report.model_dump_json())["stages"]["detector"].keys())
    assert stage_keys == ["sample_count", "p50_ms", "p95_ms", "p99_ms"]


def test_report_stage_order_matches_input_insertion_order() -> None:
    kwargs = _valid_report_kwargs(
        stage_samples_ms={
            "capture": [1.0, 2.0],
            "detector": [10.0, 20.0],
            "landmarks": [1.0, 2.0],
            "packet_build": [0.1, 0.2],
            "preview_encode": [3.0, 4.0],
        }
    )
    report = build_performance_report(**kwargs)
    assert list(report.stages.keys()) == [
        "capture",
        "detector",
        "landmarks",
        "packet_build",
        "preview_encode",
    ]


# --- percentile aggregation -------------------------------------------------


def test_compute_stage_percentiles_matches_numpy_linear_interpolation() -> None:
    samples = [12.0, 5.0, 30.0, 18.0, 24.0, 9.0, 15.0, 21.0, 27.0, 3.0]
    expected_p50, expected_p95, expected_p99 = np.percentile(
        np.asarray(samples, dtype=np.float64), [50, 95, 99], method="linear"
    )

    percentiles = compute_stage_percentiles(samples)

    assert percentiles.sample_count == len(samples)
    assert percentiles.p50_ms == pytest.approx(float(expected_p50))
    assert percentiles.p95_ms == pytest.approx(float(expected_p95))
    assert percentiles.p99_ms == pytest.approx(float(expected_p99))


def test_compute_stage_percentiles_single_sample_is_all_percentiles() -> None:
    percentiles = compute_stage_percentiles([7.5])
    assert percentiles.sample_count == 1
    assert percentiles.p50_ms == pytest.approx(7.5)
    assert percentiles.p95_ms == pytest.approx(7.5)
    assert percentiles.p99_ms == pytest.approx(7.5)


# --- rejections: NaN, empty samples, negative duration, bad provider -------


def test_compute_stage_percentiles_rejects_nan_sample() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_stage_percentiles([1.0, math.nan, 2.0])


def test_compute_stage_percentiles_rejects_infinite_sample() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_stage_percentiles([1.0, math.inf, 2.0])


def test_compute_stage_percentiles_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_stage_percentiles([])


def test_compute_stage_percentiles_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="negative"):
        compute_stage_percentiles([1.0, -0.5, 2.0])


def test_build_performance_report_rejects_empty_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        build_performance_report(**_valid_report_kwargs(provider=""))


def test_build_performance_report_rejects_unknown_provider_name() -> None:
    with pytest.raises(ValueError, match="provider"):
        build_performance_report(**_valid_report_kwargs(provider="totally-made-up"))


def test_build_performance_report_rejects_empty_stage_map() -> None:
    with pytest.raises(ValueError, match="stage"):
        build_performance_report(**_valid_report_kwargs(stage_samples_ms={}))


def test_build_performance_report_propagates_stage_sample_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_performance_report(**_valid_report_kwargs(stage_samples_ms={"detector": []}))


def test_performance_report_direct_construction_rejects_empty_stages() -> None:
    with pytest.raises(ValueError, match="stage"):
        PerformanceReport(
            commit="abc1234",
            source="synthetic",
            provider="CUDAExecutionProvider",
            resolution=RESOLUTION,
            frame_count=1,
            warmup=0,
            clock_domain="monotonic_ns",
            clock_uncertainty_ms=0.0,
            stages={},
        )


def test_stage_percentiles_direct_construction_rejects_zero_sample_count() -> None:
    with pytest.raises(ValueError):
        StagePercentiles(sample_count=0, p50_ms=1.0, p95_ms=2.0, p99_ms=3.0)


def test_stage_percentiles_direct_construction_rejects_negative_percentile() -> None:
    with pytest.raises(ValueError):
        StagePercentiles(sample_count=3, p50_ms=-1.0, p95_ms=2.0, p99_ms=3.0)
