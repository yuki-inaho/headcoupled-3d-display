"""Unit tests for the recorded-baseline summarizer (workdoc Step 7).

These tests exercise only the pure raw-JSON -> ``PerformanceReport`` transformation
and the markdown/report I/O helpers in ``scripts/summarize_performance.py``. They
never require a real GPU or ``recordings/test10.avi``: a small synthetic raw
benchmark dict (shaped like ``scripts/benchmark_recorded.py``'s output, but with
tiny sample counts) stands in as a fixture.

Written first (t-wada style RED) against ``scripts.summarize_performance``, which
does not exist yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from headcoupled_display.performance import PerformanceReport
from scripts.summarize_performance import (
    build_report_from_raw,
    format_results_section,
    load_raw_benchmark,
    reference_order_of_magnitude,
    sha256_of_file,
)


def _raw_record(**overrides: object) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "commit": "abc1234",
        "facemesh_tracking_commit": "def5678",
        "facemesh_tracking_dirty": True,
        "source": "/home/inaho-omen/Project/facemesh_tracking/recordings/test10.avi",
        "provider": "CUDAExecutionProvider",
        "providers_by_stage": {
            "detector": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "estimator": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
        "resolution": {"width_px": 1280, "height_px": 720},
        "frame_count": 294,
        "warmup": 5,
        "clock_domain": "monotonic_ns",
        "clock_uncertainty_ms": 0.001,
        "stage_order": [
            "capture_decode",
            "detector",
            "landmarks",
            "packet_build",
            "preview_resize_encode",
        ],
        "stage_separation_note": "measured individually",
        "stage_samples_ms": {
            "capture_decode": [2.0, 2.1, 2.2, 2.3, 2.4],
            "detector": [22.0, 22.5, 23.0, 23.5, 24.0],
            "landmarks": [9.0, 9.5, 10.0, 10.5, 11.0],
            "packet_build": [0.1, 0.12, 0.14, 0.16, 0.18],
            "preview_resize_encode": [1.0, 1.1, 1.2, 1.3, 1.4],
        },
        "missing_face_frame_count": 0,
        "preview": {
            "width_px": 640,
            "height_px": 360,
            "jpeg_quality": 82,
            "bytes_per_frame": [12000, 12100, 11900, 12200, 12050],
        },
        "avi_header": {"frame_count": 602, "fps": 60.0, "trusted": False},
        "generated_at": "2026-08-17T12:00:00+00:00",
    }
    record.update(overrides)
    return record


# --- load_raw_benchmark ------------------------------------------------------


def test_load_raw_benchmark_reads_json_file(tmp_path: Path) -> None:
    raw = _raw_record()
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_raw_benchmark(path)

    assert loaded == raw


# --- build_report_from_raw ---------------------------------------------------


def test_build_report_from_raw_produces_valid_performance_report() -> None:
    raw = _raw_record()

    report = build_report_from_raw(raw)

    assert isinstance(report, PerformanceReport)
    assert report.commit == "abc1234"
    assert report.source.endswith("test10.avi")
    assert report.provider == "CUDAExecutionProvider"
    assert report.resolution.width_px == 1280
    assert report.resolution.height_px == 720
    assert report.frame_count == 294
    assert report.warmup == 5
    assert report.clock_domain == "monotonic_ns"
    assert list(report.stages.keys()) == [
        "capture_decode",
        "detector",
        "landmarks",
        "packet_build",
        "preview_resize_encode",
    ]
    assert report.stages["detector"].sample_count == 5
    assert report.stages["detector"].p50_ms == pytest.approx(23.0)


def test_build_report_from_raw_round_trips_through_json() -> None:
    report = build_report_from_raw(_raw_record())
    reloaded = PerformanceReport.model_validate_json(report.model_dump_json())
    assert reloaded == report


def test_build_report_from_raw_propagates_bad_provider() -> None:
    raw = _raw_record(provider="not-a-real-provider")
    with pytest.raises(ValidationError, match="provider"):
        build_report_from_raw(raw)


def test_build_report_from_raw_propagates_empty_stage_samples() -> None:
    # compute_stage_percentiles runs before PerformanceReport is constructed, so this
    # raises a plain ValueError rather than pydantic.ValidationError -- matching
    # test_performance_report.py's own test_build_performance_report_propagates_stage_sample_validation.
    raw = _raw_record(stage_samples_ms={"detector": []})
    with pytest.raises(ValueError, match="empty"):
        build_report_from_raw(raw)


# --- reference_order_of_magnitude --------------------------------------------


def test_reference_order_of_magnitude_flags_same_order() -> None:
    report = build_report_from_raw(_raw_record())
    result = reference_order_of_magnitude(report, known_baseline_ms=33.16)
    # detector p50 (23.0) + landmarks p50 (10.0) = 33.0 ms, essentially matching.
    assert result.combined_ms == pytest.approx(33.0)
    assert result.same_order_of_magnitude is True


def test_reference_order_of_magnitude_flags_different_order() -> None:
    raw = _raw_record(
        stage_samples_ms={
            "capture_decode": [2.0],
            "detector": [2200.0],
            "landmarks": [1000.0],
            "packet_build": [0.1],
            "preview_resize_encode": [1.0],
        }
    )
    report = build_report_from_raw(raw)
    result = reference_order_of_magnitude(report, known_baseline_ms=33.16)
    assert result.same_order_of_magnitude is False


# --- sha256_of_file ------------------------------------------------------------


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_bytes(b'{"hello": "world"}')

    digest = sha256_of_file(path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


# --- format_results_section ---------------------------------------------------


def test_format_results_section_contains_key_fields() -> None:
    report = build_report_from_raw(_raw_record())
    reference = reference_order_of_magnitude(report, known_baseline_ms=33.16)

    section = format_results_section(
        report=report,
        raw_path=Path("/tmp/raw.json"),
        raw_sha256="deadbeef" * 8,
        command="uv run python scripts/benchmark_recorded.py --video x --output y",
        label="手順7: 現行録画ベースライン",
        missing_face_frame_count=0,
        reference=reference,
    )

    assert "abc1234" in section
    assert "deadbeef" in section
    assert "CUDAExecutionProvider" in section
    assert "uv run python scripts/benchmark_recorded.py" in section
    assert "294" in section
    assert "capture_decode" in section
    assert "detector" in section
