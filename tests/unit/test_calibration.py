from __future__ import annotations

from pathlib import Path

from headcoupled_display.models import HardwareProfile
from headcoupled_display.synthetic import run_synthetic_calibration

ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_calibration_recovers_mount_geometry() -> None:
    profile = HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")
    dataset, result = run_synthetic_calibration(profile)

    assert len(dataset.samples) == 36
    assert result.metrics.optimizer_success is True
    assert result.metrics.unique_target_count == 9
    assert result.metrics.mean_point_to_ray_error_mm < 2.5
    assert result.comparison_to_ground_truth is not None
    assert result.comparison_to_ground_truth["height_error_mm"] < 0.5
    assert result.comparison_to_ground_truth["translation_error_mm"] < 1.5
    assert result.comparison_to_ground_truth["pitch_error_deg"] < 0.35
    assert result.mount_summary.horizontally_centered is True
