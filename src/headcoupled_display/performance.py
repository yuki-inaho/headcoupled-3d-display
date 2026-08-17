"""Fixed-schema stage-latency performance reports.

Percentiles are aggregated with ``numpy.percentile`` using linear
interpolation (``method="linear"``, numpy's default) over per-stage duration
samples expressed in milliseconds. Reports are written once per benchmark
run and reloaded verbatim by tooling in ``docs/performance_results.md``, so
``StrictModel`` rejects unknown keys and ``model_dump_json`` serializes
fields in declaration order: saved JSON has a stable, diffable key order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from beartype import beartype
from pydantic import Field, field_validator, model_validator

from .models import StrictModel, utc_now_iso

#: Nanosecond-resolution clock domains used across this project's latency
#: pipeline: producer/server timestamps are host monotonic ns, browser
#: timestamps are Unix ns derived from ``performance.timeOrigin +
#: performance.now()``. Kept closed (not a free-form string) so adding a
#: third clock source requires an explicit schema change, not an implicit one.
ClockDomain = Literal["monotonic_ns", "unix_ns"]

# ONNX Runtime execution providers are all named "<Backend>ExecutionProvider"
# (CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider, ...).
# Validating the suffix rejects empty/placeholder values without hard-coding
# an allowlist that would need updating for every new backend.
_PROVIDER_SUFFIX = "ExecutionProvider"


class FrameResolution(StrictModel):
    """Pixel dimensions of the frames a report was measured on."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class StagePercentiles(StrictModel):
    """p50/p95/p99 latency of one pipeline stage, in milliseconds."""

    sample_count: int = Field(gt=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_percentile_order(self) -> StagePercentiles:
        if not (self.p50_ms <= self.p95_ms <= self.p99_ms):
            raise ValueError("percentiles must satisfy p50_ms <= p95_ms <= p99_ms")
        return self


class PerformanceReport(StrictModel):
    """A single benchmark run's fixed-schema latency report.

    ``stages`` maps a pipeline stage name (e.g. ``"detector"``) to its
    percentile summary; at least one stage is required.
    """

    schema_version: int = 1
    commit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    provider: str
    resolution: FrameResolution
    frame_count: int = Field(gt=0)
    warmup: int = Field(ge=0)
    clock_domain: ClockDomain
    clock_uncertainty_ms: float = Field(ge=0)
    stages: dict[str, StagePercentiles]
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if not value or not value.endswith(_PROVIDER_SUFFIX):
            raise ValueError(
                "provider must be a non-empty ONNX Runtime provider name ending in "
                f"{_PROVIDER_SUFFIX!r}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def validate_stages(self) -> PerformanceReport:
        if not self.stages:
            raise ValueError("at least one stage is required")
        return self

    @classmethod
    def load(cls, path: Path) -> PerformanceReport:
        with path.open(encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


@beartype
def compute_stage_percentiles(samples_ms: Sequence[float]) -> StagePercentiles:
    """Aggregate raw stage-duration samples (ms) into p50/p95/p99.

    Rejects an empty sample sequence and any non-finite (NaN/inf) or
    negative duration before aggregating.
    """

    if len(samples_ms) == 0:
        raise ValueError("stage samples must not be empty")
    for value in samples_ms:
        if not math.isfinite(value):
            raise ValueError(f"stage sample duration must be finite, got {value!r}")
        if value < 0:
            raise ValueError(f"stage sample duration must not be negative, got {value!r}")

    array = np.asarray(samples_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, [50, 95, 99], method="linear")
    return StagePercentiles(
        sample_count=len(samples_ms),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
    )


@beartype
def build_performance_report(
    *,
    commit: str,
    source: str,
    provider: str,
    resolution: FrameResolution,
    frame_count: int,
    warmup: int,
    clock_domain: ClockDomain,
    clock_uncertainty_ms: float,
    stage_samples_ms: Mapping[str, Sequence[float]],
) -> PerformanceReport:
    """Build a :class:`PerformanceReport` from raw per-stage duration samples (ms).

    Stage insertion order in ``stage_samples_ms`` is preserved in the
    resulting report and its saved JSON.
    """

    stages = {
        name: compute_stage_percentiles(samples) for name, samples in stage_samples_ms.items()
    }
    return PerformanceReport(
        commit=commit,
        source=source,
        provider=provider,
        resolution=resolution,
        frame_count=frame_count,
        warmup=warmup,
        clock_domain=clock_domain,
        clock_uncertainty_ms=clock_uncertainty_ms,
        stages=stages,
    )
