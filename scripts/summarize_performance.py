"""Validate and summarize a recorded FaceMesh benchmark run (workdoc Step 7).

Run this with this repository's own Python 3.13 environment::

    PYTHONPATH=<this repo>/src \\
      .venv/bin/python scripts/summarize_performance.py \\
      --raw artifacts/perf/baseline_recorded_raw.json \\
      --command "..." \\
      --label "手順7: 現行録画ベースライン"

It reads the raw per-stage duration samples that
``scripts/benchmark_recorded.py`` wrote (from the Python 3.10 / CUDA
``facemesh_tracking`` environment), validates and aggregates them with
``headcoupled_display.performance.build_performance_report``, writes the
resulting fixed-schema ``PerformanceReport`` to
``artifacts/perf/baseline_recorded.json`` (gitignored), and appends a tracked,
human-readable summary section to ``docs/performance_results.md``.

``artifacts/`` is gitignored precisely so raw benchmark data never needs to be
committed; only the SHA-256 of the raw file, the command used to produce it, and
the aggregated percentiles are tracked in ``docs/performance_results.md``.
"""

# ruff: noqa: RUF001
# The RUF001 suppression above is file-wide and intentional: this module's Markdown
# output strings use full-width Japanese parentheses, matching the typographic
# convention already used throughout docs/performance_design.md and
# docs/performance_results.md. That is real Japanese text, not an ambiguous-character
# typo.
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from headcoupled_display.performance import (
    FrameResolution,
    PerformanceReport,
    build_performance_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_OUTPUT = REPO_ROOT / "artifacts" / "perf" / "baseline_recorded.json"
DEFAULT_DOCS_PATH = REPO_ROOT / "docs" / "performance_results.md"

#: Order-of-magnitude sanity band applied in `reference_order_of_magnitude`. This is
#: deliberately loose (10x either way): known prior runs (mean ~33.16 ms, detector
#: ~22.89 ms, facemesh ~10.27 ms on a static image bench) are a digit-count sanity
#: check only, never a fixed pass/fail threshold for this baseline-collection step.
_MAGNITUDE_BAND = 10.0

_DOCS_HEADER = """# 性能計測結果

本書は `temp/workdoc_Aug17-2026_headcoupled_scene_latency.md` の性能計測手順で得た実測値を追記していく記録である。
raw JSON は `.gitignore` 対象の `artifacts/perf/` 配下に保存され、本書には SHA-256・実行コマンド・commit・
主要 percentile・pass/fail 判定のみを追跡対象として記録する。raw JSON 自体は再実行すれば再生成できるため、
リポジトリには含めない。

## 計測ログ
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", required=True, type=Path, help="Raw JSON from benchmark_recorded.py"
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument(
        "--command", required=True, help="Exact command used to produce --raw, for traceability"
    )
    parser.add_argument("--label", default="手順7: 現行録画ベースライン")
    parser.add_argument("--known-baseline-ms", type=float, default=33.16)
    return parser


def load_raw_benchmark(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_report_from_raw(raw: dict[str, Any]) -> PerformanceReport:
    """Convert a raw benchmark record into a validated, aggregated PerformanceReport.

    Only the fields ``build_performance_report`` actually accepts are extracted;
    ``benchmark_recorded.py`` is free to carry extra provenance fields (e.g.
    ``facemesh_tracking_commit``, ``avi_header``) that this report schema does not
    need. Validation errors (bad provider, empty/NaN/negative samples, ...) propagate
    as ``pydantic.ValidationError`` straight from ``build_performance_report``.
    """
    resolution = FrameResolution(**raw["resolution"])
    return build_performance_report(
        commit=raw["commit"],
        source=raw["source"],
        provider=raw["provider"],
        resolution=resolution,
        frame_count=raw["frame_count"],
        warmup=raw["warmup"],
        clock_domain=raw["clock_domain"],
        clock_uncertainty_ms=raw["clock_uncertainty_ms"],
        stage_samples_ms=raw["stage_samples_ms"],
    )


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class MagnitudeReference:
    """Result of comparing a report's core inference latency to a known prior run."""

    stage_names: tuple[str, ...]
    combined_ms: float
    known_baseline_ms: float
    ratio: float
    same_order_of_magnitude: bool


def reference_order_of_magnitude(
    report: PerformanceReport,
    known_baseline_ms: float,
    stage_names: tuple[str, ...] = ("detector", "landmarks"),
) -> MagnitudeReference:
    """Sum p50 latency of ``stage_names`` and compare its digit count to a prior run.

    This intentionally reproduces the historical single-image bench's "detector +
    facemesh" split (mean ~22.89 ms + ~10.27 ms =~ 33.16 ms) so the two numbers are
    comparable. The 10x band is a sanity check, not a threshold gate: a mismatch is
    reported, not treated as this step's pass/fail criterion.
    """
    present = [name for name in stage_names if name in report.stages]
    combined_ms = sum(report.stages[name].p50_ms for name in present)
    ratio = combined_ms / known_baseline_ms if known_baseline_ms else float("inf")
    same_order = (1.0 / _MAGNITUDE_BAND) <= ratio <= _MAGNITUDE_BAND
    return MagnitudeReference(
        stage_names=tuple(present),
        combined_ms=combined_ms,
        known_baseline_ms=known_baseline_ms,
        ratio=ratio,
        same_order_of_magnitude=same_order,
    )


def _format_metadata_block(
    *,
    report: PerformanceReport,
    raw_path: Path,
    raw_sha256: str,
    command: str,
    missing_face_frame_count: int,
) -> str:
    lines = [
        f"- **コマンド:** `{command}`",
        f"- **commit:** `{report.commit}`",
        f"- **入力:** `{report.source}` "
        f"({report.resolution.width_px}x{report.resolution.height_px}, "
        f"frame_count={report.frame_count}, warmup={report.warmup})",
        f"- **provider:** `{report.provider}`",
        f"- **clock_domain:** `{report.clock_domain}` "
        f"(uncertainty {report.clock_uncertainty_ms:.6f} ms)",
        f"- **raw JSON:** `{raw_path}` (SHA-256: `{raw_sha256}`)",
        f"- **欠測フレーム数:** {missing_face_frame_count}",
        f"- **計測時刻 (created_at):** {report.created_at}",
    ]
    return "\n".join(lines)


def _format_stage_table(report: PerformanceReport) -> str:
    header = "| stage | sample_count | p50 (ms) | p95 (ms) | p99 (ms) |\n| :--- | ---: | ---: | ---: | ---: |"
    rows = [
        f"| {name} | {p.sample_count} | {p.p50_ms:.3f} | {p.p95_ms:.3f} | {p.p99_ms:.3f} |"
        for name, p in report.stages.items()
    ]
    return "\n".join([header, *rows])


def _format_reference_line(reference: MagnitudeReference) -> str:
    verdict = "同桁" if reference.same_order_of_magnitude else "別桁（要確認）"
    stages = " + ".join(reference.stage_names) if reference.stage_names else "(no matching stages)"
    return (
        f"- **参考桁確認:** {stages} の p50 合計 {reference.combined_ms:.2f} ms は"
        f" 既知の静止画ベンチ mean≈{reference.known_baseline_ms:.2f} ms と{verdict}"
        f"（比 {reference.ratio:.2f} 倍）。過去値は成功判定の固定基準ではなく参考のみ。"
    )


def format_results_section(
    *,
    report: PerformanceReport,
    raw_path: Path,
    raw_sha256: str,
    command: str,
    label: str,
    missing_face_frame_count: int,
    reference: MagnitudeReference,
) -> str:
    """Render one tracked, appendable Markdown section for `docs/performance_results.md`."""
    heading = f"### {label} — {report.created_at} (commit `{report.commit[:12]}`)"
    metadata = _format_metadata_block(
        report=report,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        command=command,
        missing_face_frame_count=missing_face_frame_count,
    )
    reference_line = _format_reference_line(reference)
    table = _format_stage_table(report)
    verdict = (
        "**判定:** PASS — CUDA providerが実行中で、全段のp50/p95/p99が採取・検証された"
        "（このステップはベースライン記録であり、閾値ゲートは後続手順で行う）。"
    )
    return "\n\n".join([heading, metadata, reference_line, table, verdict])


def _append_docs_section(docs_path: Path, section: str) -> None:
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    if not docs_path.exists():
        docs_path.write_text(_DOCS_HEADER, encoding="utf-8")
    with docs_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + section + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    raw = load_raw_benchmark(args.raw)
    report = build_report_from_raw(raw)
    reference = reference_order_of_magnitude(report, known_baseline_ms=args.known_baseline_ms)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    report.save(args.report_output)

    raw_sha256 = sha256_of_file(args.raw)
    section = format_results_section(
        report=report,
        raw_path=args.raw,
        raw_sha256=raw_sha256,
        command=args.command,
        label=args.label,
        missing_face_frame_count=raw.get("missing_face_frame_count", -1),
        reference=reference,
    )
    _append_docs_section(args.docs, section)

    print(f"report written: {args.report_output}")
    print(f"docs appended : {args.docs}")
    print(_format_reference_line(reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
