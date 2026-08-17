"""RED/GREEN tests for the transport-candidate benchmark schema and adoption judgement
(workdoc step 31/32).

These tests exercise ``scripts.benchmark_transports``'s pydantic report schema and its
pure ``evaluate_candidate``/``evaluate_report`` judgement functions -- never the live
subprocess-based harness (that needs pyzmq/grpcio, only installed in the isolated
``.venv-transport-bench``, and takes real wall-clock time; it is exercised separately by
``scripts/benchmark_transports.py --smoke`` and the step-33 real run, per the workdoc).
Because pyzmq/grpcio are imported lazily only inside the candidate producer/consumer
functions, this file runs green under the product ``.venv`` with neither installed.
"""

from __future__ import annotations

import argparse

import pytest
from pydantic import ValidationError

import scripts.benchmark_transports as bench

CONDITION_KWARGS: dict = {
    "control_rate_hz": 60.0,
    "preview_rate_hz": 10.0,
    "control_message_count": 300,
    "preview_message_count": 50,
    "warmup_messages": 60,
    "consumer_stall_ms": 100.0,
    "control_packet_bytes": 146,
    "preview_packet_bytes_target": 25_000,
}


def _condition(**overrides: object) -> bench.BenchmarkCondition:
    kwargs = dict(CONDITION_KWARGS)
    kwargs.update(overrides)
    return bench.BenchmarkCondition(**kwargs)


def _run_kwargs(**overrides: object) -> dict:
    kwargs: dict = {
        "candidate": "zeromq",
        "run_index": 0,
        "package_name": "pyzmq",
        "package_version": "26.4.0",
        "control_rate_hz": 60.0,
        "preview_rate_hz": 10.0,
        "control_message_count": 300,
        "preview_message_count": 50,
        "warmup_messages": 60,
        "consumer_stall_ms": 100.0,
        "control_packet_bytes": 146,
        "producer_started_monotonic_ns": 1_000_000_000,
        "producer_finished_monotonic_ns": 1_005_000_000_000,
        "consumer_started_monotonic_ns": 999_000_000,
        "consumer_finished_monotonic_ns": 1_005_100_000_000,
        "sent_control_count": 300,
        "received_control_count": 250,
        "min_received_sequence": 0,
        "max_received_sequence": 299,
        "dropped_count": 50,
        "sequence_regressions": 0,
        "control_latency_p50_ms": 0.3,
        "control_latency_p95_ms": 0.9,
        "control_latency_p99_ms": 1.5,
        "max_age_ms": 20.0,
        "recovery_frames_after_stall": 1,
        "cpu_percent_producer": 12.0,
        "cpu_percent_consumer": 8.0,
        "bytes_sent_total": 500_000,
        "bytes_per_second": 100_000.0,
        "producer_max_enqueue_ms": 0.01,
        "producer_blocked": False,
    }
    kwargs.update(overrides)
    return kwargs


def _run(**overrides: object) -> bench.TransportRunResult:
    return bench.TransportRunResult(**_run_kwargs(**overrides))


def _passing_candidate_report(
    candidate: bench.CandidateName, run_count: int = 5
) -> bench.TransportCandidateReport:
    runs = tuple(_run(candidate=candidate, run_index=i) for i in range(run_count))
    return bench.TransportCandidateReport(candidate=candidate, dependency_available=True, runs=runs)


def _full_report(candidate_reports: dict) -> bench.TransportComparisonReport:
    return bench.TransportComparisonReport(
        condition=_condition(),
        candidates=tuple(candidate_reports[name] for name in bench.REQUIRED_CANDIDATES),
    )


# --- required fields / schema deficiencies ---------------------------------


def test_run_result_exposes_all_required_report_fields() -> None:
    run = _run()
    for field_name in (
        "control_latency_p50_ms",
        "control_latency_p95_ms",
        "control_latency_p99_ms",
        "cpu_percent_producer",
        "cpu_percent_consumer",
        "dropped_count",
        "max_age_ms",
        "bytes_per_second",
        "package_name",
        "package_version",
    ):
        assert hasattr(run, field_name)


@pytest.mark.parametrize(
    "missing_field",
    [
        "control_latency_p95_ms",
        "cpu_percent_producer",
        "dropped_count",
        "max_age_ms",
        "bytes_per_second",
        "package_version",
        "producer_started_monotonic_ns",
        "min_received_sequence",
    ],
)
def test_run_result_missing_required_field_is_schema_failure(missing_field: str) -> None:
    kwargs = _run_kwargs()
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        bench.TransportRunResult(**kwargs)


def test_run_result_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        bench.TransportRunResult(**_run_kwargs(unexpected_field="surprise"))


def test_run_result_rejects_percentiles_out_of_order() -> None:
    with pytest.raises(ValidationError, match="percentiles"):
        _run(control_latency_p50_ms=5.0, control_latency_p95_ms=1.0, control_latency_p99_ms=1.0)


# --- same-host monotonic timestamp + received sequence are mandatory -------


@pytest.mark.parametrize(
    "missing_field",
    [
        "producer_started_monotonic_ns",
        "producer_finished_monotonic_ns",
        "consumer_started_monotonic_ns",
        "consumer_finished_monotonic_ns",
    ],
)
def test_run_result_requires_monotonic_timestamps(missing_field: str) -> None:
    """Localhost wall clock alone must never be enough -- monotonic ns is mandatory."""

    kwargs = _run_kwargs()
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        bench.TransportRunResult(**kwargs)


@pytest.mark.parametrize("missing_field", ["min_received_sequence", "max_received_sequence"])
def test_run_result_requires_received_sequence(missing_field: str) -> None:
    kwargs = _run_kwargs()
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        bench.TransportRunResult(**kwargs)


def test_run_result_rejects_clock_domain_other_than_monotonic() -> None:
    with pytest.raises(ValidationError):
        _run(host_clock_domain="wall_clock_unix_ns")


def test_run_result_rejects_finished_before_started() -> None:
    with pytest.raises(ValidationError):
        _run(producer_finished_monotonic_ns=1, producer_started_monotonic_ns=2_000_000)


def test_run_result_rejects_max_sequence_below_min_sequence() -> None:
    with pytest.raises(ValidationError):
        _run(min_received_sequence=100, max_received_sequence=1, received_control_count=5)


# --- schema requires identical conditions across every candidate/run -------


def test_report_accepts_matching_conditions_across_all_candidates() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    report = _full_report(reports)
    assert len(report.candidates) == 4


def test_report_rejects_run_with_mismatched_message_count() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    bad_run = _run(candidate="grpc", control_message_count=999)
    reports["grpc"] = bench.TransportCandidateReport(
        candidate="grpc", dependency_available=True, runs=(bad_run,)
    )
    with pytest.raises(ValidationError, match="identical conditions"):
        _full_report(reports)


def test_report_rejects_run_with_mismatched_stall_ms() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    bad_run = _run(candidate="binary_http", consumer_stall_ms=5.0)
    reports["binary_http"] = bench.TransportCandidateReport(
        candidate="binary_http", dependency_available=True, runs=(bad_run,)
    )
    with pytest.raises(ValidationError, match="identical conditions"):
        _full_report(reports)


def test_report_rejects_run_with_mismatched_packet_bytes() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    bad_run = _run(candidate="json_http", control_packet_bytes=64)
    reports["json_http"] = bench.TransportCandidateReport(
        candidate="json_http", dependency_available=True, runs=(bad_run,)
    )
    with pytest.raises(ValidationError, match="identical conditions"):
        _full_report(reports)


# --- candidate-name integrity -----------------------------------------------


def test_report_rejects_duplicate_candidate_names() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    with pytest.raises(ValidationError, match="candidates must be exactly"):
        bench.TransportComparisonReport(
            condition=_condition(),
            candidates=(
                reports["json_http"],
                reports["json_http"],
                reports["zeromq"],
                reports["grpc"],
            ),
        )


def test_report_rejects_missing_candidate() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    with pytest.raises(ValidationError, match="candidates must be exactly"):
        bench.TransportComparisonReport(
            condition=_condition(),
            candidates=(reports["json_http"], reports["zeromq"], reports["grpc"]),
        )


def test_run_recorded_under_a_different_candidate_name_is_rejected() -> None:
    mismatched_run = _run(candidate="grpc")  # wrong name for a "zeromq" candidate report
    with pytest.raises(ValidationError, match="different candidate name"):
        bench.TransportCandidateReport(
            candidate="zeromq", dependency_available=True, runs=(mismatched_run,)
        )


# --- dependency-missing must be an explicit failure, never a silent skip ---


def test_candidate_report_allows_dependency_unavailable_with_no_runs() -> None:
    report = bench.TransportCandidateReport(
        candidate="grpc", dependency_available=False, failure_reason="grpcio is not installed"
    )
    assert report.runs == ()


def test_candidate_report_rejects_dependency_unavailable_without_a_reason() -> None:
    with pytest.raises(ValidationError, match="failure_reason"):
        bench.TransportCandidateReport(candidate="grpc", dependency_available=False)


def test_candidate_report_rejects_available_with_zero_runs() -> None:
    """A candidate must never silently claim success while reporting no measurements."""

    with pytest.raises(ValidationError, match="zero runs"):
        bench.TransportCandidateReport(candidate="zeromq", dependency_available=True, runs=())


def test_candidate_report_rejects_available_with_a_failure_reason() -> None:
    with pytest.raises(ValidationError):
        bench.TransportCandidateReport(
            candidate="zeromq",
            dependency_available=True,
            failure_reason="should not coexist with available=True",
            runs=(_run(candidate="zeromq"),),
        )


def test_evaluate_candidate_reports_missing_dependency_as_explicit_failure() -> None:
    report = bench.TransportCandidateReport(
        candidate="grpc", dependency_available=False, failure_reason="grpcio is not installed"
    )
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False
    assert verdict.dependency_available is False
    assert any("grpcio" in criterion.detail for criterion in verdict.criteria)


# --- stale-frame retention and reordering must fail the verdict ------------


def test_evaluate_candidate_fails_when_stale_frames_are_not_recovered() -> None:
    report = _passing_candidate_report("zeromq")
    report = report.model_copy(
        update={
            "runs": tuple(
                run.model_copy(update={"recovery_frames_after_stall": 5}) for run in report.runs
            )
        }
    )
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False
    recovery = next(
        c for c in verdict.criteria if c.name == "recovers_within_2_frames_after_overload"
    )
    assert recovery.passed is False


def test_evaluate_candidate_passes_recovery_within_two_frames() -> None:
    report = _passing_candidate_report("zeromq")
    report = report.model_copy(
        update={
            "runs": tuple(
                run.model_copy(update={"recovery_frames_after_stall": 2}) for run in report.runs
            )
        }
    )
    verdict = bench.evaluate_candidate(report)
    recovery = next(
        c for c in verdict.criteria if c.name == "recovers_within_2_frames_after_overload"
    )
    assert recovery.passed is True


def test_evaluate_candidate_fails_when_a_single_stale_frame_settles_the_recovery() -> None:
    """A transport that never drops (e.g. gRPC's HTTP/2 flow control just queues) can
    settle after a single receive -- a low burst count -- that is itself tens of ms
    stale. recovery_frames_after_stall alone must not be enough to pass; max_age_ms of
    the settled frame has to be within the 2-frame budget too."""

    report = _passing_candidate_report("grpc")
    runs = list(report.runs)
    runs[0] = runs[0].model_copy(update={"recovery_frames_after_stall": 1, "max_age_ms": 85.0})
    report = report.model_copy(update={"runs": tuple(runs)})
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False
    recovery = next(
        c for c in verdict.criteria if c.name == "recovers_within_2_frames_after_overload"
    )
    assert recovery.passed is False


def test_evaluate_candidate_fails_on_any_sequence_regression() -> None:
    report = _passing_candidate_report("grpc")
    runs = list(report.runs)
    runs[2] = runs[2].model_copy(update={"sequence_regressions": 1})
    report = report.model_copy(update={"runs": tuple(runs)})
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False
    regression = next(c for c in verdict.criteria if c.name == "zero_sequence_regressions")
    assert regression.passed is False


def test_evaluate_candidate_fails_when_producer_thread_blocks() -> None:
    report = _passing_candidate_report("binary_http")
    runs = list(report.runs)
    runs[0] = runs[0].model_copy(update={"producer_blocked": True, "producer_max_enqueue_ms": 42.0})
    report = report.model_copy(update={"runs": tuple(runs)})
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False
    blocked = next(c for c in verdict.criteria if c.name == "inference_thread_not_blocked_on_send")
    assert blocked.passed is False


def test_evaluate_candidate_fails_when_control_p95_exceeds_threshold() -> None:
    report = _passing_candidate_report("json_http")
    runs = list(report.runs)
    runs[0] = runs[0].model_copy(
        update={"control_latency_p95_ms": 5.0, "control_latency_p99_ms": 6.0}
    )
    report = report.model_copy(update={"runs": tuple(runs)})
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False
    p95 = next(c for c in verdict.criteria if c.name == "control_p95_le_2ms")
    assert p95.passed is False


def test_evaluate_candidate_all_criteria_pass_for_a_clean_report() -> None:
    report = _passing_candidate_report("zeromq")
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is True
    assert all(criterion.passed for criterion in verdict.criteria)


# --- adoption must never be decided by the mean alone -----------------------


def test_evaluate_candidate_fails_when_a_single_run_violates_p95_even_though_the_mean_passes() -> (
    None
):
    """4 runs at 1.0ms p95 and 1 run at 2.5ms p95 average to 1.3ms (<= 2ms) -- but the
    worst run alone must sink the candidate; the mean must never be what is judged."""

    good_kwargs = {"control_latency_p95_ms": 1.0, "control_latency_p99_ms": 1.2}
    runs = [_run(candidate="zeromq", run_index=i, **good_kwargs) for i in range(4)]
    runs.append(
        _run(
            candidate="zeromq", run_index=4, control_latency_p95_ms=2.5, control_latency_p99_ms=3.0
        )
    )
    mean_p95 = sum(r.control_latency_p95_ms for r in runs) / len(runs)
    assert mean_p95 <= bench.CONTROL_P95_THRESHOLD_MS, "test setup must keep the mean passing"

    report = bench.TransportCandidateReport(
        candidate="zeromq", dependency_available=True, runs=tuple(runs)
    )
    verdict = bench.evaluate_candidate(report)
    assert verdict.passed is False


# --- report-level adoption verdict: never implicitly adopt the least-bad ---


def test_evaluate_report_selects_the_single_passing_candidate() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    slow_runs = tuple(
        run.model_copy(update={"control_latency_p95_ms": 9.0, "control_latency_p99_ms": 9.5})
        for run in reports["json_http"].runs
    )
    reports["json_http"] = reports["json_http"].model_copy(update={"runs": slow_runs})
    for name in ("binary_http", "zeromq"):
        blocked_runs = tuple(
            run.model_copy(update={"producer_blocked": True}) for run in reports[name].runs
        )
        reports[name] = reports[name].model_copy(update={"runs": blocked_runs})

    verdict = bench.evaluate_report(_full_report(reports))
    assert verdict.selected_candidate == "grpc"


def test_evaluate_report_does_not_implicitly_adopt_the_least_bad_candidate_when_all_fail() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    for name in bench.REQUIRED_CANDIDATES:
        slow_runs = tuple(
            run.model_copy(update={"control_latency_p95_ms": 9.0, "control_latency_p99_ms": 9.5})
            for run in reports[name].runs
        )
        reports[name] = reports[name].model_copy(update={"runs": slow_runs})

    verdict = bench.evaluate_report(_full_report(reports))
    assert verdict.selected_candidate is None
    assert all(not v.passed for v in verdict.verdicts)
    assert (
        "least-bad" in verdict.summary
        or "no candidate met every adoption criterion" in verdict.summary
    )


def test_evaluate_report_does_not_auto_select_when_multiple_candidates_pass() -> None:
    reports = {name: _passing_candidate_report(name) for name in bench.REQUIRED_CANDIDATES}
    verdict = bench.evaluate_report(_full_report(reports))
    assert verdict.selected_candidate is None
    assert sum(1 for v in verdict.verdicts if v.passed) >= 2


# --- recovery-frame computation (pure, consumer-local) ----------------------


def test_compute_recovery_frames_is_zero_for_steady_state_arrivals() -> None:
    control_rate_hz = 60.0
    gap_ns = int(1e9 / control_rate_hz)
    control = [
        bench._ReceivedControl(
            sequence=i, receive_monotonic_ns=i * gap_ns, send_monotonic_ns=i * gap_ns
        )
        for i in range(10)
    ]
    frames = bench._compute_recovery_frames(
        control, stall_sequence=5, control_rate_hz=control_rate_hz
    )
    assert frames <= 1


def test_compute_recovery_frames_counts_a_backlog_drain_burst() -> None:
    """A naive FIFO queue delivers the whole 100ms backlog back-to-back before the
    normal ~16.7ms gap reappears -- that burst length is what "recovery_frames" counts.
    """

    control_rate_hz = 60.0
    gap_ns = int(1e9 / control_rate_hz)
    control = [
        bench._ReceivedControl(
            sequence=i, receive_monotonic_ns=i * gap_ns, send_monotonic_ns=i * gap_ns
        )
        for i in range(5)
    ]
    # Stall starts at sequence 5; six backlogged messages (5..10) arrive ~0ns apart
    # (queued during the 100ms stall), then normal spacing resumes from sequence 11.
    burst_start_ns = 5 * gap_ns
    for offset, sequence in enumerate(range(5, 11)):
        control.append(
            bench._ReceivedControl(
                sequence=sequence,
                receive_monotonic_ns=burst_start_ns + offset * 1000,
                send_monotonic_ns=sequence * gap_ns,
            )
        )
    control.append(
        bench._ReceivedControl(
            sequence=11,
            receive_monotonic_ns=burst_start_ns + 6000 + gap_ns,
            send_monotonic_ns=11 * gap_ns,
        )
    )
    frames = bench._compute_recovery_frames(
        control, stall_sequence=5, control_rate_hz=control_rate_hz
    )
    assert frames >= 5


def test_compute_sequence_regressions_counts_out_of_order_arrivals() -> None:
    control = [
        bench._ReceivedControl(sequence=0, receive_monotonic_ns=0, send_monotonic_ns=0),
        bench._ReceivedControl(sequence=5, receive_monotonic_ns=1, send_monotonic_ns=1),
        bench._ReceivedControl(
            sequence=3, receive_monotonic_ns=2, send_monotonic_ns=2
        ),  # regression
        bench._ReceivedControl(sequence=6, receive_monotonic_ns=3, send_monotonic_ns=3),
    ]
    assert bench._compute_sequence_regressions(control) == 1


# --- control/binary payload encoding round trips ----------------------------


def test_encode_decode_control_binary_round_trips_sequence_and_send_time() -> None:
    payload = bench._encode_control(sequence=42, send_monotonic_ns=123_456_789, as_json=False)
    assert len(payload) == bench.CONTROL_PACKET_SIZE
    decoded = bench._decode_control(payload, as_json=False)
    assert decoded == (42, 123_456_789)


def test_encode_decode_control_json_round_trips_sequence_and_send_time() -> None:
    payload = bench._encode_control(sequence=7, send_monotonic_ns=999, as_json=True)
    decoded = bench._decode_control(payload, as_json=True)
    assert decoded == (7, 999)


def test_decode_control_returns_none_for_malformed_payload() -> None:
    assert bench._decode_control(b"not a valid packet", as_json=False) is None
    assert bench._decode_control(b"{not json", as_json=True) is None


# --- measurement isolation (host-noise removal, never a threshold/scoring change) --
#
# These only exercise the pure argv/condition-building helpers behind --producer-cpus/
# --consumer-cpus/--gc-disable/--warmup-messages -- never the live subprocess harness
# (same constraint as the rest of this file; see module docstring).


def test_no_isolation_default_adds_no_taskset_prefix_or_role_args() -> None:
    isolation = bench._NO_ISOLATION
    assert isolation.taskset_prefix(isolation.producer_cpus) == []
    assert isolation.taskset_prefix(isolation.consumer_cpus) == []
    assert isolation.role_args() == []


def test_isolation_options_taskset_prefix_is_empty_when_cpus_not_given() -> None:
    isolation = bench._IsolationOptions()
    assert isolation.taskset_prefix(None) == []


def test_isolation_options_taskset_prefix_wraps_cpu_list_with_taskset_dash_c() -> None:
    isolation = bench._IsolationOptions(producer_cpus="1,2", consumer_cpus="4,5")
    assert isolation.taskset_prefix(isolation.producer_cpus) == ["taskset", "-c", "1,2"]
    assert isolation.taskset_prefix(isolation.consumer_cpus) == ["taskset", "-c", "4,5"]


def test_isolation_options_role_args_passes_gc_disable_flag_through() -> None:
    assert bench._IsolationOptions(gc_disable=False).role_args() == []
    assert bench._IsolationOptions(gc_disable=True).role_args() == ["--gc-disable"]


def test_isolate_measurement_process_disables_gc_only_when_requested(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(bench.gc, "disable", lambda: calls.append("disabled"))

    bench._isolate_measurement_process(argparse.Namespace(gc_disable=False))
    assert calls == []

    bench._isolate_measurement_process(argparse.Namespace(gc_disable=True))
    assert calls == ["disabled"]


def test_resolve_condition_leaves_warmup_untouched_when_not_overridden() -> None:
    args = argparse.Namespace(smoke=False, warmup_messages=None)
    condition = bench._resolve_condition(args)
    assert condition.warmup_messages == bench._DEFAULT_CONDITION.warmup_messages


def test_resolve_condition_overrides_warmup_without_changing_other_fields() -> None:
    args = argparse.Namespace(smoke=False, warmup_messages=90)
    condition = bench._resolve_condition(args)
    assert condition.warmup_messages == 90
    assert condition.control_rate_hz == bench._DEFAULT_CONDITION.control_rate_hz
    assert condition.control_message_count == bench._DEFAULT_CONDITION.control_message_count


def test_resolve_condition_picks_smoke_condition_when_smoke_flag_set() -> None:
    args = argparse.Namespace(smoke=True, warmup_messages=None)
    condition = bench._resolve_condition(args)
    assert condition == bench._SMOKE_CONDITION


def test_cli_parser_defaults_leave_isolation_off() -> None:
    args = bench._build_arg_parser().parse_args([])
    assert args.producer_cpus is None
    assert args.consumer_cpus is None
    assert args.gc_disable is False
    assert args.warmup_messages is None


def test_cli_parser_accepts_isolation_flags() -> None:
    args = bench._build_arg_parser().parse_args(
        [
            "--producer-cpus",
            "1,2",
            "--consumer-cpus",
            "4,5",
            "--gc-disable",
            "--warmup-messages",
            "90",
        ]
    )
    assert args.producer_cpus == "1,2"
    assert args.consumer_cpus == "4,5"
    assert args.gc_disable is True
    assert args.warmup_messages == 90
