"""Compares transport candidates for the control/preview IPC split (workdoc step 32/33).

Candidates under comparison: ``json_http`` (today's approach: JSON + base64 JPEG over a
synchronous HTTP POST), ``binary_http`` (same HTTP transport, but control uses
``headcoupled_display.protocol``'s fixed binary packet and preview is raw JPEG bytes),
``zeromq`` (pyzmq), and ``grpc`` (grpcio). pyzmq/grpcio are NOT product runtime
dependencies -- see requirements.transport-bench.in. This module is importable (for its
pydantic report schema and the pure ``evaluate_*`` judgement functions) without either
package installed: both are imported lazily, only inside the specific candidate's
producer/consumer functions, so ``tests/unit/test_transport_benchmark.py`` runs green
under the product ``.venv`` even though that venv never installs pyzmq/grpcio.

Architecture (identical across all four candidates, so the comparison isolates the
*transport*, not "did we bother writing a non-blocking sender"):

* One "inference simulator" thread writes control samples at ``control_rate_hz`` and
  preview samples at ``preview_rate_hz`` into a single-slot :class:`_LatestMailbox` per
  lane. ``mailbox.put()`` is an O(1) lock+overwrite and is asserted never to block for
  more than a few milliseconds (``producer_max_enqueue_ms`` / ``producer_blocked``) --
  this is exactly what workdoc step 36 requires of the eventual production control lane
  regardless of which transport wins here.
* One background "sender" thread per lane repeatedly pops the *current* mailbox content
  and hands it to the candidate's transport. Because the mailbox always holds only the
  newest value, a sender that was blocked mid-send (e.g. a stalled HTTP response) always
  transmits the freshest sample once it becomes free again -- backlog is never queued at
  the application layer. What differs between candidates is how much each transport's own
  plumbing helps or hurts that property under a stalled consumer (HTTP request/response
  coupling; ZeroMQ's ``ZMQ_CONFLATE``; gRPC's HTTP/2 flow control).

Clock domain: producer and consumer are separate OS processes on the *same host*. Their
control-lane latency is computed as ``consumer_receive_ns - producer_send_ns`` using
``time.perf_counter_ns()`` on both ends. On Linux CPython, ``perf_counter_ns`` is
``clock_gettime(CLOCK_MONOTONIC)``, a single system-wide clock -- NOT the per-process
counter ``headcoupled_display.protocol``'s docstring warns about for the *shipped wire
protocol* (which must stay portable off this one dev host). This benchmark is a
same-host, Linux-only, throw-away measurement tool, so that portability constraint does
not apply, and using the shared host clock directly (rather than an RTT/2 approximation)
is exactly the "same-host monotonic timestamp" comparison workdoc step 31 asks for.

ZeroMQ note: ``ZMQ_CONFLATE`` cannot be combined with multipart messages, so control and
preview each get their own single-part socket (never multipart) -- see
``_zeromq_producer``/``_zeromq_consumer``.

gRPC note: there is no bundled ``.proto``/generated stub in this repository (this script
is the only new file grpc-related code may live in for this step). Both the client and
server use grpc's low-level *generic handler* API with an identity (de)serializer, so
messages travel as the same raw bytes every other candidate uses -- no protobuf schema.
Control and preview are two independent client-streaming RPCs multiplexed over one
long-lived channel, per workdoc step 32/33's requirement to reuse the channel and to
check whether HTTP/2 flow control lets stale frames accumulate.

Measurement isolation: this dev host runs an unrelated desktop session (browser, window
manager, other agent processes) that competes for the same CPUs, so an occasional
control_latency_p95_ms outlier can come from host scheduling noise rather than the
transport itself. ``--producer-cpus``/``--consumer-cpus`` (``taskset -c``) pin the two
subprocesses to specific cores so they are not fighting each other -- or busy background
processes -- for a core; ``--gc-disable`` turns off the cyclic GC inside each short-lived
producer/consumer subprocess so a GC pause can never land between capturing a timestamp
and using it; ``_http_producer``/``_grpc_producer`` eagerly establish their
connection/channel before the timed loop starts rather than relying on
``warmup_messages`` alone to absorb connection-setup cost; ``--warmup-messages`` lets that
absorption window be widened further. None of this changes what is measured (the
producer/consumer code paths and message content are identical) or how a run is judged
(``CONTROL_P95_THRESHOLD_MS`` and the worst-run logic in ``_criterion_control_p95`` are
untouched) -- it only removes noise sources that are not the transport under test.
"""

from __future__ import annotations

import argparse
import gc
import http.client
import http.server
import json
import os
import platform
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from headcoupled_display.models import StrictModel, utc_now_iso
from headcoupled_display.performance import compute_stage_percentiles
from headcoupled_display.protocol import (
    NUM_LANDMARKS,
    ControlPacket,
    ProtocolError,
    decode_control_packet,
    encode_control_packet,
)
from headcoupled_display.protocol import PACKET_SIZE as CONTROL_PACKET_SIZE

CandidateName = Literal["json_http", "binary_http", "zeromq", "grpc"]
REQUIRED_CANDIDATES: tuple[CandidateName, ...] = ("json_http", "binary_http", "zeromq", "grpc")

# Adoption thresholds (workdoc step 33). Judged per run, worst-case across the 5 runs --
# never averaged (an average could hide one bad run passing off the back of four good ones).
CONTROL_P95_THRESHOLD_MS = 2.0
MAX_RECOVERY_FRAMES = 2
# mailbox.put() should always take low microseconds; anything above this indicates the
# "inference thread" is actually blocking on something transport-related.
PRODUCER_BLOCK_THRESHOLD_MS = 5.0

_PREVIEW_HEADER = struct.Struct("!IQ")  # sequence (uint32), send_monotonic_ns (uint64)


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class BenchmarkCondition(StrictModel):
    """Parameters shared identically across every candidate/run in one comparison.

    A single shared instance (not a per-candidate copy) is the source of truth; every
    :class:`TransportRunResult` echoes these values back and
    :meth:`TransportComparisonReport.validate_candidates` rejects any run whose echo
    disagrees, so a bug that measured one candidate under different conditions fails
    validation instead of silently producing a lopsided comparison.
    """

    control_rate_hz: float = Field(gt=0)
    preview_rate_hz: float = Field(gt=0)
    control_message_count: int = Field(gt=0)
    preview_message_count: int = Field(gt=0)
    warmup_messages: int = Field(ge=0)
    consumer_stall_ms: float = Field(gt=0)
    control_packet_bytes: int = Field(gt=0)
    preview_packet_bytes_target: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_warmup(self) -> BenchmarkCondition:
        if self.warmup_messages >= self.control_message_count:
            raise ValueError("warmup_messages must be less than control_message_count")
        return self

    def stall_sequence(self) -> int:
        """Control sequence number whose consumer-side handling triggers the one stall."""

        return self.warmup_messages + (self.control_message_count - self.warmup_messages) // 2


class TransportRunResult(StrictModel):
    """One benchmark run (one candidate, one repetition) between two isolated processes."""

    schema_version: int = 1
    candidate: CandidateName
    run_index: int = Field(ge=0)
    package_name: str = Field(min_length=1)
    package_version: str = Field(min_length=1)

    # Condition echo -- must match the report-level BenchmarkCondition exactly.
    control_rate_hz: float = Field(gt=0)
    preview_rate_hz: float = Field(gt=0)
    control_message_count: int = Field(gt=0)
    preview_message_count: int = Field(gt=0)
    warmup_messages: int = Field(ge=0)
    consumer_stall_ms: float = Field(gt=0)
    control_packet_bytes: int = Field(gt=0)

    # Same-host monotonic timing, both ends -- never wall clock alone (workdoc step 31).
    host_clock_domain: Literal["monotonic_ns"] = "monotonic_ns"
    producer_started_monotonic_ns: int = Field(ge=0)
    producer_finished_monotonic_ns: int = Field(ge=0)
    consumer_started_monotonic_ns: int = Field(ge=0)
    consumer_finished_monotonic_ns: int = Field(ge=0)

    # Received sequence -- required so staleness/reordering are judged on sequence
    # numbers, not on wall-clock deltas.
    sent_control_count: int = Field(ge=0)
    received_control_count: int = Field(ge=0)
    min_received_sequence: int = Field(ge=0)
    max_received_sequence: int = Field(ge=0)
    dropped_count: int = Field(ge=0)
    sequence_regressions: int = Field(ge=0)

    control_latency_p50_ms: float = Field(ge=0)
    control_latency_p95_ms: float = Field(ge=0)
    control_latency_p99_ms: float = Field(ge=0)

    max_age_ms: float = Field(ge=0)
    recovery_frames_after_stall: int = Field(ge=0)

    cpu_percent_producer: float = Field(ge=0)
    cpu_percent_consumer: float = Field(ge=0)
    bytes_sent_total: int = Field(ge=0)
    bytes_per_second: float = Field(ge=0)

    producer_max_enqueue_ms: float = Field(ge=0)
    producer_blocked: bool

    @model_validator(mode="after")
    def validate_consistency(self) -> TransportRunResult:
        if self.producer_finished_monotonic_ns < self.producer_started_monotonic_ns:
            raise ValueError(
                "producer_finished_monotonic_ns precedes producer_started_monotonic_ns"
            )
        if self.consumer_finished_monotonic_ns < self.consumer_started_monotonic_ns:
            raise ValueError(
                "consumer_finished_monotonic_ns precedes consumer_started_monotonic_ns"
            )
        if (
            self.received_control_count > 0
            and self.max_received_sequence < self.min_received_sequence
        ):
            raise ValueError("max_received_sequence must not be less than min_received_sequence")
        if not (
            self.control_latency_p50_ms
            <= self.control_latency_p95_ms
            <= self.control_latency_p99_ms
        ):
            raise ValueError("control latency percentiles must satisfy p50 <= p95 <= p99")
        return self


class TransportCandidateReport(StrictModel):
    """All runs for one candidate, or an explicit, non-silent dependency failure."""

    candidate: CandidateName
    dependency_available: bool
    failure_reason: str | None = None
    runs: tuple[TransportRunResult, ...] = ()

    @model_validator(mode="after")
    def validate_availability(self) -> TransportCandidateReport:
        if self.dependency_available:
            if not self.runs:
                raise ValueError(
                    f"{self.candidate}: dependency_available=True but zero runs were recorded -- "
                    "a candidate must never silently report success with no measurements"
                )
            if self.failure_reason is not None:
                raise ValueError(f"{self.candidate}: failure_reason must be null when available")
            if any(run.candidate != self.candidate for run in self.runs):
                raise ValueError(
                    f"{self.candidate}: a run is recorded under a different candidate name"
                )
        else:
            if self.runs:
                raise ValueError(f"{self.candidate}: dependency_available=False must carry no runs")
            if not self.failure_reason:
                raise ValueError(
                    f"{self.candidate}: dependency_available=False requires a non-empty failure_reason "
                    "(missing dependencies must be reported as an explicit failure, never a silent skip)"
                )
        return self


class TransportComparisonReport(StrictModel):
    """The full step-33 comparison: one shared condition, exactly the four candidates."""

    schema_version: int = 1
    created_at: str = Field(default_factory=utc_now_iso)
    condition: BenchmarkCondition
    candidates: tuple[TransportCandidateReport, ...]

    @model_validator(mode="after")
    def validate_candidates(self) -> TransportComparisonReport:
        names = [c.candidate for c in self.candidates]
        if len(names) != len(REQUIRED_CANDIDATES) or set(names) != set(REQUIRED_CANDIDATES):
            raise ValueError(
                f"candidates must be exactly {REQUIRED_CANDIDATES}, got {tuple(names)}"
            )
        for candidate_report in self.candidates:
            for run in candidate_report.runs:
                self._validate_run_matches_condition(candidate_report.candidate, run)
        return self

    def _validate_run_matches_condition(self, candidate: str, run: TransportRunResult) -> None:
        mismatched = (
            run.control_rate_hz != self.condition.control_rate_hz
            or run.preview_rate_hz != self.condition.preview_rate_hz
            or run.control_message_count != self.condition.control_message_count
            or run.preview_message_count != self.condition.preview_message_count
            or run.warmup_messages != self.condition.warmup_messages
            or run.consumer_stall_ms != self.condition.consumer_stall_ms
            or run.control_packet_bytes != self.condition.control_packet_bytes
        )
        if mismatched:
            raise ValueError(
                f"{candidate} run {run.run_index}: its condition echo does not match the "
                "report-wide BenchmarkCondition -- every candidate must be measured "
                "under identical conditions"
            )


class CriterionResult(StrictModel):
    name: str
    passed: bool
    detail: str


class CandidateVerdict(StrictModel):
    candidate: CandidateName
    dependency_available: bool
    passed: bool
    criteria: tuple[CriterionResult, ...]


class ComparisonVerdict(StrictModel):
    verdicts: tuple[CandidateVerdict, ...]
    selected_candidate: CandidateName | None
    summary: str


def _criterion_control_p95(runs: Sequence[TransportRunResult]) -> CriterionResult:
    worst = max(run.control_latency_p95_ms for run in runs)
    ok_count = sum(1 for run in runs if run.control_latency_p95_ms <= CONTROL_P95_THRESHOLD_MS)
    return CriterionResult(
        name="control_p95_le_2ms",
        passed=worst <= CONTROL_P95_THRESHOLD_MS,
        detail=(
            f"{ok_count}/{len(runs)} runs had control p95 <= {CONTROL_P95_THRESHOLD_MS} ms "
            f"(worst observed {worst:.3f} ms)"
        ),
    )


def _max_age_budget_ms(control_rate_hz: float) -> float:
    """2 control-frame periods, in ms -- how stale "caught up" is allowed to be."""

    return MAX_RECOVERY_FRAMES * 1000.0 / control_rate_hz


def _criterion_recovery(runs: Sequence[TransportRunResult]) -> CriterionResult:
    """Both must hold: few receives drained the backlog, AND what settled is fresh.

    ``recovery_frames_after_stall`` alone is not sufficient: a transport that never
    drops (e.g. gRPC's HTTP/2 flow control, which just queues) can settle after a
    single receive that is itself tens of milliseconds stale -- a low burst count with
    high staleness is exactly the "recovered but with an old frame" failure workdoc
    step 33 asks this criterion to catch, so ``max_age_ms`` is checked too.
    """

    worst_frames = max(run.recovery_frames_after_stall for run in runs)
    worst_age_ratio = max(run.max_age_ms / _max_age_budget_ms(run.control_rate_hz) for run in runs)
    passed = worst_frames <= MAX_RECOVERY_FRAMES and worst_age_ratio <= 1.0
    return CriterionResult(
        name="recovers_within_2_frames_after_overload",
        passed=passed,
        detail=(
            f"worst recovery across {len(runs)} runs: {worst_frames} frames "
            f"(limit {MAX_RECOVERY_FRAMES}); worst settled max_age was "
            f"{worst_age_ratio * 100:.0f}% of the {MAX_RECOVERY_FRAMES}-frame budget"
        ),
    )


def _criterion_producer_not_blocked(runs: Sequence[TransportRunResult]) -> CriterionResult:
    blocked_runs = [run.run_index for run in runs if run.producer_blocked]
    return CriterionResult(
        name="inference_thread_not_blocked_on_send",
        passed=not blocked_runs,
        detail=(
            "the simulated inference thread never blocked on a send"
            if not blocked_runs
            else f"producer blocked in runs {blocked_runs}"
        ),
    )


def _criterion_no_sequence_regression(runs: Sequence[TransportRunResult]) -> CriterionResult:
    total = sum(run.sequence_regressions for run in runs)
    return CriterionResult(
        name="zero_sequence_regressions",
        passed=total == 0,
        detail=f"total sequence regressions across {len(runs)} runs: {total}",
    )


def evaluate_candidate(report: TransportCandidateReport) -> CandidateVerdict:
    """Machine-judge one candidate's adoption criteria -- worst-case per run, never a mean."""

    if not report.dependency_available:
        criterion = CriterionResult(
            name="dependency_available",
            passed=False,
            detail=report.failure_reason or "dependency unavailable",
        )
        return CandidateVerdict(
            candidate=report.candidate,
            dependency_available=False,
            passed=False,
            criteria=(criterion,),
        )

    criteria = (
        _criterion_control_p95(report.runs),
        _criterion_recovery(report.runs),
        _criterion_producer_not_blocked(report.runs),
        _criterion_no_sequence_regression(report.runs),
    )
    return CandidateVerdict(
        candidate=report.candidate,
        dependency_available=True,
        passed=all(criterion.passed for criterion in criteria),
        criteria=criteria,
    )


def evaluate_report(report: TransportComparisonReport) -> ComparisonVerdict:
    """Judge every candidate; never fall back to "least bad" when nothing passes."""

    verdicts = tuple(evaluate_candidate(candidate) for candidate in report.candidates)
    passing = [verdict.candidate for verdict in verdicts if verdict.passed]
    if not passing:
        return ComparisonVerdict(
            verdicts=verdicts,
            selected_candidate=None,
            summary="no candidate met every adoption criterion; do not implicitly adopt the least-bad option",
        )
    if len(passing) == 1:
        return ComparisonVerdict(
            verdicts=verdicts,
            selected_candidate=passing[0],
            summary=f"{passing[0]} is the only candidate meeting every criterion",
        )
    return ComparisonVerdict(
        verdicts=verdicts,
        selected_candidate=None,
        summary=f"multiple candidates passed ({passing}); a manual tie-break is required",
    )


# ---------------------------------------------------------------------------
# Shared producer/consumer primitives (transport-agnostic)
# ---------------------------------------------------------------------------


class _LatestMailbox:
    """Single-slot mailbox: newer ``put`` overwrites older, ``take`` returns and clears.

    ``put`` must stay O(1) -- it stands in for the real inference thread handing a fresh
    pose to the IPC layer, which workdoc step 36 requires to never block on a send.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: tuple[int, bytes] | None = None

    def put(self, sequence: int, payload: bytes) -> None:
        with self._lock:
            self._item = (sequence, payload)

    def take(self) -> tuple[int, bytes] | None:
        with self._lock:
            item = self._item
            self._item = None
            return item


@dataclass
class _ReceivedControl:
    sequence: int
    receive_monotonic_ns: int
    send_monotonic_ns: int


@dataclass
class _ConsumerState:
    """Thread-safe accumulator filled in by a candidate's consumer implementation."""

    control: list[_ReceivedControl] = field(default_factory=list)
    preview_bytes_received: int = 0
    preview_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_control(self, sequence: int, send_monotonic_ns: int) -> None:
        with self._lock:
            self.control.append(
                _ReceivedControl(
                    sequence=sequence,
                    receive_monotonic_ns=time.perf_counter_ns(),
                    send_monotonic_ns=send_monotonic_ns,
                )
            )

    def record_preview(self, payload_len: int) -> None:
        with self._lock:
            self.preview_count += 1
            self.preview_bytes_received += payload_len


def _maybe_stall(
    sequence: int, stall_sequence: int, stall_ms: float, already_stalled: list[bool]
) -> None:
    """Sleep once, the first time ``sequence`` reaches ``stall_sequence``.

    Models a consumer thread that is momentarily too busy to read/respond -- the one
    deliberate overload event every candidate is measured against (workdoc step 33).
    """

    if sequence >= stall_sequence and not already_stalled[0]:
        already_stalled[0] = True
        time.sleep(stall_ms / 1000.0)


def _received_final_message(state: _ConsumerState, condition: BenchmarkCondition) -> bool:
    """True once the very last control sequence the producer ever enqueues has arrived.

    The mailbox/conflate pattern means many *earlier* sequence numbers are legitimately
    superseded and never sent at all, so ``received_control_count`` reaching
    ``control_message_count`` is not a safe stop condition -- it would rarely happen and
    every consumer would just sit out its full timeout on every run. The final sequence
    number, in contrast, is never superseded (nothing is enqueued after it), so once it
    has been observed the run is genuinely complete.
    """

    target = condition.control_message_count - 1
    return any(item.sequence == target for item in state.control)


def _encode_control(sequence: int, send_monotonic_ns: int, *, as_json: bool) -> bytes:
    """Build a control-lane payload carrying 12 landmarks, a score, and the send time.

    Landmark/score content is fixed and synthetic -- transport comparison only cares
    about the wire size and encoding, not the pose itself.
    """

    landmarks = tuple((100.0 + index, 200.0 + index) for index in range(NUM_LANDMARKS))
    if as_json:
        payload = {
            "sequence": sequence,
            "capture_monotonic_ns": send_monotonic_ns,
            "score": 0.95,
            "landmarks": [[u, v] for u, v in landmarks],
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")
    packet = ControlPacket(
        landmarks_px=landmarks,
        score=0.95,
        sequence=sequence,
        capture_monotonic_ns=send_monotonic_ns,
        capture_unix_ns=time.time_ns(),
        inference_monotonic_ns=send_monotonic_ns,
        inference_unix_ns=time.time_ns(),
    )
    return encode_control_packet(packet)


def _decode_control(data: bytes, *, as_json: bool) -> tuple[int, int] | None:
    """Return ``(sequence, send_monotonic_ns)``, or ``None`` for a malformed payload."""

    try:
        if as_json:
            payload = json.loads(data)
            return int(payload["sequence"]), int(payload["capture_monotonic_ns"])
        packet = decode_control_packet(data)
        return packet.sequence, packet.capture_monotonic_ns
    except (ProtocolError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _make_preview_filler(target_bytes: int) -> bytes:
    padding = max(target_bytes - _PREVIEW_HEADER.size, 0)
    return bytes(padding)


def _encode_preview(sequence: int, send_monotonic_ns: int, filler: bytes) -> bytes:
    return _PREVIEW_HEADER.pack(sequence & 0xFFFFFFFF, send_monotonic_ns) + filler


def _decode_preview_header(data: bytes) -> tuple[int, int] | None:
    if len(data) < _PREVIEW_HEADER.size:
        return None
    sequence, send_monotonic_ns = _PREVIEW_HEADER.unpack_from(data)
    return sequence, send_monotonic_ns


def _cpu_time_seconds() -> float:
    usage = os.times()
    return usage.user + usage.system


@dataclass
class _RunCondition:
    """Resolved, per-process view of :class:`BenchmarkCondition` plus wiring."""

    condition: BenchmarkCondition
    host: str
    control_port: int
    preview_port: int
    run_index: int


def _run_inference_loop(
    condition: BenchmarkCondition,
    control_mailbox: _LatestMailbox,
    preview_mailbox: _LatestMailbox,
    *,
    as_json: bool,
) -> dict[str, Any]:
    """Paces synthetic control/preview samples straight into the two mailboxes.

    Stands in for the real inference thread. Every candidate's producer creates its own
    pair of mailboxes and its own transport-appropriate sender/generator reading from
    them (see ``_generic_sender_loop`` for the HTTP/ZeroMQ shape, ``_grpc_producer`` for
    the streaming-generator shape) -- this loop never talks to a transport directly, so
    ``producer_max_enqueue_ms`` reflects only mailbox contention, never network or peer
    behaviour, for every candidate alike.
    """

    enqueue_durations_ms: list[float] = []
    filler = _make_preview_filler(condition.preview_packet_bytes_target)

    started_ns = time.perf_counter_ns()
    control_period_s = 1.0 / condition.control_rate_hz
    next_preview_index = 0
    for sequence in range(condition.control_message_count):
        target_time = started_ns / 1e9 + sequence * control_period_s
        _sleep_until(target_time)

        send_ns = time.perf_counter_ns()
        enqueue_start = time.perf_counter_ns()
        control_mailbox.put(sequence, _encode_control(sequence, send_ns, as_json=as_json))
        enqueue_durations_ms.append((time.perf_counter_ns() - enqueue_start) / 1e6)

        elapsed_s = time.perf_counter_ns() / 1e9 - started_ns / 1e9
        expected_preview_index = int(elapsed_s * condition.preview_rate_hz)
        while (
            next_preview_index <= expected_preview_index
            and next_preview_index < condition.preview_message_count
        ):
            preview_send_ns = time.perf_counter_ns()
            preview_mailbox.put(
                next_preview_index, _encode_preview(next_preview_index, preview_send_ns, filler)
            )
            next_preview_index += 1
    finished_ns = time.perf_counter_ns()

    return {
        "producer_started_monotonic_ns": started_ns,
        "producer_finished_monotonic_ns": finished_ns,
        "sent_control_count": condition.control_message_count,
        "producer_max_enqueue_ms": max(enqueue_durations_ms) if enqueue_durations_ms else 0.0,
        "producer_blocked": bool(enqueue_durations_ms)
        and max(enqueue_durations_ms) > PRODUCER_BLOCK_THRESHOLD_MS,
        "cpu_time_seconds": _cpu_time_seconds(),
    }


def _generic_sender_loop(
    mailbox: _LatestMailbox,
    stop_event: threading.Event,
    send: Callable[[bytes], None],
    bytes_counter: list[int],
    lock: threading.Lock,
) -> None:
    """Pop-and-send loop shared by the HTTP and ZeroMQ producers.

    ``send`` may block on the network (a stalled HTTP response, a full ZMQ send
    buffer); by construction that only ever delays *this* thread, never the inference
    loop, and the next iteration always sends whatever is currently newest in the
    mailbox rather than a queued backlog.
    """

    while True:
        item = mailbox.take()
        if item is None:
            if stop_event.is_set():
                return
            time.sleep(0.0005)
            continue
        _, payload = item
        send(payload)
        with lock:
            bytes_counter[0] += len(payload)


def _sleep_until(target_perf_counter_s: float) -> None:
    remaining = target_perf_counter_s - time.perf_counter_ns() / 1e9
    if remaining > 0:
        time.sleep(remaining)


def _compute_recovery_frames(
    control: Sequence[_ReceivedControl], stall_sequence: int, control_rate_hz: float
) -> int:
    """Count how many receives after the stall arrived "too fast" (draining a backlog).

    Purely consumer-local: uses only the consumer's own receive timestamps, so it needs
    no cross-process clock assumption. A conflate/latest-wins transport settles back to
    the steady ~1/control_rate_hz inter-arrival gap within one receive; a naive FIFO
    queue instead delivers the whole backlog back-to-back before that gap reappears.
    """

    ordered = sorted(control, key=lambda item: item.sequence)
    start_index = next(
        (i for i, item in enumerate(ordered) if item.sequence >= stall_sequence), None
    )
    if start_index is None or start_index + 1 >= len(ordered):
        return 0
    normal_gap_ns = (1.0 / control_rate_hz) * 1e9
    frames = 0
    for i in range(start_index, len(ordered) - 1):
        gap = ordered[i + 1].receive_monotonic_ns - ordered[i].receive_monotonic_ns
        frames += 1
        if gap >= 0.5 * normal_gap_ns:
            break
    return frames


def _compute_max_age_ms(
    control: Sequence[_ReceivedControl], stall_sequence: int, recovery_frames: int
) -> float:
    ordered = sorted(control, key=lambda item: item.sequence)
    start_index = next(
        (i for i, item in enumerate(ordered) if item.sequence >= stall_sequence), None
    )
    if start_index is None:
        return 0.0
    settle_index = min(start_index + recovery_frames, len(ordered) - 1)
    settled = ordered[settle_index]
    return max(0.0, (settled.receive_monotonic_ns - settled.send_monotonic_ns) / 1e6)


def _compute_sequence_regressions(control: Sequence[_ReceivedControl]) -> int:
    regressions = 0
    last_seen = -1
    for item in control:  # arrival order, not sorted -- this is what "regression" means
        if item.sequence < last_seen:
            regressions += 1
        last_seen = max(last_seen, item.sequence)
    return regressions


def _build_run_result(
    *,
    candidate: CandidateName,
    run_index: int,
    package_name: str,
    package_version: str,
    condition: BenchmarkCondition,
    producer: dict[str, Any],
    consumer: dict[str, Any],
    state: _ConsumerState,
) -> TransportRunResult:
    if not state.control:
        raise RuntimeError(
            f"{candidate}: zero control messages were received -- treating this as an "
            "implementation defect (e.g. connection refused), not a measured result"
        )

    non_warmup = [item for item in state.control if item.sequence >= condition.warmup_messages]
    samples_ms = [
        max(0.0, (item.receive_monotonic_ns - item.send_monotonic_ns) / 1e6) for item in non_warmup
    ]
    percentiles = compute_stage_percentiles(samples_ms)

    stall_sequence = condition.stall_sequence()
    recovery_frames = _compute_recovery_frames(
        state.control, stall_sequence, condition.control_rate_hz
    )
    max_age_ms = _compute_max_age_ms(state.control, stall_sequence, recovery_frames)
    sequence_regressions = _compute_sequence_regressions(state.control)

    sequences = [item.sequence for item in state.control]
    wall_seconds = max(
        (producer["producer_finished_monotonic_ns"] - producer["producer_started_monotonic_ns"])
        / 1e9,
        1e-9,
    )

    return TransportRunResult(
        candidate=candidate,
        run_index=run_index,
        package_name=package_name,
        package_version=package_version,
        control_rate_hz=condition.control_rate_hz,
        preview_rate_hz=condition.preview_rate_hz,
        control_message_count=condition.control_message_count,
        preview_message_count=condition.preview_message_count,
        warmup_messages=condition.warmup_messages,
        consumer_stall_ms=condition.consumer_stall_ms,
        control_packet_bytes=condition.control_packet_bytes,
        producer_started_monotonic_ns=producer["producer_started_monotonic_ns"],
        producer_finished_monotonic_ns=producer["producer_finished_monotonic_ns"],
        consumer_started_monotonic_ns=consumer["consumer_started_monotonic_ns"],
        consumer_finished_monotonic_ns=consumer["consumer_finished_monotonic_ns"],
        sent_control_count=producer["sent_control_count"],
        received_control_count=len(state.control),
        min_received_sequence=min(sequences),
        max_received_sequence=max(sequences),
        dropped_count=max(0, producer["sent_control_count"] - len(state.control)),
        sequence_regressions=sequence_regressions,
        control_latency_p50_ms=percentiles.p50_ms,
        control_latency_p95_ms=percentiles.p95_ms,
        control_latency_p99_ms=percentiles.p99_ms,
        max_age_ms=max_age_ms,
        recovery_frames_after_stall=recovery_frames,
        cpu_percent_producer=100.0 * producer["cpu_time_seconds"] / wall_seconds,
        cpu_percent_consumer=100.0 * consumer["cpu_time_seconds"] / wall_seconds,
        bytes_sent_total=producer["bytes_sent_total"],
        bytes_per_second=producer["bytes_sent_total"] / wall_seconds,
        producer_max_enqueue_ms=producer["producer_max_enqueue_ms"],
        producer_blocked=producer["producer_blocked"],
    )


# ---------------------------------------------------------------------------
# json_http / binary_http (stdlib http.client + http.server; symmetric with
# scripts/facemesh_ipc_producer.py, which already avoids requests/httpx)
# ---------------------------------------------------------------------------


def _http_producer(run: _RunCondition, *, as_json: bool) -> dict[str, Any]:
    control_conn = http.client.HTTPConnection(run.host, run.control_port, timeout=5.0)
    preview_conn = http.client.HTTPConnection(run.host, run.preview_port, timeout=5.0)
    # Establish both TCP connections before the timed inference loop starts (isolation
    # measure: without this, http.client lazily connects on the first request(), so a
    # connection-setup stall could otherwise land on an early *measured* message rather
    # than being absorbed by warmup_messages alone).
    control_conn.connect()
    preview_conn.connect()

    def post(conn: http.client.HTTPConnection, path: str, body: bytes) -> None:
        conn.request("POST", path, body=body, headers={"Content-Length": str(len(body))})
        response = conn.getresponse()
        response.read()

    control_mailbox = _LatestMailbox()
    preview_mailbox = _LatestMailbox()
    stop_event = threading.Event()
    bytes_sent = [0]
    lock = threading.Lock()
    control_thread = threading.Thread(
        target=_generic_sender_loop,
        args=(
            control_mailbox,
            stop_event,
            lambda body: post(control_conn, "/control", body),
            bytes_sent,
            lock,
        ),
    )
    preview_thread = threading.Thread(
        target=_generic_sender_loop,
        args=(
            preview_mailbox,
            stop_event,
            lambda body: post(preview_conn, "/preview", body),
            bytes_sent,
            lock,
        ),
    )
    control_thread.start()
    preview_thread.start()

    result = _run_inference_loop(run.condition, control_mailbox, preview_mailbox, as_json=as_json)

    stop_event.set()
    control_thread.join()
    preview_thread.join()
    control_conn.close()
    preview_conn.close()
    result["bytes_sent_total"] = bytes_sent[0]
    return result


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _make_http_consumer_handler(
    state: _ConsumerState,
    stall_sequence: int,
    stall_ms: float,
    already_stalled: list[bool],
    as_json: bool,
    is_control: bool,
) -> type[_QuietHandler]:
    class Handler(_QuietHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if is_control:
                decoded = _decode_control(body, as_json=as_json)
                if decoded is not None:
                    sequence, send_ns = decoded
                    _maybe_stall(sequence, stall_sequence, stall_ms, already_stalled)
                    state.record_control(sequence, send_ns)
            else:
                state.record_preview(len(body))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def _http_consumer(run: _RunCondition, *, as_json: bool) -> tuple[dict[str, Any], _ConsumerState]:
    condition = run.condition
    state = _ConsumerState()
    already_stalled = [False]
    stall_sequence = condition.stall_sequence()

    control_server = http.server.ThreadingHTTPServer(
        (run.host, run.control_port),
        _make_http_consumer_handler(
            state, stall_sequence, condition.consumer_stall_ms, already_stalled, as_json, True
        ),
    )
    preview_server = http.server.ThreadingHTTPServer(
        (run.host, run.preview_port),
        _make_http_consumer_handler(
            state, stall_sequence, condition.consumer_stall_ms, already_stalled, as_json, False
        ),
    )
    control_thread = threading.Thread(
        target=control_server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    preview_thread = threading.Thread(
        target=preview_server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    control_thread.start()
    preview_thread.start()
    print("READY", flush=True)

    started_ns = time.perf_counter_ns()
    deadline = time.monotonic() + _consumer_timeout_s(condition)
    while not _received_final_message(state, condition) and time.monotonic() < deadline:
        time.sleep(0.01)
    finished_ns = time.perf_counter_ns()

    control_server.shutdown()
    preview_server.shutdown()
    control_thread.join()
    preview_thread.join()

    return {
        "consumer_started_monotonic_ns": started_ns,
        "consumer_finished_monotonic_ns": finished_ns,
        "cpu_time_seconds": _cpu_time_seconds(),
    }, state


def _consumer_timeout_s(condition: BenchmarkCondition) -> float:
    return (
        condition.control_message_count / condition.control_rate_hz
        + condition.consumer_stall_ms / 1000.0 * 5
        + 10.0
    )


# ---------------------------------------------------------------------------
# zeromq
# ---------------------------------------------------------------------------


def _zeromq_dependency() -> tuple[str, str]:
    import zmq

    return "pyzmq", zmq.pyzmq_version()


def _zeromq_producer(run: _RunCondition) -> dict[str, Any]:
    import zmq

    context = zmq.Context.instance()
    control_socket = context.socket(zmq.PUSH)
    control_socket.setsockopt(zmq.SNDHWM, 1)
    control_socket.setsockopt(zmq.CONFLATE, 1)  # single-part payload only, per module docstring
    control_socket.connect(f"tcp://{run.host}:{run.control_port}")

    preview_socket = context.socket(zmq.PUSH)
    preview_socket.setsockopt(zmq.SNDHWM, 1)
    preview_socket.setsockopt(zmq.CONFLATE, 1)
    preview_socket.connect(f"tcp://{run.host}:{run.preview_port}")

    control_mailbox = _LatestMailbox()
    preview_mailbox = _LatestMailbox()
    stop_event = threading.Event()
    bytes_sent = [0]
    lock = threading.Lock()
    control_thread = threading.Thread(
        target=_generic_sender_loop,
        args=(control_mailbox, stop_event, control_socket.send, bytes_sent, lock),
    )
    preview_thread = threading.Thread(
        target=_generic_sender_loop,
        args=(preview_mailbox, stop_event, preview_socket.send, bytes_sent, lock),
    )
    control_thread.start()
    preview_thread.start()

    result = _run_inference_loop(run.condition, control_mailbox, preview_mailbox, as_json=False)

    stop_event.set()
    control_thread.join()
    preview_thread.join()
    control_socket.close(0)
    preview_socket.close(0)
    result["bytes_sent_total"] = bytes_sent[0]
    return result


def _zeromq_consumer(run: _RunCondition) -> tuple[dict[str, Any], _ConsumerState]:
    import zmq

    condition = run.condition
    state = _ConsumerState()
    already_stalled = [False]
    stall_sequence = condition.stall_sequence()

    context = zmq.Context.instance()
    control_socket = context.socket(zmq.PULL)
    control_socket.setsockopt(zmq.RCVHWM, 1)
    control_socket.setsockopt(zmq.CONFLATE, 1)
    control_socket.bind(f"tcp://{run.host}:{run.control_port}")

    preview_socket = context.socket(zmq.PULL)
    preview_socket.setsockopt(zmq.RCVHWM, 1)
    preview_socket.bind(f"tcp://{run.host}:{run.preview_port}")

    poller = zmq.Poller()
    poller.register(control_socket, zmq.POLLIN)
    poller.register(preview_socket, zmq.POLLIN)

    print("READY", flush=True)
    started_ns = time.perf_counter_ns()
    deadline = time.monotonic() + _consumer_timeout_s(condition)

    while not _received_final_message(state, condition) and time.monotonic() < deadline:
        events = dict(poller.poll(timeout=50))
        if control_socket in events:
            data = control_socket.recv()
            decoded = _decode_control(data, as_json=False)
            if decoded is not None:
                sequence, send_ns = decoded
                state.record_control(sequence, send_ns)
                _maybe_stall(sequence, stall_sequence, condition.consumer_stall_ms, already_stalled)
        if preview_socket in events:
            data = preview_socket.recv()
            state.record_preview(len(data))

    finished_ns = time.perf_counter_ns()
    control_socket.close(0)
    preview_socket.close(0)

    return {
        "consumer_started_monotonic_ns": started_ns,
        "consumer_finished_monotonic_ns": finished_ns,
        "cpu_time_seconds": _cpu_time_seconds(),
    }, state


# ---------------------------------------------------------------------------
# grpc -- raw-bytes generic handlers, no .proto/codegen (only this file may be new)
# ---------------------------------------------------------------------------

_GRPC_SERVICE = "headcoupled.bench.Transport"
_GRPC_CONTROL_METHOD = f"/{_GRPC_SERVICE}/PushControl"
_GRPC_PREVIEW_METHOD = f"/{_GRPC_SERVICE}/PushPreview"


def _grpc_dependency() -> tuple[str, str]:
    import grpc

    return "grpcio", grpc.__version__


def _identity(data: bytes) -> bytes:
    return data


def _grpc_producer(run: _RunCondition) -> dict[str, Any]:
    """Client-streaming control/preview over one long-lived channel.

    The generator fed to each ``stream_unary`` call reads directly from the SAME
    mailbox the inference loop writes to (no relay/second mailbox in between): if
    HTTP/2 flow control ever blocks a ``yield`` here because the consumer fell behind,
    that is visible exactly the way it would be for any other candidate -- the next
    successful yield simply carries whatever is freshest in the mailbox at that moment,
    with no intermediate buffering hiding or amplifying the effect.
    """

    import grpc

    channel = grpc.insecure_channel(f"{run.host}:{run.control_port}")
    # Block until the HTTP/2 connection is actually up (isolation measure, mirrors the
    # explicit TCP connect() in _http_producer): without this, the handshake happens
    # lazily on the first stream_unary call and could otherwise land on an early
    # *measured* message rather than being absorbed by warmup_messages alone.
    grpc.channel_ready_future(channel).result(timeout=10.0)
    control_mailbox = _LatestMailbox()
    preview_mailbox = _LatestMailbox()
    stop_event = threading.Event()
    bytes_sent = [0]
    lock = threading.Lock()

    def request_stream(mailbox: _LatestMailbox) -> Any:
        while True:
            item = mailbox.take()
            if item is None:
                if stop_event.is_set():
                    return
                time.sleep(0.0005)
                continue
            payload = item[1]
            with lock:
                bytes_sent[0] += len(payload)
            yield payload

    control_call = channel.stream_unary(
        _GRPC_CONTROL_METHOD, request_serializer=_identity, response_deserializer=_identity
    )
    preview_call = channel.stream_unary(
        _GRPC_PREVIEW_METHOD, request_serializer=_identity, response_deserializer=_identity
    )
    control_thread = threading.Thread(target=lambda: control_call(request_stream(control_mailbox)))
    preview_thread = threading.Thread(target=lambda: preview_call(request_stream(preview_mailbox)))
    control_thread.start()
    preview_thread.start()

    result = _run_inference_loop(run.condition, control_mailbox, preview_mailbox, as_json=False)

    stop_event.set()
    control_thread.join(timeout=10.0)
    preview_thread.join(timeout=10.0)
    channel.close()
    result["bytes_sent_total"] = bytes_sent[0]
    return result


def _grpc_consumer(run: _RunCondition) -> tuple[dict[str, Any], _ConsumerState]:
    import grpc

    condition = run.condition
    state = _ConsumerState()
    already_stalled = [False]
    stall_sequence = condition.stall_sequence()
    done = threading.Event()

    def handle_control(request_iterator: Any, context: Any) -> bytes:
        for data in request_iterator:
            decoded = _decode_control(data, as_json=False)
            if decoded is not None:
                sequence, send_ns = decoded
                state.record_control(sequence, send_ns)
                _maybe_stall(sequence, stall_sequence, condition.consumer_stall_ms, already_stalled)
            if _received_final_message(state, condition):
                done.set()
        return b"ack"

    def handle_preview(request_iterator: Any, context: Any) -> bytes:
        for data in request_iterator:
            state.record_preview(len(data))
        return b"ack"

    handlers = {
        "PushControl": grpc.stream_unary_rpc_method_handler(
            handle_control, request_deserializer=_identity, response_serializer=_identity
        ),
        "PushPreview": grpc.stream_unary_rpc_method_handler(
            handle_preview, request_deserializer=_identity, response_serializer=_identity
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(_GRPC_SERVICE, handlers)

    from concurrent import futures

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_insecure_port(f"{run.host}:{run.control_port}")
    server.start()
    print("READY", flush=True)

    started_ns = time.perf_counter_ns()
    done.wait(timeout=_consumer_timeout_s(condition))
    finished_ns = time.perf_counter_ns()

    server.stop(grace=1.0)

    return {
        "consumer_started_monotonic_ns": started_ns,
        "consumer_finished_monotonic_ns": finished_ns,
        "cpu_time_seconds": _cpu_time_seconds(),
    }, state


# ---------------------------------------------------------------------------
# Candidate registry + role dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CandidateImpl:
    dependency: Callable[[], tuple[str, str]]
    producer: Callable[[_RunCondition], dict[str, Any]]
    consumer: Callable[[_RunCondition], tuple[dict[str, Any], _ConsumerState]]
    single_port: bool = False  # grpc multiplexes control+preview over one channel/port


_CANDIDATES: dict[CandidateName, _CandidateImpl] = {
    "json_http": _CandidateImpl(
        dependency=lambda: ("http.client/http.server (stdlib)", platform.python_version()),
        producer=lambda run: _http_producer(run, as_json=True),
        consumer=lambda run: _http_consumer(run, as_json=True),
    ),
    "binary_http": _CandidateImpl(
        dependency=lambda: ("http.client/http.server (stdlib)", platform.python_version()),
        producer=lambda run: _http_producer(run, as_json=False),
        consumer=lambda run: _http_consumer(run, as_json=False),
    ),
    "zeromq": _CandidateImpl(
        dependency=_zeromq_dependency, producer=_zeromq_producer, consumer=_zeromq_consumer
    ),
    "grpc": _CandidateImpl(
        dependency=_grpc_dependency,
        producer=_grpc_producer,
        consumer=_grpc_consumer,
        single_port=True,
    ),
}


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _isolate_measurement_process(args: argparse.Namespace) -> None:
    """Strip GC pauses from a producer/consumer subprocess's own timing (isolation
    measure only -- never touches product code, see module docstring "Measurement
    isolation" section). Each role process is short-lived and exits right after
    writing its result file, so there is no long-run leak risk from disabling GC."""

    if args.gc_disable:
        gc.disable()


def _run_role_consumer(args: argparse.Namespace) -> None:
    _isolate_measurement_process(args)
    impl = _CANDIDATES[args.candidate]
    try:
        package_name, package_version = impl.dependency()
    except ImportError as error:
        print(f"DEPENDENCY_MISSING {error}", flush=True)
        return

    condition = BenchmarkCondition.model_validate_json(Path(args.condition_file).read_text())
    run = _RunCondition(
        condition=condition,
        host=args.host,
        control_port=args.control_port,
        preview_port=args.preview_port,
        run_index=args.run_index,
    )
    consumer_result, state = impl.consumer(run)
    Path(args.result_file).write_text(
        json.dumps(
            {
                "package_name": package_name,
                "package_version": package_version,
                "consumer": consumer_result,
                "control": [
                    {
                        "sequence": item.sequence,
                        "receive_monotonic_ns": item.receive_monotonic_ns,
                        "send_monotonic_ns": item.send_monotonic_ns,
                    }
                    for item in state.control
                ],
            }
        )
    )


def _run_role_producer(args: argparse.Namespace) -> None:
    _isolate_measurement_process(args)
    impl = _CANDIDATES[args.candidate]
    condition = BenchmarkCondition.model_validate_json(Path(args.condition_file).read_text())
    run = _RunCondition(
        condition=condition,
        host=args.host,
        control_port=args.control_port,
        preview_port=args.preview_port,
        run_index=args.run_index,
    )
    producer_result = impl.producer(run)
    Path(args.result_file).write_text(json.dumps(producer_result))


@dataclass(frozen=True)
class _IsolationOptions:
    """Opt-in, benchmark-only noise-removal knobs (see module docstring "Measurement
    isolation"). Every field defaults to "do nothing", so a bare CLI invocation (and
    every existing recipe/test) behaves exactly as before this option was added."""

    producer_cpus: str | None = None
    consumer_cpus: str | None = None
    gc_disable: bool = False

    def taskset_prefix(self, cpus: str | None) -> list[str]:
        return ["taskset", "-c", cpus] if cpus else []

    def role_args(self) -> list[str]:
        return ["--gc-disable"] if self.gc_disable else []


_NO_ISOLATION = _IsolationOptions()


def _run_one_candidate_run(
    candidate: CandidateName,
    condition: BenchmarkCondition,
    run_index: int,
    host: str,
    isolation: _IsolationOptions = _NO_ISOLATION,
) -> TransportRunResult | None:
    """Spawn isolated consumer+producer subprocesses for one run; None on missing dependency."""

    impl = _CANDIDATES[candidate]
    control_port = _free_port()
    preview_port = control_port if impl.single_port else _free_port()

    import tempfile

    with tempfile.TemporaryDirectory(prefix="transport-bench-") as tmp:
        condition_file = Path(tmp) / "condition.json"
        condition_file.write_text(condition.model_dump_json())
        consumer_result_file = Path(tmp) / "consumer.json"
        producer_result_file = Path(tmp) / "producer.json"

        base_args = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--candidate",
            candidate,
            "--host",
            host,
            "--control-port",
            str(control_port),
            "--preview-port",
            str(preview_port),
            "--run-index",
            str(run_index),
            "--condition-file",
            str(condition_file),
            *isolation.role_args(),
        ]
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}

        consumer_proc = subprocess.Popen(
            [
                *isolation.taskset_prefix(isolation.consumer_cpus),
                *base_args,
                "--role",
                "consumer",
                "--result-file",
                str(consumer_result_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        ready_line = consumer_proc.stdout.readline() if consumer_proc.stdout else ""
        if ready_line.startswith("DEPENDENCY_MISSING"):
            consumer_proc.wait(timeout=5.0)
            return None

        producer_proc = subprocess.run(
            [
                *isolation.taskset_prefix(isolation.producer_cpus),
                *base_args,
                "--role",
                "producer",
                "--result-file",
                str(producer_result_file),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=_consumer_timeout_s(condition),
        )
        if producer_proc.returncode != 0:
            raise RuntimeError(f"{candidate} producer failed: {producer_proc.stderr}")

        consumer_proc.wait(timeout=_consumer_timeout_s(condition))
        if consumer_proc.returncode != 0:
            stderr = consumer_proc.stderr.read() if consumer_proc.stderr else ""
            raise RuntimeError(f"{candidate} consumer failed: {stderr}")

        producer_result = json.loads(producer_result_file.read_text())
        consumer_payload = json.loads(consumer_result_file.read_text())

    state = _ConsumerState()
    for item in consumer_payload["control"]:
        state.control.append(
            _ReceivedControl(
                sequence=item["sequence"],
                receive_monotonic_ns=item["receive_monotonic_ns"],
                send_monotonic_ns=item["send_monotonic_ns"],
            )
        )
    return _build_run_result(
        candidate=candidate,
        run_index=run_index,
        package_name=consumer_payload["package_name"],
        package_version=consumer_payload["package_version"],
        condition=condition,
        producer=producer_result,
        consumer=consumer_payload["consumer"],
        state=state,
    )


def run_candidate_benchmark(
    candidate: CandidateName,
    condition: BenchmarkCondition,
    runs: int,
    host: str = "127.0.0.1",
    isolation: _IsolationOptions = _NO_ISOLATION,
) -> TransportCandidateReport:
    first_run = _run_one_candidate_run(candidate, condition, 0, host, isolation)
    if first_run is None:
        impl = _CANDIDATES[candidate]
        try:
            impl.dependency()
            reason = "dependency import succeeded in-process but failed in the isolated subprocess"
        except ImportError as error:
            reason = str(error)
        return TransportCandidateReport(
            candidate=candidate, dependency_available=False, failure_reason=reason
        )

    collected = [first_run]
    for run_index in range(1, runs):
        collected.append(_run_one_candidate_run(candidate, condition, run_index, host, isolation))
    return TransportCandidateReport(
        candidate=candidate, dependency_available=True, runs=tuple(collected)
    )


def run_full_benchmark(
    condition: BenchmarkCondition,
    runs: int,
    host: str = "127.0.0.1",
    candidates: Sequence[CandidateName] = REQUIRED_CANDIDATES,
    isolation: _IsolationOptions = _NO_ISOLATION,
) -> TransportComparisonReport:
    reports = [
        run_candidate_benchmark(candidate, condition, runs, host, isolation)
        for candidate in candidates
    ]
    return TransportComparisonReport(condition=condition, candidates=tuple(reports))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_CONDITION = BenchmarkCondition(
    control_rate_hz=60.0,
    preview_rate_hz=10.0,
    control_message_count=300,
    preview_message_count=50,
    warmup_messages=60,
    consumer_stall_ms=100.0,
    control_packet_bytes=CONTROL_PACKET_SIZE,
    preview_packet_bytes_target=25_000,
)

_SMOKE_CONDITION = BenchmarkCondition(
    control_rate_hz=60.0,
    preview_rate_hz=10.0,
    control_message_count=100,
    preview_message_count=17,
    warmup_messages=10,
    consumer_stall_ms=100.0,
    control_packet_bytes=CONTROL_PACKET_SIZE,
    preview_packet_bytes_target=25_000,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("producer", "consumer"), default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--candidate", choices=REQUIRED_CANDIDATES, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--preview-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--condition-file", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--candidates", default=",".join(REQUIRED_CANDIDATES))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--smoke", action="store_true", help="100-message schema-validation smoke run"
    )
    parser.add_argument("--output", default="artifacts/perf/transport_comparison.json")
    # Measurement isolation (opt-in; see module docstring "Measurement isolation").
    # None of these change producer/consumer behaviour or the pass/fail thresholds --
    # they only remove host-scheduling/GC/connection-setup noise from what gets timed.
    parser.add_argument(
        "--producer-cpus",
        default=None,
        help="taskset -c CPU list to pin the producer subprocess to, e.g. '1,2'",
    )
    parser.add_argument(
        "--consumer-cpus",
        default=None,
        help="taskset -c CPU list to pin the consumer subprocess to, e.g. '3,4'",
    )
    parser.add_argument(
        "--gc-disable",
        action="store_true",
        help="disable the cyclic GC inside the producer/consumer subprocesses",
    )
    parser.add_argument(
        "--warmup-messages",
        type=int,
        default=None,
        help="override the condition's warmup_messages (percentile window is unaffected)",
    )
    return parser


def _resolve_condition(args: argparse.Namespace) -> BenchmarkCondition:
    condition = _SMOKE_CONDITION if args.smoke else _DEFAULT_CONDITION
    if args.warmup_messages is not None:
        condition = condition.model_copy(update={"warmup_messages": args.warmup_messages})
    return condition


def _print_verdict(verdict: ComparisonVerdict) -> None:
    for candidate_verdict in verdict.verdicts:
        status = "PASS" if candidate_verdict.passed else "FAIL"
        print(f"[{status}] {candidate_verdict.candidate}")
        for criterion in candidate_verdict.criteria:
            mark = "ok" if criterion.passed else "NG"
            print(f"    ({mark}) {criterion.name}: {criterion.detail}")
    print(verdict.summary)


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.role == "consumer":
        _run_role_consumer(args)
        return
    if args.role == "producer":
        _run_role_producer(args)
        return

    condition = _resolve_condition(args)
    runs = 1 if args.smoke else args.runs
    candidates = tuple(name.strip() for name in args.candidates.split(","))  # type: ignore[assignment]
    isolation = _IsolationOptions(
        producer_cpus=args.producer_cpus,
        consumer_cpus=args.consumer_cpus,
        gc_disable=args.gc_disable,
    )

    report = run_full_benchmark(condition, runs, args.host, candidates, isolation)  # type: ignore[arg-type]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n")

    _print_verdict(evaluate_report(report))


if __name__ == "__main__":
    main()
