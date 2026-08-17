"""A dead input must be reported as unavailable, not rendered as a perfectly still head.

Repeating the last good pose at its original confidence is the failure this guards
against: the display looks correct while the producer is gone.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from headcoupled_display.models import HardwareProfile, TrackingState
from headcoupled_display.runtime import RuntimeCoordinator

ROOT = Path(__file__).resolve().parents[2]


def make_state(sequence: int) -> TrackingState:
    return TrackingState(
        sequence=sequence,
        timestamp_unix_s=0.0,
        source="synthetic",
        confidence=1.0,
        cyclopean_eye_display_m=(0.01, 0.02, 0.6),
        left_eye_display_m=(-0.02, 0.02, 0.6),
        right_eye_display_m=(0.04, 0.02, 0.6),
        head_forward_display=(0.0, 0.0, -1.0),
        tracking_fps=30.0,
        inference_ms=5.0,
        stable=True,
        diagnostics={"face_count": 1},
    )


class ScriptedProvider:
    """Produces a few good samples, then fails the way a dead producer does."""

    def __init__(self, good_samples: int) -> None:
        self.good_samples = good_samples
        self.calls = 0
        self.closed = False

    def sample(self) -> tuple[TrackingState, bytes]:
        self.calls += 1
        if self.calls > self.good_samples:
            raise TimeoutError("no frame from the producer")
        return make_state(self.calls), b"\xff\xd8jpeg"

    def close(self) -> None:
        self.closed = True


def hardware() -> HardwareProfile:
    return HardwareProfile.load(ROOT / "config" / "hardware_profile.demo.json")


async def drive(good_samples: int, stale_after_s: float) -> list[TrackingState]:
    provider = ScriptedProvider(good_samples)
    coordinator = RuntimeCoordinator(
        hardware(),
        lambda: provider,
        target_fps=200.0,
        stale_after_s=stale_after_s,
    )
    received: list[TrackingState] = []
    await coordinator.start()
    try:
        generation = 0
        for _ in range(good_samples + 1):
            generation, state = await coordinator.wait_for_state(generation, timeout_s=5.0)
            received.append(state)
    finally:
        await coordinator.stop()
    return received


def test_stale_state_is_published_after_the_input_stops() -> None:
    received = asyncio.run(drive(good_samples=2, stale_after_s=0.0))

    assert [state.confidence for state in received[:2]] == [1.0, 1.0]
    stale = received[-1]
    assert stale.confidence == 0.0
    assert stale.stable is False
    assert stale.diagnostics["stale"] is True
    assert "TimeoutError" in str(stale.diagnostics["last_error"])


def test_stale_state_keeps_the_last_eye_position_so_the_view_does_not_jump() -> None:
    received = asyncio.run(drive(good_samples=2, stale_after_s=0.0))
    assert received[-1].cyclopean_eye_display_m == received[-2].cyclopean_eye_display_m


def test_stale_state_reuses_the_provider_sequence() -> None:
    """The sequence counts frames the provider made; a stall did not make one."""

    received = asyncio.run(drive(good_samples=2, stale_after_s=0.0))
    assert received[-1].sequence == received[-2].sequence


def test_generation_advances_even_though_the_sequence_did_not() -> None:
    async def run() -> tuple[int, int]:
        provider = ScriptedProvider(good_samples=1)
        coordinator = RuntimeCoordinator(
            hardware(), lambda: provider, target_fps=200.0, stale_after_s=0.0
        )
        await coordinator.start()
        try:
            first_generation, first = await coordinator.wait_for_state(0, timeout_s=5.0)
            second_generation, second = await coordinator.wait_for_state(
                first_generation, timeout_s=5.0
            )
            assert first.sequence == second.sequence
            return first_generation, second_generation
        finally:
            await coordinator.stop()

    first_generation, second_generation = asyncio.run(run())
    assert second_generation > first_generation


def test_a_stall_shorter_than_the_threshold_is_not_reported_as_stale() -> None:
    async def run() -> bool:
        provider = ScriptedProvider(good_samples=1)
        coordinator = RuntimeCoordinator(
            hardware(), lambda: provider, target_fps=200.0, stale_after_s=30.0
        )
        await coordinator.start()
        try:
            await coordinator.wait_for_state(0, timeout_s=5.0)
            await asyncio.sleep(0.2)
            return coordinator.stale
        finally:
            await coordinator.stop()

    assert asyncio.run(run()) is False
