"""Asynchronous runtime coordinator and latest-value WebSocket fan-out."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from .models import HardwareProfile, RuntimeStatus, TrackingState
from .tracking import TrackingProvider

#: How long the input may stall before subscribers are told tracking is unavailable.
#: Chosen just above two frame periods at the 30 fps target: long enough that a single
#: slow frame is not reported as a failure, short enough that a dead producer does not
#: leave a confident-looking pose on screen.
DEFAULT_STALE_AFTER_S = 0.5


class RuntimeCoordinator:
    """Runs the tracking provider and fans the latest value out to subscribers.

    Fan-out is keyed on a coordinator-owned *generation* counter rather than on
    ``TrackingState.sequence``. The sequence belongs to the provider and counts the
    frames it produced; the generation counts everything subscribers must see, including
    the synthetic "tracking is stale" states published when the provider stops producing.
    Reusing the sequence for both would either collide on recovery or make a stalled
    input indistinguishable from a still head.
    """

    def __init__(
        self,
        profile: HardwareProfile,
        provider_factory: Callable[[], TrackingProvider],
        *,
        target_fps: float = 30.0,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
    ) -> None:
        self.profile = profile
        self._provider_factory = provider_factory
        self._target_period = 1.0 / target_fps
        self._stale_after_s = stale_after_s
        self._provider: TrackingProvider | None = None
        self._task: asyncio.Task[None] | None = None
        self._condition = asyncio.Condition()
        self._state: TrackingState | None = None
        self._frame: bytes | None = None
        self._generation = 0
        self._frame_generation = 0
        self._last_error: str | None = None
        self._running = False
        self._stale = False

    @property
    def latest_state(self) -> TrackingState | None:
        return self._state

    @property
    def latest_frame(self) -> bytes | None:
        return self._frame

    @property
    def stale(self) -> bool:
        return self._stale

    async def start(self) -> None:
        if self._running:
            return
        self._provider = self._provider_factory()
        self._running = True
        self._task = asyncio.create_task(self._run(), name="headcoupled-tracking-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._provider is not None:
            await asyncio.to_thread(self._provider.close)
        self._provider = None
        self._task = None

    async def _run(self) -> None:
        assert self._provider is not None
        loop = asyncio.get_running_loop()
        last_success = loop.time()
        while self._running:
            started = loop.time()
            try:
                state, frame = await asyncio.to_thread(self._provider.sample)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - integration failures are surfaced
                self._last_error = f"{type(exc).__name__}: {exc}"
                if loop.time() - last_success >= self._stale_after_s:
                    await self._publish_stale()
                await asyncio.sleep(0.25)
            else:
                last_success = loop.time()
                await self._publish(state, frame)
            elapsed = loop.time() - started
            await asyncio.sleep(max(0.0, self._target_period - elapsed))

    async def _publish(self, state: TrackingState, frame: bytes) -> None:
        async with self._condition:
            self._generation += 1
            self._state = state
            self._frame = frame
            self._frame_generation = self._generation
            self._last_error = None
            self._stale = False
            self._condition.notify_all()

    async def _publish_stale(self) -> None:
        """Tell subscribers the input stopped, instead of leaving a confident pose up.

        Repeating the last good pose at its original confidence would render a dead
        producer as a perfectly still observer. The eye position is kept so the view does
        not jump, but confidence drops to zero and ``diagnostics.stale`` says why.
        """

        if self._state is None or self._stale:
            return
        async with self._condition:
            self._generation += 1
            self._stale = True
            self._state = self._state.model_copy(
                update={
                    "confidence": 0.0,
                    "stable": False,
                    "diagnostics": {
                        **self._state.diagnostics,
                        "stale": True,
                        "stale_after_s": self._stale_after_s,
                        "last_error": self._last_error,
                    },
                }
            )
            self._condition.notify_all()

    async def wait_for_state(
        self,
        after_generation: int,
        *,
        timeout_s: float = 3.0,
    ) -> tuple[int, TrackingState]:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: self._state is not None and self._generation > after_generation
                ),
                timeout=timeout_s,
            )
            assert self._state is not None
            return self._generation, self._state

    async def wait_for_frame(
        self,
        after_generation: int,
        *,
        timeout_s: float = 3.0,
    ) -> tuple[int, bytes]:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: self._frame is not None and self._frame_generation > after_generation
                ),
                timeout=timeout_s,
            )
            assert self._frame is not None
            return self._frame_generation, self._frame

    def status(self, source: str) -> RuntimeStatus:
        return RuntimeStatus(
            running=self._running,
            source=source,
            sequence=-1 if self._state is None else self._state.sequence,
            profile_id=self.profile.profile_id,
            provenance=self.profile.provenance,
            last_error=self._last_error,
        )
