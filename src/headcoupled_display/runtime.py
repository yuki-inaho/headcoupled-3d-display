"""Asynchronous runtime coordinator and latest-value WebSocket fan-out."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from .models import HardwareProfile, RuntimeStatus, TrackingState
from .tracking import TrackingProvider


class RuntimeCoordinator:
    def __init__(
        self,
        profile: HardwareProfile,
        provider_factory: Callable[[], TrackingProvider],
        *,
        target_fps: float = 30.0,
    ) -> None:
        self.profile = profile
        self._provider_factory = provider_factory
        self._target_period = 1.0 / target_fps
        self._provider: TrackingProvider | None = None
        self._task: asyncio.Task[None] | None = None
        self._condition = asyncio.Condition()
        self._state: TrackingState | None = None
        self._frame: bytes | None = None
        self._frame_sequence = -1
        self._last_error: str | None = None
        self._running = False

    @property
    def latest_state(self) -> TrackingState | None:
        return self._state

    @property
    def latest_frame(self) -> bytes | None:
        return self._frame

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
        while self._running:
            started = loop.time()
            try:
                state, frame = await asyncio.to_thread(self._provider.sample)
                async with self._condition:
                    self._state = state
                    self._frame = frame
                    self._frame_sequence = state.sequence
                    self._last_error = None
                    self._condition.notify_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - integration failures are surfaced
                self._last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.25)
            elapsed = loop.time() - started
            await asyncio.sleep(max(0.0, self._target_period - elapsed))

    async def wait_for_state(
        self,
        after_sequence: int,
        *,
        timeout_s: float = 3.0,
    ) -> TrackingState:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: self._state is not None and self._state.sequence > after_sequence
                ),
                timeout=timeout_s,
            )
            assert self._state is not None
            return self._state

    async def wait_for_frame(
        self,
        after_sequence: int,
        *,
        timeout_s: float = 3.0,
    ) -> tuple[int, bytes]:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: self._frame is not None and self._frame_sequence > after_sequence
                ),
                timeout=timeout_s,
            )
            assert self._frame is not None
            return self._frame_sequence, self._frame

    def status(self, source: str) -> RuntimeStatus:
        return RuntimeStatus(
            running=self._running,
            source=source,
            sequence=-1 if self._state is None else self._state.sequence,
            profile_id=self.profile.profile_id,
            provenance=self.profile.provenance,
            last_error=self._last_error,
        )
