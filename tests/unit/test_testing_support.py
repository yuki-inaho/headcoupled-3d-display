"""Child-process cleanup must hold even when the test body fails.

A browser end-to-end test that leaves uvicorn running keeps port 8000 bound and, in
live-camera runs, /dev/video0 open. The next run then fails for an unrelated reason, so
cleanup is verified here rather than assumed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from headcoupled_display.testing_support import managed_child, terminate_child


def spawn(script: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_terminate_child_stops_a_cooperative_process() -> None:
    process = spawn("import time; time.sleep(60)")
    terminate_child(process)
    assert process.poll() is not None


def test_terminate_child_escalates_to_kill_when_sigterm_is_ignored() -> None:
    """A server that swallows SIGTERM must still be gone when the helper returns."""

    process = spawn(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"

    terminate_child(process)
    assert process.poll() is not None


def test_terminate_child_is_idempotent_for_an_already_exited_process() -> None:
    process = spawn("raise SystemExit(0)")
    process.wait(timeout=10)
    terminate_child(process)
    assert process.poll() == 0


def test_managed_child_cleans_up_when_the_body_raises() -> None:
    process = spawn("import time; time.sleep(60)")
    with pytest.raises(RuntimeError, match="boom"), managed_child(process):
        raise RuntimeError("boom")
    assert process.poll() is not None


def test_managed_child_cleans_up_on_the_happy_path() -> None:
    process = spawn("import time; time.sleep(60)")
    with managed_child(process) as child:
        assert child.poll() is None
    assert process.poll() is not None
