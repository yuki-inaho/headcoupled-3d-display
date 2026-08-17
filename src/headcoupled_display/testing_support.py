"""Test-only helpers: Chromium policy relaxation and guaranteed child-process cleanup."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Seconds to wait for a child to exit after SIGTERM before escalating to SIGKILL.
TERMINATE_GRACE_S = 5.0
#: Seconds to wait after SIGKILL before giving up and reporting the survivor.
KILL_GRACE_S = 5.0


def _read_blocking_policy(path: Path) -> tuple[bytes, dict[str, object]] | None:
    try:
        original = path.read_bytes()
        policy = json.loads(original)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(policy, dict):
        return None
    blocklist = policy.get("URLBlocklist")
    if not isinstance(blocklist, list) or "*" not in blocklist:
        return None
    return original, policy


def _allow_localhost(policy: dict[str, object]) -> str:
    policy.pop("URLBlocklist", None)
    allowlist = policy.setdefault("URLAllowlist", [])
    if not isinstance(allowlist, list):
        raise ValueError("Chromium URLAllowlist policy must be a list")
    for pattern in ("http://127.0.0.1:*", "http://localhost:*"):
        if pattern not in allowlist:
            allowlist.append(pattern)
    return json.dumps(policy, indent=2) + "\n"


def _make_policy_testable(path: Path) -> bytes | None:
    if not path.parent.is_dir() or not path.exists():
        return None
    loaded = _read_blocking_policy(path)
    if loaded is None:
        return None
    original, policy = loaded
    try:
        path.write_text(_allow_localhost(policy), encoding="utf-8")
    except (OSError, ValueError):
        return None
    return original


@contextmanager
def allow_localhost_for_managed_chromium() -> Iterator[None]:
    """Remove a global URLBlocklist only for the lifetime of a browser smoke test.

    Some controlled CI images ship Chromium with ``URLBlocklist: ["*"]``. This helper
    modifies only writable JSON policy files, keeps byte-for-byte backups, and restores
    them in ``finally``. Ordinary developer machines are left untouched.
    """

    policy_root = Path("/etc/chromium/policies/managed")
    candidates = list(policy_root.glob("*.json")) if policy_root.is_dir() else []
    backups: dict[Path, bytes] = {}
    try:
        for path in candidates:
            original = _make_policy_testable(path)
            if original is not None:
                backups[path] = original
        yield
    finally:
        for path, original in backups.items():
            path.write_bytes(original)


def terminate_child(process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> None:
    """Stop a test-owned child process and do not return while it is still alive.

    ``Popen.terminate()`` alone only *requests* an exit; a server wedged in a blocking
    call keeps the port bound and the camera open, so the next test fails for reasons
    that have nothing to do with the code under test. This escalates TERM -> KILL and
    raises if even KILL leaves the process behind, rather than letting a leak pass
    silently as a clean run.
    """

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    try:
        process.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - requires an unkillable child
        raise RuntimeError(
            f"child process {process.pid} survived SIGKILL; a port or camera may stay held"
        ) from exc


@contextmanager
def managed_child(
    process: subprocess.Popen[str] | subprocess.Popen[bytes],
) -> Iterator[subprocess.Popen[str] | subprocess.Popen[bytes]]:
    """Guarantee :func:`terminate_child` runs even when the test body raises."""

    try:
        yield process
    finally:
        terminate_child(process)
