"""Test-only helpers: Chromium policy relaxation and guaranteed child-process cleanup."""

from __future__ import annotations

import json
import os
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


#: Chromium flags every browser test shares.
_CHROMIUM_BASE_ARGS = (
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
)

#: Ask ANGLE for the system OpenGL driver. On this machine that resolves to the real
#: GPU; SwiftShader's software rasteriser presents roughly 1.8 frames per second for
#: this scene, which makes every browser-side latency figure a property of the
#: rasteriser rather than of the renderer under test.
_CHROMIUM_HARDWARE_GL_ARGS = ("--use-gl=angle", "--use-angle=gl", "--enable-gpu")

#: Software fallback for machines with no usable GPU. Correctness tests still pass here;
#: latency figures measured under it must not be presented as hardware results.
_CHROMIUM_SOFTWARE_GL_ARGS = ("--use-angle=swiftshader",)


def chromium_args(*, software: bool | None = None) -> list[str]:
    """Chromium launch flags, defaulting to hardware GL.

    ``HEADCOUPLED_FORCE_SOFTWARE_GL=1`` forces the software rasteriser, for machines
    without a usable GPU. The choice is deliberately explicit rather than silently
    probed: a latency number measured on SwiftShader and one measured on a GPU differ by
    more than an order of magnitude, so which was used has to be recorded, not guessed.
    """

    if software is None:
        software = os.getenv("HEADCOUPLED_FORCE_SOFTWARE_GL") == "1"
    tail = _CHROMIUM_SOFTWARE_GL_ARGS if software else _CHROMIUM_HARDWARE_GL_ARGS
    return [*_CHROMIUM_BASE_ARGS, *tail]


def webgl_renderer(page: object) -> str:
    """Read the unmasked WebGL renderer string, so an artefact records what drew it."""

    return page.evaluate(
        """() => {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl2');
            if (!gl) return 'no webgl2';
            const info = gl.getExtension('WEBGL_debug_renderer_info');
            return info
                ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL)
                : gl.getParameter(gl.RENDERER);
        }"""
    )
