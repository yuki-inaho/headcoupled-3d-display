"""Temporary localhost allowance for locked-down CI Chromium installations."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


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
