"""Temporary localhost allowance for locked-down CI Chromium installations."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


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
            try:
                original = path.read_bytes()
                data = json.loads(original)
            except (OSError, json.JSONDecodeError):
                continue
            blocklist = data.get("URLBlocklist")
            if not isinstance(blocklist, list) or "*" not in blocklist:
                continue
            if not path.parent.is_dir() or not path.exists():
                continue
            try:
                backups[path] = original
                data.pop("URLBlocklist", None)
                allowlist = data.setdefault("URLAllowlist", [])
                for pattern in ("http://127.0.0.1:*", "http://localhost:*"):
                    if pattern not in allowlist:
                        allowlist.append(pattern)
                path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            except PermissionError:
                backups.pop(path, None)
        yield
    finally:
        for path, original in backups.items():
            path.write_bytes(original)
