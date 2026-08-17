#!/usr/bin/env python3
"""Small fallback task runner for environments where the `just` binary is absent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/hardware_profile.demo.json"


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/tasks.py <serve|test|test-e2e|check|profile-summary|synthetic-calibration|playwright-cli>")
    task = sys.argv[1]
    extra = sys.argv[2:]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    commands = {
        "serve": [sys.executable, "-m", "headcoupled_display.cli", "serve", "--profile", str(PROFILE), *extra],
        "test": [sys.executable, "-m", "pytest", "-m", "not e2e", *extra],
        "test-e2e": [sys.executable, "-m", "pytest", "-m", "e2e", "tests/e2e", *extra],
        "check": [sys.executable, "-m", "pytest", *extra],
        "profile-summary": [sys.executable, "-m", "headcoupled_display.cli", "profile-summary", *(extra or [str(PROFILE)])],
        "synthetic-calibration": [sys.executable, "-m", "headcoupled_display.cli", "synthetic-calibrate", "--profile", str(PROFILE), *extra],
        "playwright-cli": [sys.executable, "scripts/playwright_cli_smoke.py", *extra],
    }
    command = commands.get(task)
    if command is None:
        raise SystemExit(f"unknown task: {task}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
