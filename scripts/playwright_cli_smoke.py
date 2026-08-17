#!/usr/bin/env python3
"""Run a real Playwright CLI screenshot smoke test against the synthetic server."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from browser_policy import allow_localhost_for_managed_chromium

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for(url: str, process: subprocess.Popen[str], timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate(timeout=1)
            raise RuntimeError(f"server exited early:\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError(f"server not ready: {url}")


def provision_system_chromium_for_cli() -> None:
    """Use system Chromium when the pinned Playwright browser bundle is absent.

    Playwright 1.57.0 expects revision 1200. A normal workstation runs
    ``playwright install chromium`` instead, so this shim writes into the shared
    ms-playwright cache only when HEADCOUPLED_SYSTEM_CHROMIUM_SHIM=1 asks for it.
    """

    if os.getenv("HEADCOUPLED_SYSTEM_CHROMIUM_SHIM") != "1":
        return
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium is None:
        return
    executable = (
        Path.home()
        / ".cache/ms-playwright/chromium_headless_shell-1200"
        / "chrome-headless-shell-linux64/chrome-headless-shell"
    )
    if executable.exists():
        return
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.symlink_to(chromium)


def main() -> None:
    # Do not resolve() first: in a uv venv sys.executable is a symlink into the shared
    # interpreter directory, where no playwright entry point exists.
    local_cli = Path(sys.executable).parent / "playwright"
    playwright_cli = str(local_cli) if local_cli.is_file() else shutil.which("playwright")
    if playwright_cli is None:
        raise RuntimeError("playwright CLI is not installed; run `uv pip sync requirements.lock`")
    provision_system_chromium_for_cli()
    ARTIFACTS.mkdir(exist_ok=True)
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "HEADCOUPLED_PROFILE": str(ROOT / "config/hardware_profile.demo.json"),
            "HEADCOUPLED_USER_PROFILE": str(ROOT / "config/user_profile.demo.json"),
            "HEADCOUPLED_SOURCE": "synthetic",
        }
    )
    server_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "headcoupled_display.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    server = subprocess.Popen(
        server_command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    output = ARTIFACTS / "playwright-cli-dashboard.png"
    try:
        wait_for(f"{url}/api/health", server)
        command = [
            playwright_cli,
            "screenshot",
            "--browser",
            "chromium",
            "--viewport-size",
            "1440,900",
            "--wait-for-selector",
            'body[data-ready="true"]',
            "--wait-for-timeout",
            "1500",
            "--full-page",
            url,
            str(output),
        ]
        with allow_localhost_for_managed_chromium():
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        if not output.is_file() or output.stat().st_size < 10_000:
            raise RuntimeError(f"Playwright CLI did not produce a valid screenshot: {output}")
        (ARTIFACTS / "playwright-cli-command.txt").write_text(
            " ".join(command) + "\n",
            encoding="utf-8",
        )
        print(f"Playwright CLI smoke test passed: {output}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
