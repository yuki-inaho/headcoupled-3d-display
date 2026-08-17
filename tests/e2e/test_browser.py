from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from headcoupled_display.testing_support import allow_localhost_for_managed_chromium

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, process: subprocess.Popen[str], timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate(timeout=1)
            raise RuntimeError(f"server exited early ({process.returncode}):\n{stdout}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError(f"server did not become ready: {url}")


@pytest.mark.e2e
def test_dashboard_websocket_renderer_and_calibration() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "HEADCOUPLED_PROFILE": str(ROOT / "config" / "hardware_profile.demo.json"),
            "HEADCOUPLED_USER_PROFILE": str(ROOT / "config" / "user_profile.demo.json"),
            "HEADCOUPLED_SOURCE": "synthetic",
        }
    )
    command = [
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
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(f"{base_url}/api/health", process)
        with allow_localhost_for_managed_chromium(), sync_playwright() as playwright:
            # Default to the Playwright-managed Chromium. HEADCOUPLED_CHROMIUM overrides it
            # for images that ship only a system browser.
            executable_path = os.getenv("HEADCOUPLED_CHROMIUM")
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=executable_path or None,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--enable-webgl",
                    "--ignore-gpu-blocklist",
                    "--use-angle=swiftshader",
                ],
            )
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_function("document.body.dataset.ready === 'true'", timeout=15_000)
            page.wait_for_selector('#gl-canvas[data-renderer-ready="true"]', timeout=15_000)
            page.wait_for_function(
                "Number(document.querySelector('#pose-sequence').dataset.sequence) >= 2",
                timeout=15_000,
            )

            assert page.title() == "Head-Coupled 3D Display"
            assert "20.0 cm" in page.locator("#mount-height").inner_text()
            assert "10.0°" in page.locator("#mount-pitch").inner_text()
            assert "13,810 points" in page.locator("#renderer-status").inner_text()
            assert page.locator("#camera-preview").get_attribute("src", timeout=10_000).startswith("blob:")

            page.locator("#calibrate-synthetic").click()
            page.wait_for_function(
                "document.querySelector('#calibration-result').dataset.status === 'success'",
                timeout=20_000,
            )
            calibration_text = page.locator("#calibration-result").inner_text()
            assert "較正成功" in calibration_text
            assert "高さ誤差" in calibration_text

            screenshot = ARTIFACTS / "playwright-e2e-dashboard.png"
            page.screenshot(path=str(screenshot), full_page=True)
            browser.close()
            assert screenshot.is_file() and screenshot.stat().st_size > 10_000
            assert not page_errors, json.dumps(page_errors, ensure_ascii=False)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
