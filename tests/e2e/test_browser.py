from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import sync_playwright

from headcoupled_display.face_model import (
    HEAD_TO_OPENCV,
    LEFT_IRIS_CENTRE,
    RIGHT_IRIS_CENTRE,
    canonical_face_model,
)
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


def write_valid_personal_pcd(path: Path) -> None:
    """Make a deterministic reconstructed-model fixture in its documented PCD frame."""

    points_head_m = np.zeros((478, 3), dtype=np.float64)
    points_head_m[:468] = canonical_face_model().points_head_m
    points_head_m[LEFT_IRIS_CENTRE] = (-0.031, 0.026, 0.030)
    points_head_m[RIGHT_IRIS_CENTRE] = (0.033, 0.027, 0.031)
    points_opencv_mm = points_head_m * HEAD_TO_OPENCV * 1000.0
    path.write_text(
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 478\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 478\nDATA ascii\n"
        + "".join(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in points_opencv_mm),
        encoding="ascii",
    )


@pytest.mark.e2e
def test_dashboard_websocket_renderer_and_calibration(tmp_path: Path) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    port = free_port()
    personal_mesh = tmp_path / "shape.pcd"
    write_valid_personal_pcd(personal_mesh)
    user_profile = json.loads((ROOT / "config" / "user_profile.demo.json").read_text())
    user_profile["face_model_path"] = "shape.pcd"
    user_profile_path = tmp_path / "user_profile.json"
    user_profile_path.write_text(json.dumps(user_profile), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "HEADCOUPLED_PROFILE": str(ROOT / "config" / "hardware_profile.demo.json"),
            "HEADCOUPLED_USER_PROFILE": str(user_profile_path),
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
            with urllib.request.urlopen(f"{base_url}/api/profile") as response:
                profile = json.loads(response.read())
            assert profile["user_profile"]["face_model_path"] == str(personal_mesh.resolve())

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
