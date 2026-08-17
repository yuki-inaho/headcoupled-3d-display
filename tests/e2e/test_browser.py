from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
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
from headcoupled_display.tracking import HeadPoseEstimator

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


def write_replay_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a short video and the exact JSON schema emitted by FaceMesh."""

    camera_matrix = np.array(
        [[950.0, 0.0, 640.0], [0.0, 950.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    indices = HeadPoseEstimator.LANDMARK_INDICES
    points = canonical_face_model().pnp_points_opencv_m[indices]
    records: list[dict[str, object]] = []
    video_path = tmp_path / "recording.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280, 720))
    assert writer.isOpened()
    for frame_index, x_offset_m in enumerate((-0.015, 0.0, 0.015)):
        projected, _ = cv2.projectPoints(
            points,
            np.array([[0.03], [0.0], [0.0]], dtype=np.float64),
            np.array([[x_offset_m], [0.0], [0.65]], dtype=np.float64),
            camera_matrix,
            None,
        )
        landmarks = np.zeros((478, 3), dtype=np.float64)
        landmarks[indices, :2] = projected.reshape(-1, 2)
        records.append(
            {"frame": frame_index, "faces": [{"score": 0.99, "landmarks": landmarks.tolist()}]}
        )
        writer.write(np.full((720, 1280, 3), 32 + frame_index * 32, dtype=np.uint8))
    writer.release()
    landmarks_path = tmp_path / "recording.landmarks.json"
    landmarks_path.write_text(json.dumps(records), encoding="utf-8")
    intrinsics_path = tmp_path / "calibration.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "image_width": 1280,
                "image_height": 720,
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": [],
            }
        ),
        encoding="utf-8",
    )
    return landmarks_path, video_path, intrinsics_path


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
            assert (
                page.locator("#camera-preview")
                .get_attribute("src", timeout=10_000)
                .startswith("blob:")
            )
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


@pytest.mark.e2e
def test_recorded_facemesh_replay_drives_dashboard(tmp_path: Path) -> None:
    port = free_port()
    personal_mesh = tmp_path / "shape.pcd"
    write_valid_personal_pcd(personal_mesh)
    landmarks, recording, intrinsics = write_replay_fixture(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "headcoupled_display.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--profile",
        str(ROOT / "config" / "hardware_profile.demo.json"),
        "--source",
        "replay",
        "--replay-landmarks",
        str(landmarks),
        "--replay-video",
        str(recording),
        "--face-model",
        str(personal_mesh),
        "--intrinsics",
        str(intrinsics),
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
            browser = playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--enable-webgl", "--use-angle=swiftshader"],
            )
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_function("document.body.dataset.ready === 'true'", timeout=15_000)
            page.wait_for_function(
                "Number(document.querySelector('#pose-sequence').dataset.sequence) >= 2",
                timeout=15_000,
            )
            assert "録画再生" in page.locator("#tracking-status").inner_text()
            assert (
                page.locator("#camera-preview")
                .get_attribute("src", timeout=10_000)
                .startswith("blob:")
            )
            with urllib.request.urlopen(f"{base_url}/api/profile") as response:
                profile = json.loads(response.read())
            assert (
                profile["hardware_profile"]["quality_metrics"]["camera_intrinsics_imported"] is True
            )
            assert profile["user_profile"]["face_model_path"] == str(personal_mesh.resolve())
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.e2e
def test_browser_pcd_loader_reports_known_bunny_bounds() -> None:
    """`loadAsciiPcd` must report the AABB of the parsed points, not only positions.

    Expected values for `static/assets/bunny.pcd` (13,810 points) were computed
    independently with NumPy over the same file, outside of this test and outside
    of `pcd.js`. `center` here is the AABB midpoint `(min + max) / 2`, which is
    NOT the point centroid (`(-0.00378993, 0.22580822, 0.05619751)`); the two must
    not be confused. This test calls the real browser-side `loadAsciiPcd` via a
    dynamic import so a bug in `pcd.js` cannot be hidden by a Python-side
    recomputation of the bounds.
    """
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "HEADCOUPLED_PROFILE": str(ROOT / "config" / "hardware_profile.demo.json"),
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
            page = browser.new_page()
            # Navigate to the dashboard origin first so the dynamic import below and
            # the PCD fetch inside loadAsciiPcd are same-origin (no CORS involved).
            page.goto(base_url, wait_until="domcontentloaded")
            loaded = page.evaluate(
                """async () => {
                    const module = await import("/static/pcd.js");
                    const result = await module.loadAsciiPcd("/static/assets/bunny.pcd");
                    return { pointCount: result.pointCount, bounds: result.bounds };
                }"""
            )
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    assert loaded["pointCount"] == 13810
    assert loaded["bounds"] is not None, f"loadAsciiPcd must return bounds, got: {loaded}"
    assert loaded["bounds"]["min"] == pytest.approx([-0.5836868, -0.5006086, -0.5753633], abs=1e-6)
    assert loaded["bounds"]["max"] == pytest.approx([0.5809016, 1.1862427, 0.4353630], abs=1e-6)
    assert loaded["bounds"]["center"] == pytest.approx(
        [-0.0013926, 0.34281705, -0.07000015], abs=1e-6
    )


@pytest.mark.e2e
def test_point_cloud_is_anchored_on_the_display_plane() -> None:
    """The scene profile, not the renderer, decides where the cloud sits.

    Asserts the placement numerically from the renderer's own dataset readout so a
    screenshot that merely looks plausible cannot pass this test.
    """
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "HEADCOUPLED_PROFILE": str(ROOT / "config" / "hardware_profile.demo.json"),
            "HEADCOUPLED_SCENE": str(ROOT / "config" / "scene_profile.default.json"),
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
            page = browser.new_page()
            page.goto(base_url, wait_until="domcontentloaded")
            canvas = page.locator("#gl-canvas")
            canvas.wait_for(state="attached")
            page.wait_for_function(
                "() => document.querySelector('#gl-canvas')?.dataset.rendererReady === 'true'",
                timeout=20000,
            )
            scale = float(canvas.get_attribute("data-model-scale"))
            centre = json.loads(canvas.get_attribute("data-model-center-display-m"))
            minimum = json.loads(canvas.get_attribute("data-model-min-display-m"))
            maximum = json.loads(canvas.get_attribute("data-model-max-display-m"))
            scene_id = canvas.get_attribute("data-scene-id")
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert scene_id == "bunny-on-display-plane"
    # The AABB midpoint must land exactly on the screen plane anchor (0, 0, 0).
    assert centre == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    # Uniform scale derived from the asset, not hard-coded: 0.24 m / 1.6868513 m.
    assert scale == pytest.approx(0.142276916, rel=1e-6)
    span = [maximum[axis] - minimum[axis] for axis in range(3)]
    assert max(span) == pytest.approx(0.24, abs=1e-6)
    # The cloud straddles the window, so front and back show opposite parallax.
    assert minimum[2] < 0.0 < maximum[2]
