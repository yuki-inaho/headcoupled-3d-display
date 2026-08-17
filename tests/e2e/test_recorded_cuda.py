"""Production-equivalent run: recording -> real CUDA FaceMesh -> IPC -> WebGL.

Marked ``recorded_cuda`` rather than ``e2e`` on purpose. This needs a CUDA GPU, the
recording, the personal mesh and the tagcal intrinsics; folding it into ``e2e`` would
make an ordinary run either fail for want of hardware or skip and look green.

Missing prerequisites fail the test. They are never skipped: a skip here reads as "the
production path is fine" when in fact nothing was exercised.

No synthetic input is involved. The synthetic source is a unit-regression device and is
not a stand-in for an acceptance result on real input.
"""

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

from headcoupled_display.testing_support import (
    allow_localhost_for_managed_chromium,
    chromium_args,
    terminate_child,
    webgl_renderer,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "perf"

FACEMESH_PROJECT = Path(
    os.getenv("FACEMESH_TRACKING_PROJECT", str(ROOT.parent / "facemesh_tracking"))
)
RECORDING = FACEMESH_PROJECT / "recordings" / "test10.avi"
PERSONAL_MESH = FACEMESH_PROJECT / "recordings" / "me" / "shape.pcd"
INTRINSICS = Path(
    os.getenv(
        "HEADCOUPLED_TAGCAL_CALIBRATION",
        str(
            ROOT.parent
            / "apriltag-camera-calibrator"
            / "artifacts"
            / "eval_refine"
            / "calibration.json"
        ),
    )
)

#: Enough frames to populate the timing ring without making the run long. The recording
#: decodes 294 frames in total; the producer stops itself at this count.
PRODUCER_FRAMES = 120

PREVIEW_WIDTH_PX = 640
PREVIEW_HEIGHT_PX = 360


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, process: subprocess.Popen[str], timeout_s: float = 40.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"server exited early ({process.returncode}):\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"{url} did not become ready within {timeout_s}s")


def require_prerequisites() -> None:
    missing = [str(path) for path in (RECORDING, PERSONAL_MESH, INTRINSICS) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "recorded CUDA acceptance requires real inputs that are absent: "
            + ", ".join(missing)
            + " -- this run proves nothing and must not be reported as a skip"
        )
    if not (FACEMESH_PROJECT / "pyproject.toml").is_file():
        raise RuntimeError(f"FaceMesh project not found at {FACEMESH_PROJECT}")


def start_server(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "headcoupled_display.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--profile",
            str(ROOT / "config" / "hardware_profile.local.json"),
            "--scene",
            str(ROOT / "config" / "scene_profile.default.json"),
            "--source",
            "ipc",
            "--face-model",
            str(PERSONAL_MESH),
            "--intrinsics",
            str(INTRINSICS),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def start_producer(base_url: str) -> subprocess.Popen[str]:
    """Run the producer inside the FaceMesh project's own Python 3.10 / CUDA environment.

    ``--backend cuda`` is not a request: the producer inspects the live ONNX Runtime
    sessions and exits non-zero unless CUDAExecutionProvider is actually leading, so a
    silent CPU fallback fails this test rather than passing it slowly.
    """

    return subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            str(ROOT / "scripts" / "facemesh_ipc_producer.py"),
            "--source",
            str(RECORDING),
            "--pacing",
            "realtime",
            "--backend",
            "cuda",
            "--max-frames",
            str(PRODUCER_FRAMES),
            "--endpoint",
            f"{base_url}/api/input/facemesh",
        ],
        cwd=FACEMESH_PROJECT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def measure_clock_offset(page: object) -> dict[str, float]:
    """Measure the browser<->server Unix clock offset the NTP way, over a WebSocket.

    Success condition 10 subtracts a producer Unix timestamp from a browser one. Those
    are two clocks; assuming they agree is the thing this measurement exists to stop.
    Each exchange is bracketed by two browser readings, so the offset is known to within
    half the round trip, and that half-round-trip is the uncertainty.

    A WebSocket echo rather than an HTTP fetch: the fetch path carries request parsing
    and response handling that put the bound several milliseconds above the 2 ms the
    condition allows, which would reject every run for a reason that has nothing to do
    with the latency being measured. The fastest exchange of many is kept, which is
    standard NTP practice -- a slow exchange only ever widens the bound, so the minimum
    is the tightest honest bound rather than a relaxation of one.
    """

    return page.evaluate(
        """() => new Promise((resolve, reject) => {
            const url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/clock`;
            const socket = new WebSocket(url);
            const samples = [];
            let sentAt = 0;
            const total = 40;
            const ping = () => { sentAt = performance.timeOrigin + performance.now(); socket.send('t'); };
            socket.addEventListener('open', ping);
            socket.addEventListener('error', () => reject(new Error('clock socket failed')));
            socket.addEventListener('message', (event) => {
                const after = performance.timeOrigin + performance.now();
                const serverMs = JSON.parse(event.data).server_unix_ns / 1e6;
                samples.push({
                    offsetMs: serverMs - (sentAt + after) / 2,
                    uncertaintyMs: (after - sentAt) / 2,
                });
                if (samples.length < total) { ping(); return; }
                socket.close();
                samples.sort((a, b) => a.uncertaintyMs - b.uncertaintyMs);
                resolve({
                    offset_ms: samples[0].offsetMs,
                    uncertainty_ms: samples[0].uncertaintyMs,
                    samples: samples.length,
                });
            });
        })"""
    )


@pytest.mark.recorded_cuda
def test_recording_reaches_webgl_through_real_cuda_inference() -> None:
    require_prerequisites()
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = start_server(port)
    producer: subprocess.Popen[str] | None = None
    try:
        wait_for_server(f"{base_url}/api/health", server)
        producer = start_producer(base_url)
        with allow_localhost_for_managed_chromium(), sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=os.getenv("HEADCOUPLED_CHROMIUM") or None,
                args=chromium_args(),
            )
            page = browser.new_page()
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_function("() => document.body.dataset.ready === 'true'", timeout=30000)
            # The producer needs to load its CUDA models before the first packet lands.
            page.wait_for_function(
                "() => Number(document.querySelector('#pose-sequence').dataset.sequence) > 20",
                timeout=120000,
            )
            page.wait_for_function(
                "() => (window.headcoupledTimingSummary()?.inferenceSampleCount ?? 0) >= 10",
                timeout=60000,
            )
            clock = measure_clock_offset(page)
            webgl = webgl_renderer(page)
            summary = page.evaluate("() => window.headcoupledTimingSummary()")
            canvas = page.locator("#gl-canvas")
            centre = json.loads(canvas.get_attribute("data-model-center-display-m"))
            renderer_mode = canvas.get_attribute("data-renderer-mode")
            preview = page.evaluate(
                """() => {
                    const image = document.querySelector('#camera-preview');
                    return {width: image.naturalWidth, height: image.naturalHeight};
                }"""
            )
            runtime = json.loads(
                page.evaluate(
                    "async () => JSON.stringify(await (await fetch('/api/runtime')).json())"
                )
            )
            browser.close()
    finally:
        if producer is not None:
            terminate_child(producer)
        terminate_child(server)

    assert runtime["source"] == "ipc", runtime
    assert renderer_mode == "WebGL2"
    assert centre == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert preview == {"width": PREVIEW_WIDTH_PX, "height": PREVIEW_HEIGHT_PX}, preview
    assert summary["sequenceReversalCount"] == 0
    assert summary["inferenceSampleCount"] >= 10

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "recorded_cuda_e2e.json").write_text(
        json.dumps(
            {
                "median_ms": summary["inferenceToDrawP50Ms"],
                "p95_ms": summary["inferenceToDrawP95Ms"],
                "sample_count": summary["inferenceSampleCount"],
                "receive_to_draw_p95_ms": summary["receiveToDrawP95Ms"],
                "cpu_draw_p95_ms": summary["cpuDrawP95Ms"],
                "frame_interval_p50_ms": summary["frameIntervalP50Ms"],
                "sequence_reversals": summary["sequenceReversalCount"],
                "renderer_mode": renderer_mode,
                "preview": preview,
                "source": runtime["source"],
                "recording": str(RECORDING),
                "producer_frames": PRODUCER_FRAMES,
                "clock_offset_ms": clock["offset_ms"],
                "clock_uncertainty_ms": clock["uncertainty_ms"],
                "clock_domain": "unix_ns",
                "excludes": "camera exposure and capture; this is a recording",
                "webgl_renderer": webgl,
                "environment": "headless chromium (not the physical display)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
