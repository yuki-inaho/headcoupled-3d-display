set dotenv-load := true

root := justfile_directory()
profile := root / "config/hardware_profile.demo.json"
python := root / ".venv/bin/python"
facemesh_project := env_var_or_default("FACEMESH_TRACKING_PROJECT", root / "../facemesh_tracking")

default:
    @just --list

# Create the exact tested controller environment without building/installing the source tree.
setup:
    uv venv --python 3.13 .venv
    uv pip sync --python .venv/bin/python requirements.lock

# Fetch the Playwright-managed Chromium used by test-e2e and playwright-cli.
setup-browsers:
    {{python}} -m playwright install chromium

# Run the synthetic demo server.
serve host="127.0.0.1" port="8000":
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli serve --host {{host}} --port {{port}} --profile {{profile}} --source synthetic

# Real USB-camera FaceMesh viewer. Use the V4L2 path rather than numeric index 0.
facemesh-live device="/dev/video0":
    cd {{facemesh_project}} && just cam {{device}}

# Replay a recorded FaceMesh result in the browser dashboard. All four inputs must come
# from the same camera session: the video, its saved landmarks, the personal shape, and K,D.
replay-recording landmarks video face_model intrinsics host="127.0.0.1" port="8000":
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli serve --host {{host}} --port {{port}} --profile {{profile}} --source replay --replay-landmarks "{{landmarks}}" --replay-video "{{video}}" --face-model "{{face_model}}" --intrinsics "{{intrinsics}}"

# Start the 3D dashboard ready for a separate Python-3.10 FaceMesh producer on localhost.
serve-ipc face_model intrinsics host="127.0.0.1" port="8000":
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli serve --host {{host}} --port {{port}} --profile {{profile}} --source ipc --face-model "{{face_model}}" --intrinsics "{{intrinsics}}"

# Run this in the FaceMesh Python/CUDA environment after `serve-ipc` is ready.
facemesh-ipc endpoint="http://127.0.0.1:8000/api/input/facemesh" device="/dev/video0":
    cd {{facemesh_project}} && uv run python {{root}}/scripts/facemesh_ipc_producer.py --camera "{{device}}" --endpoint "{{endpoint}}"

# Validate and summarize a hardware profile.
profile-summary path=profile:
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli profile-summary {{path}}

# Run deterministic synthetic calibration and write its result.
synthetic-calibration output="artifacts/synthetic_calibration_result.json":
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli synthetic-calibrate --profile {{profile}} --output {{output}}

# Generate the bundled synthetic bunny point cloud.
bunny:
    {{python}} scripts/generate_bunny.py --output src/headcoupled_display/static/assets/bunny.pcd

# Unit and API tests.
test:
    PYTHONPATH={{root}}/src {{python}} -m pytest -m "not e2e"

# Browser end-to-end test using system Chromium.
test-e2e:
    PYTHONPATH={{root}}/src {{python}} -m pytest -m e2e tests/e2e

# Playwright CLI screenshot smoke test.
playwright-cli:
    PYTHONPATH={{root}}/src {{python}} scripts/playwright_cli_smoke.py

# Static checks.
lint:
    {{python}} -m ruff check src tests scripts

# Report cyclomatic complexity (C or worse) and maintainability for the Python source.
complexity:
    {{python}} -m radon cc -s -n C src
    {{python}} -m radon mi -s src

# Measure only the cached PnP/coordinate-transform hot path; camera inference is excluded.
benchmark-tracking iterations="5000":
    PYTHONPATH={{root}}/src {{python}} scripts/benchmark_tracking.py --iterations {{iterations}}

# Full verification.
check:
    {{python}} -m ruff check src tests scripts
    {{python}} -m radon cc -s -n C src
    PYTHONPATH={{root}}/src {{python}} -m pytest

# Merge a tagcal calibration.json/YAML into the hardware profile.
import-tagcal calibration output="config/hardware_profile.measured.json":
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli import-tagcal {{calibration}} --profile {{profile}} --output {{output}}

# Set up the bundled AprilTag camera-intrinsics tool (Python 3.11-3.13).
setup-tagcal:
    cd components/tagcal && uv sync --all-extras

# Open the bundled AprilTag calibration panel.
tagcal-panel:
    cd components/tagcal && uv run --extra gui tagcal panel

# Set up the bundled UniFace/FaceMesh tracker (its own Python 3.10 GPU environment).
setup-facemesh:
    cd components/facemesh_tracking && uv sync

# Inspect GPU, ONNX Runtime, models, and cameras for the FaceMesh component.
facemesh-doctor:
    cd components/facemesh_tracking && just doctor

# Fallback task runner when the `just` binary is unavailable.
task name *args:
    PYTHONPATH={{root}}/src {{python}} scripts/tasks.py {{name}} {{args}}
