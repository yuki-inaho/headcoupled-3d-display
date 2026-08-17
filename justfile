set dotenv-load := true

root := justfile_directory()
# Synthetic demonstration geometry (20 cm / 10 deg). Used only by the synthetic demo
# recipes below; real-input recipes must not use it.
profile := root / "config/hardware_profile.demo.json"
# The mount geometry confirmed on this machine (15 cm / 12 deg). Every recipe that
# feeds *real* input -- recorded replay, live IPC -- uses this, because running real
# faces through demonstration geometry produces a plausible-looking but wrong scene.
local_profile := root / "config/hardware_profile.local.json"
scene_profile := root / "config/scene_profile.default.json"
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
replay-recording landmarks video face_model intrinsics host="127.0.0.1" port="8000" profile_path=local_profile scene_path=scene_profile:
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli serve --host {{host}} --port {{port}} --profile {{profile_path}} --scene {{scene_path}} --source replay --replay-landmarks "{{landmarks}}" --replay-video "{{video}}" --face-model "{{face_model}}" --intrinsics "{{intrinsics}}"

# Start the 3D dashboard ready for a separate Python-3.10 FaceMesh producer on localhost.
serve-ipc face_model intrinsics host="127.0.0.1" port="8000" profile_path=local_profile scene_path=scene_profile:
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli serve --host {{host}} --port {{port}} --profile {{profile_path}} --scene {{scene_path}} --source ipc --face-model "{{face_model}}" --intrinsics "{{intrinsics}}"

# Run this in the FaceMesh Python/CUDA environment after `serve-ipc` is ready.
facemesh-ipc endpoint="http://127.0.0.1:8000/api/input/facemesh" device="/dev/video0":
    cd {{facemesh_project}} && uv run python {{root}}/scripts/facemesh_ipc_producer.py --camera "{{device}}" --endpoint "{{endpoint}}"

# Validate and summarize a hardware profile.
profile-summary path=profile:
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli profile-summary {{path}}

# Run deterministic synthetic calibration and write its result.
synthetic-calibration output="artifacts/synthetic_calibration_result.json":
    PYTHONPATH={{root}}/src {{python}} -m headcoupled_display.cli synthetic-calibrate --profile {{profile}} --output {{output}}

# Generate the retired synthetic bunny point cloud (history only -- do not run this
# against the shipped asset path, it would overwrite the real Stanford Bunny below).
bunny:
    {{python}} scripts/generate_bunny.py --output src/headcoupled_display/static/assets/bunny.pcd

# Regenerate the shipped bunny.pcd from the real Stanford Bunny PLY (bun_zipper,
# 35947 vertices). The default `input` matches import_stanford_bunny.py's own
# DEFAULT_INPUT; pass a different path if that local copy lives elsewhere.
bunny-stanford input="/home/inaho-omen/open3d_data/extract/BunnyMesh/BunnyMesh.ply":
    {{python}} scripts/import_stanford_bunny.py --input "{{input}}" --output src/headcoupled_display/static/assets/bunny.pcd

# Unit and API tests. recorded_cuda is excluded as well as e2e: it needs a CUDA GPU,
# the recording, the personal mesh and the tagcal intrinsics, so leaving it in would
# make this recipe fail on any machine without that hardware.
test:
    PYTHONPATH={{root}}/src {{python}} -m pytest -m "not e2e and not recorded_cuda"

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

# Full verification. Runs unit, API and browser E2E, but NOT recorded_cuda: that one
# needs a CUDA GPU and the real recording, and is run explicitly with
# `just test-e2e-recorded-cuda`.
check:
    {{python}} -m ruff check src tests scripts
    {{python}} -m ruff format --check src tests scripts
    {{python}} -m radon cc -s -n C src
    PYTHONPATH={{root}}/src {{python}} -m pytest -m "not recorded_cuda"

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

# Absolute path to the recorded test video and the default location for its raw
# (pre-validation) per-stage latency samples, shared by benchmark-recorded and
# summarize-performance below (workdoc Aug17-2026 Step 7).
test10_video := facemesh_project / "recordings/test10.avi"
baseline_recorded_raw := root / "artifacts/perf/baseline_recorded_raw.json"

# Step 7: record per-stage latency samples against a recorded video. Must run inside
# facemesh_tracking's own Python 3.10 / CUDA environment (this recipe cd's into it and
# uses `uv run`, not {{python}}) because pydantic/beartype are not installed there.
# Writes raw, not-yet-schema-validated JSON; follow with `just summarize-performance`.
benchmark-recorded video=test10_video output=baseline_recorded_raw *extra:
    cd {{facemesh_project}} && PYTHONPATH={{root}} uv run python {{root}}/scripts/benchmark_recorded.py --video {{video}} --output {{output}} {{extra}}

# Step 7: validate benchmark-recorded's raw JSON against headcoupled_display.performance's
# strict schema, write artifacts/perf/baseline_recorded.json, and append a tracked summary
# to docs/performance_results.md. --command (required) records the exact benchmark-recorded
# invocation for traceability, e.g.:
#   just summarize-performance extra='--command "just benchmark-recorded"'
summarize-performance raw=baseline_recorded_raw *extra:
    PYTHONPATH={{root}}/src {{python}} scripts/summarize_performance.py --raw {{raw}} {{extra}}

# Step 27: sweep detector-refresh interval candidates over the recording. Runs in
# facemesh_tracking's Python 3.10 / CUDA environment; writes raw landmarks and timings.
sweep_raw := root / "artifacts/perf/refresh_sweep_raw.json"
sweep_report := root / "artifacts/perf/refresh_sweep.json"

sweep-refresh video=test10_video output=sweep_raw *extra:
    cd {{facemesh_project}} && uv run python {{root}}/scripts/sweep_detector_refresh.py --video {{video}} --output {{output}} {{extra}}

# Step 27: choose the interval from the sweep, or refuse if none meets every threshold.
# Accuracy is compared against the full-detect reference through the product's own
# HeadPoseEstimator, using the real intrinsics and personal mesh for that recording.
analyze-refresh raw=sweep_raw output=sweep_report *extra:
    PYTHONPATH={{root}}/src {{python}} scripts/analyze_refresh_sweep.py --raw {{raw}} --output {{output}} --profile config/hardware_profile.local.json --user-profile config/user_profile.demo.json {{extra}}

# Step 41: machine-judge the measured reports against the workdoc's thresholds.
validate-performance *args:
    PYTHONPATH={{root}}/src {{python}} scripts/validate_performance.py {{args}}

# Step 32: isolated pyzmq/grpcio venv for the transport-candidate benchmark. Kept fully
# separate from the product .venv -- see requirements.transport-bench.in. pyzmq/grpcio
# must never appear in pyproject.toml or requirements.lock.
setup-transport-bench:
    uv venv --python 3.13 .venv-transport-bench
    uv pip sync --python .venv-transport-bench/bin/python requirements.transport-bench.lock

# Step 33: compare json_http/binary_http/zeromq/grpc under identical control(60Hz)/
# preview(10Hz)/consumer-stall(100ms) conditions, `runs` repetitions each, and write
# artifacts/perf/transport_comparison.json. Must run inside .venv-transport-bench (see
# setup-transport-bench above), not {{python}} -- the product venv never installs
# pyzmq/grpcio.
benchmark-transports runs="5" output="artifacts/perf/transport_comparison.json":
    .venv-transport-bench/bin/python scripts/benchmark_transports.py --runs {{runs}} --output {{output}}

# Step 32: 100-message schema-validation smoke run (all four candidates).
benchmark-transports-smoke:
    .venv-transport-bench/bin/python scripts/benchmark_transports.py --smoke --output artifacts/perf/transport_comparison_smoke.json

# Step 33 (noise-isolated): same comparison as benchmark-transports, but pins the
# producer/consumer subprocesses to disjoint CPUs (taskset -c, no root needed),
# disables the cyclic GC inside them, and widens warmup_messages -- removing host-
# scheduling/GC/connection-setup noise from the timed window without touching
# CONTROL_P95_THRESHOLD_MS or the worst-run judgement in _criterion_control_p95. Pick
# producer_cpus/consumer_cpus from `cat /proc/loadavg` + a fresh /proc/stat per-core
# sample on this host, not blindly -- there is no cgroup/root isolation available, so
# these are a best-effort pin, not an exclusive reservation.
benchmark-transports-isolated runs="25" producer_cpus="1,2" consumer_cpus="4,5" warmup_messages="90" output="artifacts/perf/transport_comparison_isolated.json":
    .venv-transport-bench/bin/python scripts/benchmark_transports.py --runs {{runs}} --producer-cpus {{producer_cpus}} --consumer-cpus {{consumer_cpus}} --gc-disable --warmup-messages {{warmup_messages}} --output {{output}}

# Step 40: production-equivalent end-to-end run. Real recording -> real CUDA FaceMesh
# inference in the Python 3.10 environment -> the adopted two-lane IPC -> metric PnP ->
# /ws/pose -> the off-axis renderer. Exits non-zero when CUDA is unavailable or an input
# artifact is missing; it never skips. Marked recorded_cuda, not e2e, so an ordinary CI
# run neither fails for want of a GPU nor passes by skipping.
test-e2e-recorded-cuda *extra:
    PYTHONPATH={{root}}/src {{python}} -m pytest -m recorded_cuda tests/e2e -q {{extra}}
