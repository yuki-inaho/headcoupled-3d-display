"""Unit tests for the live FaceMesh IPC producer's CUDA provider attestation.

These tests run under this repository's own (Python 3.13) virtualenv, so they must not
import ``facemesh_tracking`` or require a real ONNX Runtime / CUDA session. The producer
script defers those imports to inside ``main()`` for exactly this reason; the pieces under
test here (``assert_cuda_providers`` and CLI parsing) are pure and take duck-typed fakes
that mimic the shape of a UniFace-backed pipeline stage.
"""

from __future__ import annotations

import pytest

import scripts.facemesh_ipc_producer as producer


class _FakeSession:
    """Stands in for an ``onnxruntime.InferenceSession``."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class _FakeModel:
    """Stands in for UniFace's internal model object, which owns ``session``."""

    def __init__(self, providers: list[str]) -> None:
        self.session = _FakeSession(providers)


class _FakeStage:
    """Stands in for facemesh_tracking's ``UnifaceFaceDetector`` / ``UnifaceFaceMesh``."""

    def __init__(self, providers: list[str]) -> None:
        self._model = _FakeModel(providers)
        #: The requested (not actual) provider list, as facemesh_tracking exposes it.
        self.providers = providers


class _FakePipeline:
    def __init__(self, detector_providers: list[str], estimator_providers: list[str]) -> None:
        self.detector = _FakeStage(detector_providers)
        self.estimator = _FakeStage(estimator_providers)


class _StageWithoutSession:
    """A stage whose internal layout no longer exposes ``_model.session``."""


def test_assert_cuda_providers_succeeds_when_both_stages_lead_with_cuda() -> None:
    pipeline = _FakePipeline(
        detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        estimator_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    actual = producer.assert_cuda_providers(pipeline)

    assert actual == {
        "detector": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "estimator": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }


def test_assert_cuda_providers_raises_when_detector_is_cpu_only() -> None:
    pipeline = _FakePipeline(
        detector_providers=["CPUExecutionProvider"],
        estimator_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="detector"):
        producer.assert_cuda_providers(pipeline)


def test_assert_cuda_providers_raises_when_estimator_is_cpu_only() -> None:
    pipeline = _FakePipeline(
        detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        estimator_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="estimator"):
        producer.assert_cuda_providers(pipeline)


def test_assert_cuda_providers_does_not_treat_cpu_fallback_as_success() -> None:
    """Both stages silently fell back to CPU: this must fail, not just warn."""
    pipeline = _FakePipeline(
        detector_providers=["CPUExecutionProvider"],
        estimator_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="detector") as excinfo:
        producer.assert_cuda_providers(pipeline)
    assert "estimator" in str(excinfo.value)


def test_assert_cuda_providers_raises_when_session_access_path_is_broken() -> None:
    """If the internal ``_model.session`` layout ever changes, fail loudly.

    The requested provider list must never be substituted as evidence of the actual
    provider - a broken introspection path is an error, not a silent pass.
    """
    pipeline = _FakePipeline(
        detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        estimator_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    pipeline.detector = _StageWithoutSession()

    with pytest.raises(RuntimeError, match="detector"):
        producer.assert_cuda_providers(pipeline)


def test_cli_backend_choices_do_not_offer_tensorrt() -> None:
    parser = producer.build_arg_parser()
    backend_action = next(action for action in parser._actions if action.dest == "backend")

    # The selectable values themselves are what "no TensorRT option" means; TensorRT may
    # still be *mentioned in prose* explaining why it is a documented non-goal (R-PERF-2).
    assert tuple(backend_action.choices) == ("cuda", "cpu")
    assert "tensorrt" not in parser.format_usage().lower()

    args = parser.parse_args(["--backend", "cuda"])
    assert args.backend == "cuda"

    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "tensorrt"])
