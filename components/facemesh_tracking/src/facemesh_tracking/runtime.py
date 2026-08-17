"""Execution-backend selection and CUDA library resolution.

Why the library preloading below exists
---------------------------------------
``onnxruntime-gpu==1.18.0`` (the PyPI wheel) is built against CUDA 11.8 + cuDNN 8.9.
cuDNN 9 ships no Conv execution kernels for Pascal (sm_61, e.g. GTX 1070), so a
system-wide cuDNN 9 makes every convolution fail with ``CUDNN_STATUS_EXECUTION_FAILED``.
The ``nvidia-*-cu11`` wheels declared in ``pyproject.toml`` provide a matching CUDA 11.8
runtime inside the venv; :func:`preload_cuda_libraries` binds those *before* onnxruntime
is imported, so the loader never reaches the system copies.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Loaded in dependency order; cuDNN sub-libraries must follow the cuDNN facade.
_CUDA_LIBRARY_ORDER: tuple[str, ...] = (
    "libcudart.so.11.0",
    "libcublasLt.so.11",
    "libcublas.so.11",
    "libcufft.so.10",
    "libcurand.so.10",
    "libcudnn.so.8",
    "libcudnn_ops_infer.so.8",
    "libcudnn_cnn_infer.so.8",
    "libcudnn_adv_infer.so.8",
)

_preloaded = False


class Backend(str, Enum):
    """Execution backend requested by the caller."""

    CUDA = "cuda"
    TENSORRT = "tensorrt"
    CPU = "cpu"


def _nvidia_lib_dirs() -> list[Path]:
    """Return ``site-packages/nvidia/*/lib`` directories present in this environment."""
    dirs: list[Path] = []
    for site_dir in {Path(p) for p in sys.path if p}:
        nvidia_root = site_dir / "nvidia"
        if not nvidia_root.is_dir():
            continue
        dirs.extend(sorted(p for p in nvidia_root.glob("*/lib") if p.is_dir()))
    return dirs


def preload_cuda_libraries() -> list[Path]:
    """Bind the venv-local CUDA 11.8 / cuDNN 8 shared objects with ``RTLD_GLOBAL``.

    Must run before ``import onnxruntime`` — including the import UniFace does internally.
    Idempotent; missing libraries are skipped so CPU-only environments keep working. Also
    prepends the directories to ``LD_LIBRARY_PATH`` for any lazily ``dlopen``-ed
    sub-library.

    Returns the directories that were made available.
    """
    global _preloaded
    lib_dirs = _nvidia_lib_dirs()
    if _preloaded or not lib_dirs:
        return lib_dirs

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(d) for d in lib_dirs), os.environ.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)

    by_name = {path.name: path for d in lib_dirs for path in d.glob("*.so*")}
    for soname in _CUDA_LIBRARY_ORDER:
        path = by_name.get(soname)
        if path is None:
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # pragma: no cover - depends on local CUDA install
            logger.debug("Could not preload %s: %s", path, exc)

    _preloaded = True
    return lib_dirs


def providers_for(backend: Backend, device_id: int = 0) -> list:
    """Build the onnxruntime provider list for ``backend``, best first."""
    if backend is Backend.CPU:
        return ["CPUExecutionProvider"]
    cuda = ("CUDAExecutionProvider", {"device_id": device_id})
    if backend is Backend.CUDA:
        return [cuda, "CPUExecutionProvider"]
    return [
        ("TensorrtExecutionProvider", {"device_id": device_id, "trt_fp16_enable": True}),
        cuda,
        "CPUExecutionProvider",
    ]
