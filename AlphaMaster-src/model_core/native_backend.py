"""Optional C++/CUDA backend loader for AlphaMaster formula kernels."""
from __future__ import annotations

import os
import pathlib
import shutil
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


ROOT = pathlib.Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"


@dataclass(frozen=True)
class NativeBuildStatus:
    available: bool
    reason: str = ""
    cuda_home: str | None = None
    has_nvcc: bool = False
    has_cl: bool = False


def probe_native_build() -> NativeBuildStatus:
    cuda_home = str(CUDA_HOME) if CUDA_HOME else None
    nvcc = shutil.which("nvcc.exe") or shutil.which("nvcc")
    cl = shutil.which("cl.exe") or shutil.which("cl")
    if not torch.cuda.is_available():
        return NativeBuildStatus(False, "torch CUDA is not available", cuda_home, bool(nvcc), bool(cl))
    if not CUDA_HOME and not nvcc:
        return NativeBuildStatus(False, "CUDA Toolkit/nvcc not found", cuda_home, False, bool(cl))
    if not cl:
        return NativeBuildStatus(False, "MSVC cl.exe not found", cuda_home, bool(nvcc), False)
    return NativeBuildStatus(True, "", cuda_home, True, True)


def load_native_extension(verbose: bool = False):
    status = probe_native_build()
    if not status.available:
        raise RuntimeError(f"native CUDA extension cannot be built: {status.reason}")
    build_dir = ROOT.parent / "build" / "alphamaster_native"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="alphamaster_native",
        sources=[
            str(NATIVE_DIR / "alphamaster_native.cpp"),
            str(NATIVE_DIR / "alphamaster_native_cuda.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
        with_cuda=True,
    )
