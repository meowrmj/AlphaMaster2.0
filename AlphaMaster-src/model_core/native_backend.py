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


NATIVE_BINARY_OPS = {
    "ADD": 1,
    "SUB": 2,
    "MUL": 3,
    "DIV": 4,
    "MAX": 5,
    "MIN": 6,
}

NATIVE_TERNARY_OPS = {
    "IF_GT": 101,
    "GATE": 102,
}


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


class NativeElementwiseOps:
    """Thin checked wrapper around the optional native extension."""

    def __init__(self, ext=None, verbose: bool = False):
        self.ext = ext if ext is not None else load_native_extension(verbose=verbose)

    @staticmethod
    def supports(op_name: str, arity: int) -> bool:
        if arity == 2:
            return op_name in NATIVE_BINARY_OPS
        if arity == 3:
            return op_name in NATIVE_TERNARY_OPS
        return False

    def apply(self, op_name: str, *args: torch.Tensor) -> torch.Tensor:
        arity = len(args)
        if arity == 2 and op_name in NATIVE_BINARY_OPS:
            return self.ext.elementwise2(args[0], args[1], NATIVE_BINARY_OPS[op_name])
        if arity == 3 and op_name in NATIVE_TERNARY_OPS:
            return self.ext.elementwise3(args[0], args[1], args[2], NATIVE_TERNARY_OPS[op_name])
        raise NotImplementedError(f"native op not supported: {op_name}/{arity}")
