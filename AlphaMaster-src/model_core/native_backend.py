"""Optional C++/CUDA backend loader for AlphaMaster formula kernels."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load


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

NATIVE_ROLLING_BINARY_OPS = {
    "TS_CORR_10": 2010,
    "COVARIANCE_10": 2020,
}

NATIVE_UNARY_OPS = {
    "NEG": 51,
    "ABS": 52,
    "SIGN": 53,
    "POWER": 54,
    "SIGNED_POWER_2": 55,
    "SIGNED_LOG": 56,
    "SQRT": 57,
    "CLIP": 58,
    "SIGMOID": 59,
    "TANH_SQUASH": 60,
}

NATIVE_TERNARY_OPS = {
    "IF_GT": 101,
    "GATE": 102,
}

NATIVE_SHIFT_OPS = {
    "DELAY1": 201,
    "DELAY4": 204,
    "DELTA": 211,
    "DELTA_5": 215,
}

NATIVE_ROLLING_OPS = {
    "TS_MEAN_5": 305,
    "TS_MEAN_10": 310,
    "TS_MEAN_20": 320,
    "TS_SUM_5": 405,
    "TS_SUM_10": 410,
    "TS_SUM_20": 420,
    "TS_ZSCORE_10": 510,
    "TS_ZSCORE_20": 520,
    "WINSORIZE": 580,
    "TS_STD_5": 605,
    "TS_STD_10": 610,
    "TS_STD_20": 620,
    "TS_RANK_5": 705,
    "TS_RANK_10": 710,
    "TS_RANK_20": 720,
    "TS_MIN_10": 810,
    "TS_MIN_20": 820,
    "TS_MAX_10": 910,
    "TS_MAX_20": 920,
    "TS_QUANTILE_10": 1010,
    "TS_SKEW_10": 1020,
    "TS_ARG_MAX_5": 1105,
    "TS_ARG_MIN_5": 1205,
    "DECAY": 1303,
    "WMA": 1304,
    "DECAY_LINEAR_5": 1305,
    "TS_DECAY_EXP_5": 1306,
    "EMA_5": 1405,
    "EMA_20": 1420,
    "MOMENTUM_5": 1505,
    "MOMENTUM_10": 1510,
    "MAX3": 1603,
}

NATIVE_CROSS_SECTIONAL_OPS = {
    "CS_SCALE": 3020,
    "CS_NEUTRALIZE": 3030,
}

NUMERICALLY_STABLE_NATIVE_OPS = (
    set(NATIVE_BINARY_OPS)
    | set(NATIVE_ROLLING_BINARY_OPS)
    | set(NATIVE_UNARY_OPS)
    | set(NATIVE_TERNARY_OPS)
    | set(NATIVE_SHIFT_OPS)
    | set(NATIVE_ROLLING_OPS)
    | set(NATIVE_CROSS_SECTIONAL_OPS)
)

AGGRESSIVE_DISABLED_OPS = {
    "DECAY",
    "MOMENTUM_10",
}

STRICT_DISABLED_OPS = {
    "COVARIANCE_10",
    "CS_NEUTRALIZE",
    "CS_SCALE",
    "DECAY",
    "DECAY_LINEAR_5",
    "DIV",
    "EMA_5",
    "EMA_20",
    "MOMENTUM_5",
    "MOMENTUM_10",
    "TS_ARG_MAX_5",
    "TS_ARG_MIN_5",
    "TS_CORR_10",
    "TS_DECAY_EXP_5",
    "TS_MAX_10",
    "TS_MAX_20",
    "TS_MEAN_5",
    "TS_MEAN_10",
    "TS_MEAN_20",
    "TS_MIN_10",
    "TS_MIN_20",
    "TS_QUANTILE_10",
    "TS_RANK_5",
    "TS_RANK_10",
    "TS_RANK_20",
    "TS_SKEW_10",
    "TS_STD_5",
    "TS_STD_10",
    "TS_STD_20",
    "TS_SUM_5",
    "TS_SUM_10",
    "TS_SUM_20",
    "TS_ZSCORE_10",
    "TS_ZSCORE_20",
    "WINSORIZE",
    "WMA",
}


@dataclass(frozen=True)
class NativeBuildStatus:
    available: bool
    reason: str = ""
    cuda_home: str | None = None
    has_nvcc: bool = False
    has_cl: bool = False


def _prepend_path(path: pathlib.Path | str | None) -> None:
    if not path:
        return
    path = pathlib.Path(path)
    if not path.exists():
        return
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    path_s = str(path)
    if not any(pathlib.Path(p) == path for p in parts if p):
        os.environ["PATH"] = path_s + (os.pathsep + current if current else "")


def _discover_cuda_home() -> str | None:
    if cpp_extension.CUDA_HOME:
        return str(cpp_extension.CUDA_HOME)
    env_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    if env_home and (pathlib.Path(env_home) / "bin" / "nvcc.exe").exists():
        return env_home
    nvcc = shutil.which("nvcc.exe") or shutil.which("nvcc")
    if nvcc:
        return str(pathlib.Path(nvcc).resolve().parents[1])
    cuda_root = pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        candidates = sorted(cuda_root.glob("v*"), reverse=True)
        for path in candidates:
            if (path / "bin" / "nvcc.exe").exists():
                return str(path)
    return None


def _discover_msvc_cl() -> pathlib.Path | None:
    cl = shutil.which("cl.exe") or shutil.which("cl")
    if cl:
        return pathlib.Path(cl).resolve()

    vswhere = pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        try:
            out = subprocess.check_output(
                [
                    str(vswhere),
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                candidate = pathlib.Path(line.strip())
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    for root in (
        pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        pathlib.Path(r"C:\Program Files\Microsoft Visual Studio"),
    ):
        if not root.exists():
            continue
        matches = sorted(root.glob(r"**\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"), reverse=True)
        for candidate in matches:
            if candidate.exists():
                return candidate
    return None


def _configure_native_build_environment() -> tuple[str | None, pathlib.Path | None]:
    _prepend_path(pathlib.Path(sys.executable).resolve().parent)

    cuda_home = _discover_cuda_home()
    if cuda_home:
        cuda_home_path = pathlib.Path(cuda_home)
        os.environ.setdefault("CUDA_HOME", str(cuda_home_path))
        os.environ.setdefault("CUDA_PATH", str(cuda_home_path))
        cpp_extension.CUDA_HOME = str(cuda_home_path)
        _prepend_path(cuda_home_path / "bin")

    cl_path = _discover_msvc_cl()
    if cl_path:
        _prepend_path(cl_path.parent)
    return cuda_home, cl_path


def probe_native_build() -> NativeBuildStatus:
    cuda_home, cl_path = _configure_native_build_environment()
    nvcc = shutil.which("nvcc.exe") or shutil.which("nvcc")
    cl = shutil.which("cl.exe") or shutil.which("cl") or (str(cl_path) if cl_path else None)
    if not torch.cuda.is_available():
        return NativeBuildStatus(False, "torch CUDA is not available", cuda_home, bool(nvcc), bool(cl))
    if not cuda_home and not nvcc:
        return NativeBuildStatus(False, "CUDA Toolkit/nvcc not found", cuda_home, False, bool(cl))
    if not cl:
        return NativeBuildStatus(False, "MSVC cl.exe not found", cuda_home, bool(nvcc), False)
    return NativeBuildStatus(True, "", cuda_home, True, True)


def load_native_extension(verbose: bool = False):
    status = probe_native_build()
    if not status.available:
        raise RuntimeError(f"native CUDA extension cannot be built: {status.reason}")
    if status.cuda_home:
        os.environ.setdefault("CUDA_HOME", status.cuda_home)
        os.environ.setdefault("CUDA_PATH", status.cuda_home)
        cpp_extension.CUDA_HOME = status.cuda_home
    if hasattr(cpp_extension, "SUBPROCESS_DECODE_ARGS"):
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
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
        extra_cuda_cflags=["-O3", "--fmad=false"],
        verbose=verbose,
        with_cuda=True,
    )


class NativeElementwiseOps:
    """Thin checked wrapper around the optional native extension."""

    def __init__(self, ext=None, verbose: bool = False):
        self.ext = ext if ext is not None else load_native_extension(verbose=verbose)
        self.policy = os.getenv("ALPHAMASTER_NATIVE_OP_POLICY", "aggressive").strip().lower()
        if self.policy in {"strict", "verified"}:
            disabled_ops = set(STRICT_DISABLED_OPS)
        elif self.policy == "aggressive":
            disabled_ops = set(AGGRESSIVE_DISABLED_OPS)
        else:
            raise ValueError(f"unknown native op policy: {self.policy}")

        extra_disabled = os.getenv("ALPHAMASTER_NATIVE_DISABLED_OPS", "")
        disabled_ops.update(op.strip().upper() for op in extra_disabled.split(",") if op.strip())
        self.disabled_ops = disabled_ops

    def supports(self, op_name: str, arity: int) -> bool:
        if op_name in self.disabled_ops:
            return False
        if op_name not in NUMERICALLY_STABLE_NATIVE_OPS:
            return False
        if arity == 1:
            return (
                op_name in NATIVE_UNARY_OPS
                or op_name in NATIVE_SHIFT_OPS
                or op_name in NATIVE_ROLLING_OPS
                or op_name in NATIVE_CROSS_SECTIONAL_OPS
            )
        if arity == 2:
            return op_name in NATIVE_BINARY_OPS or op_name in NATIVE_ROLLING_BINARY_OPS
        if arity == 3:
            return op_name in NATIVE_TERNARY_OPS
        return False

    def supports_shift_unary(self, shift_op_name: str, unary_op_name: str) -> bool:
        if os.getenv("ALPHAMASTER_NATIVE_FUSED_SHIFT_UNARY", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return (
            shift_op_name in NATIVE_SHIFT_OPS
            and unary_op_name in NATIVE_UNARY_OPS
            and self.supports(shift_op_name, 1)
            and self.supports(unary_op_name, 1)
            and hasattr(self.ext, "fused_shift_unary")
        )

    def apply_shift_unary(self, shift_op_name: str, unary_op_name: str, arg: torch.Tensor) -> torch.Tensor:
        if not self.supports_shift_unary(shift_op_name, unary_op_name):
            raise NotImplementedError(f"native fused shift+unary not supported: {shift_op_name}->{unary_op_name}")
        return self.ext.fused_shift_unary(
            arg,
            NATIVE_SHIFT_OPS[shift_op_name],
            NATIVE_UNARY_OPS[unary_op_name],
        )

    def supports_binary_branch(self, binary_op_name: str, branch_op_name: str) -> bool:
        if os.getenv("ALPHAMASTER_NATIVE_FUSED_BINARY_BRANCH", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return (
            binary_op_name in NATIVE_BINARY_OPS
            and branch_op_name in NATIVE_TERNARY_OPS
            and self.supports(binary_op_name, 2)
            and self.supports(branch_op_name, 3)
            and hasattr(self.ext, "fused_binary_branch")
        )

    def apply_binary_branch(
        self,
        binary_op_name: str,
        branch_op_name: str,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        branch_condition: torch.Tensor,
        branch_true_value: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports_binary_branch(binary_op_name, branch_op_name):
            raise NotImplementedError(f"native fused binary+branch not supported: {binary_op_name}->{branch_op_name}")
        return self.ext.fused_binary_branch(
            lhs,
            rhs,
            branch_condition,
            branch_true_value,
            NATIVE_BINARY_OPS[binary_op_name],
            NATIVE_TERNARY_OPS[branch_op_name],
        )

    def supports_unary_binary_branch(self, unary_op_name: str, binary_op_name: str, branch_op_name: str) -> bool:
        if os.getenv("ALPHAMASTER_NATIVE_FUSED_UNARY_BINARY_BRANCH", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return (
            unary_op_name in NATIVE_UNARY_OPS
            and binary_op_name in NATIVE_BINARY_OPS
            and branch_op_name in NATIVE_TERNARY_OPS
            and self.supports(unary_op_name, 1)
            and self.supports(binary_op_name, 2)
            and self.supports(branch_op_name, 3)
            and hasattr(self.ext, "fused_unary_binary_branch")
        )

    def apply_unary_binary_branch(
        self,
        unary_op_name: str,
        binary_op_name: str,
        branch_op_name: str,
        unary_arg: torch.Tensor,
        binary_lhs: torch.Tensor,
        branch_condition: torch.Tensor,
        branch_true_value: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports_unary_binary_branch(unary_op_name, binary_op_name, branch_op_name):
            raise NotImplementedError(
                f"native fused unary+binary+branch not supported: {unary_op_name}->{binary_op_name}->{branch_op_name}"
            )
        return self.ext.fused_unary_binary_branch(
            unary_arg,
            binary_lhs,
            branch_condition,
            branch_true_value,
            NATIVE_UNARY_OPS[unary_op_name],
            NATIVE_BINARY_OPS[binary_op_name],
            NATIVE_TERNARY_OPS[branch_op_name],
        )

    def supports_unary_unary_branch(self, first_unary_op_name: str, second_unary_op_name: str, branch_op_name: str) -> bool:
        if os.getenv("ALPHAMASTER_NATIVE_FUSED_UNARY_UNARY_BRANCH", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return (
            first_unary_op_name in {"MAX3", "SIGMOID", "TANH_SQUASH", "SIGNED_POWER_2", "SIGNED_LOG"}
            and first_unary_op_name in NATIVE_UNARY_OPS
            and second_unary_op_name in {"TS_ZSCORE_10", "TS_ZSCORE_20"}
            and second_unary_op_name in NATIVE_ROLLING_OPS
            and branch_op_name in NATIVE_TERNARY_OPS
            and self.supports(first_unary_op_name, 1)
            and self.supports(second_unary_op_name, 1)
            and self.supports(branch_op_name, 3)
            and hasattr(self.ext, "fused_unary_unary_branch")
        )

    def apply_unary_unary_branch(
        self,
        first_unary_op_name: str,
        second_unary_op_name: str,
        branch_op_name: str,
        unary_arg: torch.Tensor,
        branch_condition: torch.Tensor,
        branch_true_value: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports_unary_unary_branch(first_unary_op_name, second_unary_op_name, branch_op_name):
            raise NotImplementedError(
                f"native fused unary+unary+branch not supported: {first_unary_op_name}->{second_unary_op_name}->{branch_op_name}"
            )
        return self.ext.fused_unary_unary_branch(
            unary_arg,
            branch_condition,
            branch_true_value,
            NATIVE_UNARY_OPS[first_unary_op_name],
            NATIVE_ROLLING_OPS[second_unary_op_name],
            NATIVE_TERNARY_OPS[branch_op_name],
        )

    def supports_rolling_binary_branch(self, rolling_op_name: str, binary_op_name: str, branch_op_name: str) -> bool:
        if os.getenv("ALPHAMASTER_NATIVE_FUSED_ROLLING_BINARY_BRANCH", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return (
            rolling_op_name in {"TS_MAX_10", "TS_MAX_20"}
            and rolling_op_name in NATIVE_ROLLING_OPS
            and binary_op_name in NATIVE_BINARY_OPS
            and branch_op_name in NATIVE_TERNARY_OPS
            and self.supports(rolling_op_name, 1)
            and self.supports(binary_op_name, 2)
            and self.supports(branch_op_name, 3)
            and hasattr(self.ext, "fused_rolling_binary_branch")
        )

    def apply_rolling_binary_branch(
        self,
        rolling_op_name: str,
        binary_op_name: str,
        branch_op_name: str,
        rolling_arg: torch.Tensor,
        binary_lhs: torch.Tensor,
        branch_condition: torch.Tensor,
        branch_true_value: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports_rolling_binary_branch(rolling_op_name, binary_op_name, branch_op_name):
            raise NotImplementedError(
                f"native fused rolling+binary+branch not supported: {rolling_op_name}->{binary_op_name}->{branch_op_name}"
            )
        return self.ext.fused_rolling_binary_branch(
            rolling_arg,
            binary_lhs,
            branch_condition,
            branch_true_value,
            NATIVE_ROLLING_OPS[rolling_op_name],
            NATIVE_BINARY_OPS[binary_op_name],
            NATIVE_TERNARY_OPS[branch_op_name],
        )

    def apply(self, op_name: str, *args: torch.Tensor) -> torch.Tensor:
        arity = len(args)
        if arity == 1 and op_name in NATIVE_UNARY_OPS:
            return self.ext.elementwise1(args[0], NATIVE_UNARY_OPS[op_name])
        if arity == 1 and op_name in NATIVE_SHIFT_OPS:
            return self.ext.shift1(args[0], NATIVE_SHIFT_OPS[op_name])
        if arity == 1 and op_name in NATIVE_ROLLING_OPS:
            return self.ext.rolling1(args[0], NATIVE_ROLLING_OPS[op_name])
        if arity == 1 and op_name in NATIVE_CROSS_SECTIONAL_OPS:
            return self.ext.cross_sectional1(args[0], NATIVE_CROSS_SECTIONAL_OPS[op_name])
        if arity == 2 and op_name in NATIVE_BINARY_OPS:
            return self.ext.elementwise2(args[0], args[1], NATIVE_BINARY_OPS[op_name])
        if arity == 2 and op_name in NATIVE_ROLLING_BINARY_OPS:
            return self.ext.rolling2(args[0], args[1], NATIVE_ROLLING_BINARY_OPS[op_name])
        if arity == 3 and op_name in NATIVE_TERNARY_OPS:
            return self.ext.elementwise3(args[0], args[1], args[2], NATIVE_TERNARY_OPS[op_name])
        raise NotImplementedError(f"native op not supported: {op_name}/{arity}")
