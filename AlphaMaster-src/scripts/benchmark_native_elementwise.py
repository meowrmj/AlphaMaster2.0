"""Benchmark optional native elementwise CUDA kernels against PyTorch ops."""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.native_backend import NativeElementwiseOps, probe_native_build
from model_core.batch_ops import BATCH_OPS_CONFIG


def _batch_op(name: str):
    for op_name, func, _arity in BATCH_OPS_CONFIG:
        if op_name == name:
            return func
    raise KeyError(name)


def _time_cuda(fn, repeat: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--bars", type=int, default=5555)
    parser.add_argument(
        "--op",
        default="ADD",
        choices=[
            "NEG", "ABS", "SIGN", "POWER", "SIGNED_POWER_2", "SIGNED_LOG", "SQRT",
            "CLIP", "SIGMOID", "TANH_SQUASH",
            "ADD", "SUB", "MUL", "DIV", "MAX", "MIN", "IF_GT", "GATE",
            "DELAY1", "DELAY4", "DELTA", "DELTA_5",
            "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
            "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
            "TS_ZSCORE_10", "TS_ZSCORE_20",
            "WINSORIZE",
            "TS_STD_5", "TS_STD_10", "TS_STD_20",
            "TS_RANK_5", "TS_RANK_10", "TS_RANK_20",
            "TS_MIN_10", "TS_MIN_20", "TS_MAX_10", "TS_MAX_20",
            "TS_QUANTILE_10", "TS_SKEW_10", "TS_ARG_MAX_5", "TS_ARG_MIN_5",
            "DECAY", "WMA", "DECAY_LINEAR_5", "TS_DECAY_EXP_5",
            "EMA_5", "EMA_20", "MOMENTUM_5", "MOMENTUM_10",
            "MAX3",
            "TS_CORR_10", "COVARIANCE_10",
        ],
    )
    args = parser.parse_args()

    status = probe_native_build()
    if not status.available:
        print(f"SKIP native build unavailable: {status.reason}")
        return

    native = NativeElementwiseOps()
    torch.manual_seed(17)
    shape = (args.batch, args.symbols, args.bars)
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")
    c = torch.randn(shape, device="cuda")

    if args.op == "NEG":
        torch_fn = lambda: -a
        native_fn = lambda: native.apply("NEG", a)
    elif args.op == "ABS":
        torch_fn = lambda: torch.abs(a)
        native_fn = lambda: native.apply("ABS", a)
    elif args.op == "SIGN":
        torch_fn = lambda: torch.sign(a)
        native_fn = lambda: native.apply("SIGN", a)
    elif args.op == "POWER":
        torch_fn = lambda: torch.sign(a) * torch.abs(a).pow(2.0)
        native_fn = lambda: native.apply("POWER", a)
    elif args.op == "SIGNED_POWER_2":
        torch_fn = lambda: torch.sign(a) * torch.abs(a).pow(2.0)
        native_fn = lambda: native.apply("SIGNED_POWER_2", a)
    elif args.op == "SIGNED_LOG":
        torch_fn = lambda: torch.sign(a) * torch.log1p(torch.abs(a))
        native_fn = lambda: native.apply("SIGNED_LOG", a)
    elif args.op == "SQRT":
        torch_fn = lambda: torch.sign(a) * torch.sqrt(torch.abs(a))
        native_fn = lambda: native.apply("SQRT", a)
    elif args.op == "CLIP":
        torch_fn = lambda: torch.clamp(a, -3.0, 3.0)
        native_fn = lambda: native.apply("CLIP", a)
    elif args.op == "SIGMOID":
        torch_fn = lambda: 2 * torch.sigmoid(a) - 1
        native_fn = lambda: native.apply("SIGMOID", a)
    elif args.op == "TANH_SQUASH":
        torch_fn = lambda: torch.tanh(a)
        native_fn = lambda: native.apply("TANH_SQUASH", a)
    elif args.op in {
        "DELAY1", "DELAY4", "DELTA", "DELTA_5",
        "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
        "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
        "TS_ZSCORE_10", "TS_ZSCORE_20",
        "WINSORIZE",
        "TS_STD_5", "TS_STD_10", "TS_STD_20",
        "TS_RANK_5", "TS_RANK_10", "TS_RANK_20",
        "TS_MIN_10", "TS_MIN_20", "TS_MAX_10", "TS_MAX_20",
        "TS_QUANTILE_10", "TS_SKEW_10", "TS_ARG_MAX_5", "TS_ARG_MIN_5",
        "DECAY", "WMA", "DECAY_LINEAR_5", "TS_DECAY_EXP_5",
        "EMA_5", "EMA_20", "MOMENTUM_5", "MOMENTUM_10",
        "MAX3",
    }:
        torch_op = _batch_op(args.op)
        torch_fn = lambda: torch_op(a)
        native_fn = lambda: native.apply(args.op, a)
    elif args.op in {"TS_CORR_10", "COVARIANCE_10"}:
        torch_op = _batch_op(args.op)
        torch_fn = lambda: torch_op(a, b)
        native_fn = lambda: native.apply(args.op, a, b)
    elif args.op == "ADD":
        torch_fn = lambda: a + b
        native_fn = lambda: native.apply("ADD", a, b)
    elif args.op == "SUB":
        torch_fn = lambda: a - b
        native_fn = lambda: native.apply("SUB", a, b)
    elif args.op == "MUL":
        torch_fn = lambda: a * b
        native_fn = lambda: native.apply("MUL", a, b)
    elif args.op == "DIV":
        torch_fn = lambda: a / (b + 1e-6)
        native_fn = lambda: native.apply("DIV", a, b)
    elif args.op == "MAX":
        torch_fn = lambda: torch.maximum(a, b)
        native_fn = lambda: native.apply("MAX", a, b)
    elif args.op == "MIN":
        torch_fn = lambda: torch.minimum(a, b)
        native_fn = lambda: native.apply("MIN", a, b)
    elif args.op == "IF_GT":
        torch_fn = lambda: torch.where(a > 0, b, c)
        native_fn = lambda: native.apply("IF_GT", a, b, c)
    else:
        torch_fn = lambda: (a > 0).float() * b + (a <= 0).float() * c
        native_fn = lambda: native.apply("GATE", a, b, c)

    got = native_fn()
    expected = torch.nan_to_num(torch_fn(), nan=0.0, posinf=0.0, neginf=0.0)
    max_diff = (got - expected).abs().max().item()
    if max_diff > 1e-5:
        raise AssertionError(f"{args.op} max_diff too high: {max_diff}")

    torch_times = _time_cuda(torch_fn, args.repeat)
    native_times = _time_cuda(native_fn, args.repeat)
    print(f"shape={shape} op={args.op} max_diff={max_diff}")
    print(f"torch_ms avg={statistics.mean(torch_times):.6f} p50={statistics.median(torch_times):.6f}")
    print(f"native_ms avg={statistics.mean(native_times):.6f} p50={statistics.median(native_times):.6f}")
    speedup = statistics.mean(torch_times) / statistics.mean(native_times)
    print(f"native_speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
