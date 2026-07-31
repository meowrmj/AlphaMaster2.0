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
    parser.add_argument("--op", default="ADD", choices=["ADD", "SUB", "MUL", "DIV", "MAX", "MIN", "IF_GT", "GATE"])
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

    if args.op == "ADD":
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
