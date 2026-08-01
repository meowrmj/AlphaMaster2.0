"""Benchmark BatchStackVM3D with optional native formula ops."""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.gpu_batch import BatchStackVM3D
from model_core.batch_ops import BATCH_OPS_CONFIG
from model_core.vocab import FORMULA_VOCAB


def _time_cuda(fn, repeat: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _sample_formulas(batch: int, length: int, feature_count: int, device: torch.device) -> torch.Tensor:
    ops_by_arity: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for idx, (_name, _fn, arity) in enumerate(BATCH_OPS_CONFIG):
        ops_by_arity[arity].append(FORMULA_VOCAB.operator_offset + idx)
    formulas: list[list[int]] = []
    gen = torch.Generator(device="cpu").manual_seed(123)
    for _ in range(batch):
        stack_depth = 0
        tokens: list[int] = []
        for step in range(length):
            remaining = length - step
            if stack_depth == 0 or (remaining > 1 and torch.rand((), generator=gen).item() < 0.45):
                tokens.append(int(torch.randint(0, feature_count, (), generator=gen).item()))
                stack_depth += 1
                continue
            choices = [arity for arity in (1, 2, 3) if stack_depth >= arity]
            arity = int(choices[int(torch.randint(0, len(choices), (), generator=gen).item())])
            op_tokens = ops_by_arity[arity]
            tokens.append(int(op_tokens[int(torch.randint(0, len(op_tokens), (), generator=gen).item())]))
            stack_depth = stack_depth - arity + 1
        while stack_depth != 1:
            tokens[-1] = ops_by_arity[1][0]
            stack_depth = 1
        formulas.append(tokens)
    return torch.tensor(formulas, dtype=torch.long, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--bars", type=int, default=5555)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--length", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP CUDA unavailable")
        return
    device = torch.device("cuda")
    torch.manual_seed(321)
    feat = torch.randn(args.symbols, args.features, args.bars, device=device)
    formulas = _sample_formulas(args.batch, args.length, args.features, device)

    os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "0"
    torch_vm = BatchStackVM3D()
    expected_factors, expected_valid = torch_vm.execute_batch(formulas, feat)

    os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "1"
    native_vm = BatchStackVM3D()
    got_factors, got_valid = native_vm.execute_batch(formulas, feat)

    valid_equal = bool(torch.equal(expected_valid, got_valid))
    max_diff = (expected_factors - got_factors).abs().max().item()
    print(f"shape=({args.batch},{args.symbols},{args.bars}) valid_equal={valid_equal} max_diff={max_diff}")
    if not valid_equal or max_diff > 1e-4:
        raise AssertionError(f"native VM mismatch valid_equal={valid_equal} max_diff={max_diff}")

    os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "0"
    torch_times = _time_cuda(lambda: torch_vm.execute_batch(formulas, feat), args.repeat)
    os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "1"
    native_times = _time_cuda(lambda: native_vm.execute_batch(formulas, feat), args.repeat)
    print(f"torch_vm_ms avg={statistics.mean(torch_times):.6f} p50={statistics.median(torch_times):.6f}")
    print(f"native_vm_ms avg={statistics.mean(native_times):.6f} p50={statistics.median(native_times):.6f}")
    print(f"native_vm_speedup={statistics.mean(torch_times) / statistics.mean(native_times):.3f}x")


if __name__ == "__main__":
    main()
