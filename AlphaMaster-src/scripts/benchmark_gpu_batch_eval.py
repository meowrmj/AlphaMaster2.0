"""Benchmark the experimental batch backtest scorer.

This does not start or stop live training.  It compares the current scalar
MT5Backtest.evaluate_fold loop with the new vectorized BatchBacktestEvaluator on
synthetic candidate factors shaped like one training step.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.backtest import MT5Backtest
from model_core.gpu_batch import BatchBacktestEvaluator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--bars", type=int, default=7247)
    parser.add_argument("--periods-per-year", type=int, default=3612)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")
    else:
        device = torch.device(args.device)

    torch.manual_seed(7)
    factors = torch.randn(args.batch, args.symbols, args.bars, device=device)
    target_ret = torch.randn(args.symbols, args.bars, device=device) * 0.002
    split = int(args.bars * 0.8)
    train_start, train_end = 0, max(10, split // 2)
    val_start, val_end = train_end, split

    scalar = MT5Backtest(periods_per_year=args.periods_per_year)
    batch = BatchBacktestEvaluator(
        cost_rate=scalar.cost_rate,
        periods_per_year=args.periods_per_year,
    )

    _sync(device)
    t0 = time.perf_counter()
    scalar_train = []
    scalar_val = []
    for i in range(args.batch):
        tr, vl = scalar.evaluate_fold(
            factors[i],
            target_ret,
            train_start,
            train_end,
            val_start,
            val_end,
        )
        scalar_train.append(tr)
        scalar_val.append(vl)
    scalar_train_t = torch.stack(scalar_train)
    scalar_val_t = torch.stack(scalar_val)
    _sync(device)
    scalar_s = time.perf_counter() - t0

    _sync(device)
    t1 = time.perf_counter()
    batch_res = batch.evaluate_fold_batch(
        factors,
        target_ret,
        train_start,
        train_end,
        val_start,
        val_end,
    )
    _sync(device)
    batch_s = time.perf_counter() - t1

    train_diff = (scalar_train_t - batch_res.train_scores).abs().max().item()
    val_diff = (scalar_val_t - batch_res.val_scores).abs().max().item()
    speedup = scalar_s / batch_s if batch_s > 0 else float("inf")

    print(f"device={device}")
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"shape: factors=[{args.batch},{args.symbols},{args.bars}]")
    print(f"scalar_loop_seconds={scalar_s:.4f}")
    print(f"batch_eval_seconds={batch_s:.4f}")
    print(f"speedup={speedup:.2f}x")
    print(f"max_abs_train_diff={train_diff:.6f}")
    print(f"max_abs_val_diff={val_diff:.6f}")


if __name__ == "__main__":
    main()
