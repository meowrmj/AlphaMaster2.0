"""Benchmark complete scalar formula evaluation vs batch formula pipeline."""
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
from model_core.gpu_batch import BatchFormulaPipeline
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB


def token_id(name: str) -> int:
    return FORMULA_VOCAB.token_names.index(name)


def sample_formulas(batch: int) -> list[list[int]]:
    seeds = [
        ["SUPERTREND_DIR", "EMA_20", "TS_SUM_10", "MUL", "SUPERTREND_DIR", "EMA_20", "TS_SUM_10", "MUL"],
        ["HURST_50", "TS_MEAN_20", "SUPERTREND_DIR", "SIGNED_POWER_2", "TS_QUANTILE_10", "TS_STD_10", "SUB", "CS_RANK"],
        ["SLOPE20", "TS_MIN_10", "TANH_SQUASH", "MACD_HIST", "CLIP", "TRIX_SIGNAL", "ADD", "SUB"],
        ["BOLL_WIDTH", "FRACTAL_DIM_30", "TS_STD_10", "MAX", "REL_VOL", "ABS", "CMF_20", "GATE"],
    ]
    toks = [[token_id(x) for x in row] for row in seeds]
    return [toks[i % len(toks)] for i in range(batch)]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--features", type=int, default=FORMULA_VOCAB.feature_count)
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

    torch.manual_seed(17)
    feat = torch.randn(1, args.features, args.bars, device=device)
    target_ret = torch.randn(1, args.bars, device=device) * 0.002
    split = int(args.bars * 0.8)
    train_start, train_end = 0, max(10, split // 2)
    val_start, val_end = train_end, split
    formulas = sample_formulas(args.batch)
    formula_t = torch.tensor(formulas, dtype=torch.long, device=device)

    scalar_vm = StackVM()
    scalar_bt = MT5Backtest(periods_per_year=args.periods_per_year)
    pipeline = BatchFormulaPipeline(
        cost_rate=scalar_bt.cost_rate,
        periods_per_year=args.periods_per_year,
    )

    sync(device)
    t0 = time.perf_counter()
    scalar_train = []
    scalar_val = []
    scalar_valid = []
    for fml in formulas:
        res = scalar_vm.execute(fml, feat)
        scalar_valid.append(res is not None)
        if res is None:
            scalar_train.append(torch.tensor(-5.0, dtype=feat.dtype, device=device))
            scalar_val.append(torch.tensor(-5.0, dtype=feat.dtype, device=device))
            continue
        tr, vl = scalar_bt.evaluate_fold(
            res,
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
    scalar_valid_t = torch.tensor(scalar_valid, dtype=torch.bool, device=device)
    sync(device)
    scalar_s = time.perf_counter() - t0

    sync(device)
    t1 = time.perf_counter()
    batch_res = pipeline.evaluate_fold_batch(
        formula_t,
        feat,
        target_ret,
        train_start,
        train_end,
        val_start,
        val_end,
    )
    sync(device)
    batch_s = time.perf_counter() - t1

    train_diff = (scalar_train_t - batch_res.train_scores).abs().max().item()
    val_diff = (scalar_val_t - batch_res.val_scores).abs().max().item()
    speedup = scalar_s / batch_s if batch_s > 0 else float("inf")

    print(f"device={device}")
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
    print(f"shape: formulas=[{args.batch},8] feat=[1,{args.features},{args.bars}]")
    print(f"scalar_pipeline_seconds={scalar_s:.4f}")
    print(f"batch_pipeline_seconds={batch_s:.4f}")
    print(f"speedup={speedup:.2f}x")
    print(f"valid_equal={bool(torch.equal(scalar_valid_t, batch_res.valid))}")
    print(f"max_abs_train_diff={train_diff:.6f}")
    print(f"max_abs_val_diff={val_diff:.6f}")


if __name__ == "__main__":
    main()
