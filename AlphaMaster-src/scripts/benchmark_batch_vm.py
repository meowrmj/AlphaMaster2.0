"""Compare scalar StackVM execution with experimental BatchStackVM."""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.gpu_batch import BatchStackVM3D
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
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--features", type=int, default=FORMULA_VOCAB.feature_count)
    parser.add_argument("--bars", type=int, default=7247)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")
    else:
        device = torch.device(args.device)

    torch.manual_seed(11)
    feat = torch.randn(args.symbols, args.features, args.bars, device=device)
    formulas = sample_formulas(args.batch)
    formula_t = torch.tensor(formulas, dtype=torch.long, device=device)

    scalar_vm = StackVM()
    batch_vm = BatchStackVM3D()

    sync(device)
    t0 = time.perf_counter()
    scalar_results = []
    scalar_valid = []
    for fml in formulas:
        res = scalar_vm.execute(fml, feat)
        scalar_valid.append(res is not None)
        scalar_results.append(torch.zeros(args.symbols, args.bars, device=device) if res is None else res)
    scalar_t = torch.stack(scalar_results, dim=0)
    scalar_valid_t = torch.tensor(scalar_valid, dtype=torch.bool, device=device)
    sync(device)
    scalar_s = time.perf_counter() - t0

    sync(device)
    t1 = time.perf_counter()
    batch_t, batch_valid = batch_vm.execute_batch(formula_t, feat)
    sync(device)
    batch_s = time.perf_counter() - t1

    diff = (scalar_t - batch_t).abs().max().item()
    valid_equal = bool(torch.equal(scalar_valid_t, batch_valid))
    speedup = scalar_s / batch_s if batch_s > 0 else float("inf")
    print(f"device={device}")
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
    print(f"shape: formulas=[{args.batch},8] feat=[{args.symbols},{args.features},{args.bars}]")
    print(f"scalar_vm_seconds={scalar_s:.4f}")
    print(f"batch_vm_seconds={batch_s:.4f}")
    print(f"speedup={speedup:.2f}x")
    print(f"valid_equal={valid_equal}")
    print(f"max_abs_factor_diff={diff:.6f}")


if __name__ == "__main__":
    main()
