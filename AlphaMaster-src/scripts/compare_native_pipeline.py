"""Compare full formula pipeline with native formula ops on and off."""
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

from model_core.gpu_batch import BatchFormulaPipeline
from model_core.batch_ops import BATCH_OPS_CONFIG
from model_core.vocab import FORMULA_VOCAB
from scripts.benchmark_native_vm import _sample_formulas


def _sync() -> None:
    torch.cuda.synchronize()


def _top_stable(ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    ref_idx = int(ref.argmax().item())
    got_idx = int(got.argmax().item())
    if ref_idx == got_idx:
        return True
    return abs(float(ref[ref_idx].item()) - float(ref[got_idx].item())) <= tol


def _token_name(token: int) -> str:
    if token < FORMULA_VOCAB.operator_offset:
        return f"F{token}:{FORMULA_VOCAB.feature_names[token]}"
    idx = token - FORMULA_VOCAB.operator_offset
    if 0 <= idx < len(BATCH_OPS_CONFIG):
        return BATCH_OPS_CONFIG[idx][0]
    return f"OP?{token}"


def _formula_names(formula: torch.Tensor) -> str:
    return " -> ".join(_token_name(int(tok)) for tok in formula.detach().cpu().tolist())


def _timed(fn, repeat: int) -> tuple[object, list[float]]:
    out = None
    times: list[float] = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--bars", type=int, default=5555)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--score-tol", type=float, default=1e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    device = torch.device("cuda")
    torch.manual_seed(321)
    feat = torch.randn(args.symbols, args.features, args.bars, device=device)
    target_ret = torch.randn(args.symbols, args.bars, device=device) * 0.002
    formulas = _sample_formulas(args.batch, args.length, args.features, device)
    split = int(args.bars * 0.8)
    train_start, train_end = 0, max(10, split // 2)
    val_start, val_end = train_end, split

    os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "0"
    ref_pipe = BatchFormulaPipeline()
    ref, ref_times = _timed(
        lambda: ref_pipe.evaluate_fold_batch(formulas, feat, target_ret, train_start, train_end, val_start, val_end),
        args.repeat,
    )

    os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "1"
    native_pipe = BatchFormulaPipeline()
    got, got_times = _timed(
        lambda: native_pipe.evaluate_fold_batch(formulas, feat, target_ret, train_start, train_end, val_start, val_end),
        args.repeat,
    )

    assert ref is not None and got is not None
    factor_diff = (ref.factors - got.factors).abs().max().item()
    train_diff = (ref.train_scores - got.train_scores).abs().max().item()
    val_diff = (ref.val_scores - got.val_scores).abs().max().item()
    sortino_diff = (ref.oos_sortino - got.oos_sortino).abs().max().item()
    valid_equal = bool(torch.equal(ref.valid, got.valid))
    train_top = _top_stable(ref.train_scores, got.train_scores, args.score_tol)
    val_top = _top_stable(ref.val_scores, got.val_scores, args.score_tol)

    print(f"shape=({args.batch},{args.symbols},{args.bars}) valid_equal={valid_equal}")
    print(f"factor_max_diff={factor_diff:.9g}")
    print(f"train_score_max_diff={train_diff:.9g}")
    print(f"val_score_max_diff={val_diff:.9g}")
    print(f"oos_sortino_max_diff={sortino_diff:.9g}")
    for label, ref_scores, got_scores in (
        ("train", ref.train_scores, got.train_scores),
        ("val", ref.val_scores, got.val_scores),
        ("oos_sortino", ref.oos_sortino, got.oos_sortino),
    ):
        delta = (ref_scores - got_scores).abs()
        idx = int(delta.argmax().item())
        print(
            f"{label}_max_idx={idx} ref={float(ref_scores[idx].item()):.9g} "
            f"native={float(got_scores[idx].item()):.9g} diff={float(delta[idx].item()):.9g}"
        )
        print(f"{label}_max_formula={_formula_names(formulas[idx])}")
    print(f"train_top_stable={train_top} val_top_stable={val_top}")
    print(f"torch_pipeline_ms avg={statistics.mean(ref_times):.6f} p50={statistics.median(ref_times):.6f}")
    print(f"native_pipeline_ms avg={statistics.mean(got_times):.6f} p50={statistics.median(got_times):.6f}")
    print(f"native_pipeline_speedup={statistics.mean(ref_times) / statistics.mean(got_times):.3f}x")

    if not valid_equal:
        raise AssertionError("native valid mask changed")
    if train_diff > args.score_tol or val_diff > args.score_tol or sortino_diff > args.score_tol:
        raise AssertionError("native score diff exceeds tolerance")
    if not train_top or not val_top:
        raise AssertionError("native top choice is unstable")


if __name__ == "__main__":
    main()
