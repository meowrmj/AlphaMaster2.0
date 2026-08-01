"""Numerical smoke tests for the optional native CUDA extension."""
from __future__ import annotations

import pathlib
import sys

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


def main() -> None:
    status = probe_native_build()
    if not status.available:
        print(f"SKIP native build unavailable: {status.reason}")
        return
    native = NativeElementwiseOps(verbose=False)
    torch.manual_seed(7)
    a = torch.randn(16, 3, 257, device="cuda")
    b = torch.randn_like(a)
    c = torch.randn_like(a)
    flat = torch.ones_like(a) * 0.25

    checks = [
        ("NEG", native.apply("NEG", a), -a),
        ("ABS", native.apply("ABS", a), torch.abs(a)),
        ("SIGN", native.apply("SIGN", a), torch.sign(a)),
        ("POWER", native.apply("POWER", a), torch.sign(a) * torch.abs(a).pow(2.0)),
        ("SIGNED_POWER_2", native.apply("SIGNED_POWER_2", a), torch.sign(a) * torch.abs(a).pow(2.0)),
        ("SIGNED_LOG", native.apply("SIGNED_LOG", a), torch.sign(a) * torch.log1p(torch.abs(a))),
        ("SQRT", native.apply("SQRT", a), torch.sign(a) * torch.sqrt(torch.abs(a))),
        ("CLIP", native.apply("CLIP", a), torch.clamp(a, -3.0, 3.0)),
        ("SIGMOID", native.apply("SIGMOID", a), 2 * torch.sigmoid(a) - 1),
        ("TANH_SQUASH", native.apply("TANH_SQUASH", a), torch.tanh(a)),
        ("ADD", native.apply("ADD", a, b), a + b),
        ("SUB", native.apply("SUB", a, b), a - b),
        ("MUL", native.apply("MUL", a, b), a * b),
        ("DIV", native.apply("DIV", a, b), a / (b + 1e-6)),
        ("IF_GT", native.apply("IF_GT", a, b, c), torch.where(a > 0, b, c)),
        ("GATE", native.apply("GATE", a, b, c), (a > 0).float() * b + (a <= 0).float() * c),
        ("DELAY1", native.apply("DELAY1", a), _batch_op("DELAY1")(a)),
        ("DELAY4", native.apply("DELAY4", a), _batch_op("DELAY4")(a)),
        ("DELTA", native.apply("DELTA", a), _batch_op("DELTA")(a)),
        ("DELTA_5", native.apply("DELTA_5", a), _batch_op("DELTA_5")(a)),
        ("TS_MEAN_5", native.apply("TS_MEAN_5", a), _batch_op("TS_MEAN_5")(a)),
        ("TS_MEAN_10", native.apply("TS_MEAN_10", a), _batch_op("TS_MEAN_10")(a)),
        ("TS_MEAN_20", native.apply("TS_MEAN_20", a), _batch_op("TS_MEAN_20")(a)),
        ("TS_SUM_5", native.apply("TS_SUM_5", a), _batch_op("TS_SUM_5")(a)),
        ("TS_SUM_10", native.apply("TS_SUM_10", a), _batch_op("TS_SUM_10")(a)),
        ("TS_SUM_20", native.apply("TS_SUM_20", a), _batch_op("TS_SUM_20")(a)),
        ("TS_ZSCORE_10", native.apply("TS_ZSCORE_10", a), _batch_op("TS_ZSCORE_10")(a)),
        ("TS_ZSCORE_20", native.apply("TS_ZSCORE_20", a), _batch_op("TS_ZSCORE_20")(a)),
        ("WINSORIZE", native.apply("WINSORIZE", a), _batch_op("WINSORIZE")(a)),
        ("TS_STD_5", native.apply("TS_STD_5", a), _batch_op("TS_STD_5")(a)),
        ("TS_STD_10", native.apply("TS_STD_10", a), _batch_op("TS_STD_10")(a)),
        ("TS_STD_20", native.apply("TS_STD_20", a), _batch_op("TS_STD_20")(a)),
        ("TS_RANK_5", native.apply("TS_RANK_5", a), _batch_op("TS_RANK_5")(a)),
        ("TS_RANK_10", native.apply("TS_RANK_10", a), _batch_op("TS_RANK_10")(a)),
        ("TS_RANK_20", native.apply("TS_RANK_20", a), _batch_op("TS_RANK_20")(a)),
        ("TS_MIN_10", native.apply("TS_MIN_10", a), _batch_op("TS_MIN_10")(a)),
        ("TS_MIN_20", native.apply("TS_MIN_20", a), _batch_op("TS_MIN_20")(a)),
        ("TS_MAX_10", native.apply("TS_MAX_10", a), _batch_op("TS_MAX_10")(a)),
        ("TS_MAX_20", native.apply("TS_MAX_20", a), _batch_op("TS_MAX_20")(a)),
        ("TS_QUANTILE_10", native.apply("TS_QUANTILE_10", a), _batch_op("TS_QUANTILE_10")(a)),
        ("TS_SKEW_10", native.apply("TS_SKEW_10", a), _batch_op("TS_SKEW_10")(a)),
        ("TS_ARG_MAX_5", native.apply("TS_ARG_MAX_5", a), _batch_op("TS_ARG_MAX_5")(a)),
        ("TS_ARG_MIN_5", native.apply("TS_ARG_MIN_5", a), _batch_op("TS_ARG_MIN_5")(a)),
        ("DECAY", native.apply("DECAY", a), _batch_op("DECAY")(a)),
        ("WMA", native.apply("WMA", a), _batch_op("WMA")(a)),
        ("DECAY_LINEAR_5", native.apply("DECAY_LINEAR_5", a), _batch_op("DECAY_LINEAR_5")(a)),
        ("TS_DECAY_EXP_5", native.apply("TS_DECAY_EXP_5", a), _batch_op("TS_DECAY_EXP_5")(a)),
        ("EMA_5", native.apply("EMA_5", a), _batch_op("EMA_5")(a)),
        ("EMA_20", native.apply("EMA_20", a), _batch_op("EMA_20")(a)),
        ("MOMENTUM_5", native.apply("MOMENTUM_5", a), _batch_op("MOMENTUM_5")(a)),
        ("MOMENTUM_10", native.apply("MOMENTUM_10", a), _batch_op("MOMENTUM_10")(a)),
        ("MAX3", native.apply("MAX3", a), _batch_op("MAX3")(a)),
        ("CS_SCALE", native.apply("CS_SCALE", a), _batch_op("CS_SCALE")(a)),
        ("CS_NEUTRALIZE", native.apply("CS_NEUTRALIZE", a), _batch_op("CS_NEUTRALIZE")(a)),
        ("TS_CORR_10", native.apply("TS_CORR_10", a, b), _batch_op("TS_CORR_10")(a, b)),
        ("COVARIANCE_10", native.apply("COVARIANCE_10", a, b), _batch_op("COVARIANCE_10")(a, b)),
    ]
    for name, got, expected in checks:
        diff = (got - torch.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)).abs().max().item()
        print(name, "max_diff", diff)
        assert diff <= 1e-5, (name, diff)
    for name in ("TS_ZSCORE_10", "TS_ZSCORE_20"):
        got = native.apply(name, flat)
        expected = _batch_op(name)(flat)
        diff = (got - torch.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)).abs().max().item()
        print(name, "flat_max_diff", diff)
        assert diff <= 1e-5, (name, diff)
    print("native_elementwise_ok")


if __name__ == "__main__":
    main()
