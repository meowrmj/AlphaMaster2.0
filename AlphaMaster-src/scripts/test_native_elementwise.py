"""Numerical smoke tests for the optional native CUDA extension."""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.native_backend import NativeElementwiseOps, probe_native_build


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

    checks = [
        ("ADD", native.apply("ADD", a, b), a + b),
        ("SUB", native.apply("SUB", a, b), a - b),
        ("MUL", native.apply("MUL", a, b), a * b),
        ("DIV", native.apply("DIV", a, b), a / (b + 1e-6)),
        ("IF_GT", native.apply("IF_GT", a, b, c), torch.where(a > 0, b, c)),
        ("GATE", native.apply("GATE", a, b, c), (a > 0).float() * b + (a <= 0).float() * c),
    ]
    for name, got, expected in checks:
        diff = (got - torch.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)).abs().max().item()
        print(name, "max_diff", diff)
        assert diff <= 1e-5, (name, diff)
    print("native_elementwise_ok")


if __name__ == "__main__":
    main()
