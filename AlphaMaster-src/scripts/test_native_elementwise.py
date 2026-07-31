"""Numerical smoke tests for the optional native CUDA extension."""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.native_backend import load_native_extension, probe_native_build


OP_ADD = 1
OP_SUB = 2
OP_MUL = 3
OP_DIV = 4
OP_IF_GT = 101
OP_GATE = 102


def main() -> None:
    status = probe_native_build()
    if not status.available:
        print(f"SKIP native build unavailable: {status.reason}")
        return
    ext = load_native_extension(verbose=False)
    torch.manual_seed(7)
    a = torch.randn(16, 3, 257, device="cuda")
    b = torch.randn_like(a)
    c = torch.randn_like(a)

    checks = [
        ("ADD", ext.elementwise2(a, b, OP_ADD), a + b),
        ("SUB", ext.elementwise2(a, b, OP_SUB), a - b),
        ("MUL", ext.elementwise2(a, b, OP_MUL), a * b),
        ("DIV", ext.elementwise2(a, b, OP_DIV), a / (b + 1e-6)),
        ("IF_GT", ext.elementwise3(a, b, c, OP_IF_GT), torch.where(a > 0, b, c)),
        ("GATE", ext.elementwise3(a, b, c, OP_GATE), (a > 0).float() * b + (a <= 0).float() * c),
    ]
    for name, got, expected in checks:
        diff = (got - torch.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)).abs().max().item()
        print(name, "max_diff", diff)
        assert diff <= 1e-5, (name, diff)
    print("native_elementwise_ok")


if __name__ == "__main__":
    main()
