"""Find the first bitwise mismatch between PyTorch batch ops and native ops."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.batch_ops import BATCH_OPS_CONFIG
from model_core.gpu_batch import BatchStackVM3D
from model_core.native_backend import NativeElementwiseOps
from model_core.vocab import FORMULA_VOCAB
from scripts.benchmark_native_vm import _sample_formulas


@dataclass(frozen=True)
class StepResult:
    name: str
    value: torch.Tensor
    inputs: tuple[torch.Tensor, ...] = ()


def _token_name(token: int) -> str:
    if token < FORMULA_VOCAB.operator_offset:
        return f"F{token}:{FORMULA_VOCAB.feature_names[token]}"
    op_idx = token - FORMULA_VOCAB.operator_offset
    if 0 <= op_idx < len(BATCH_OPS_CONFIG):
        return BATCH_OPS_CONFIG[op_idx][0]
    return f"OP?{token}"


def _formula_names(formula: torch.Tensor) -> list[str]:
    return [_token_name(int(tok)) for tok in formula.detach().cpu().tolist()]


def _first_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float, float]:
    mask = a != b
    flat_idx = int(mask.reshape(-1).nonzero(as_tuple=False)[0].item())
    return flat_idx, float(a.reshape(-1)[flat_idx].item()), float(b.reshape(-1)[flat_idx].item())


def _run_trace(
    formula: torch.Tensor,
    feat: torch.Tensor,
    *,
    native: NativeElementwiseOps | None,
    force_native_ops: set[str] | None = None,
) -> tuple[list[StepResult], bool]:
    stack: list[torch.Tensor] = []
    trace: list[StepResult] = []
    valid = True
    for token_t in formula:
        token = int(token_t.item())
        if token < FORMULA_VOCAB.operator_offset:
            if token >= feat.shape[1]:
                valid = False
                break
            value = feat[:, token, :].unsqueeze(0).clone()
            stack.append(value)
            trace.append(StepResult(_token_name(token), value.clone()))
            continue

        op_idx = token - FORMULA_VOCAB.operator_offset
        if not (0 <= op_idx < len(BATCH_OPS_CONFIG)):
            valid = False
            break
        op_name, op_func, arity = BATCH_OPS_CONFIG[op_idx]
        if len(stack) < arity:
            valid = False
            break
        args = stack[-arity:]
        del stack[-arity:]
        use_native = native is not None and native.supports(op_name, arity)
        if force_native_ops is not None:
            use_native = use_native and op_name in force_native_ops
        input_snapshot = tuple(arg.clone() for arg in args)
        value = native.apply(op_name, *args) if use_native else op_func(*args)
        value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
        stack.append(value)
        trace.append(StepResult(op_name, value.clone(), input_snapshot))
    if len(stack) != 1:
        valid = False
    return trace, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--bars", type=int, default=5555)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--formula-index", type=int)
    parser.add_argument("--op")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda")
    torch.manual_seed(321)
    feat = torch.randn(args.symbols, args.features, args.bars, device=device)
    formulas = _sample_formulas(args.batch, args.length, args.features, device)
    native = NativeElementwiseOps(verbose=False)
    force_ops = {args.op} if args.op else None

    formula_indices = [args.formula_index] if args.formula_index is not None else list(range(formulas.shape[0]))
    for formula_idx in formula_indices:
        formula = formulas[formula_idx]
        torch_trace, torch_valid = _run_trace(formula, feat, native=None)
        native_trace, native_valid = _run_trace(formula, feat, native=native, force_native_ops=force_ops)
        if torch_valid != native_valid:
            print(f"VALID_MISMATCH formula={formula_idx} torch={torch_valid} native={native_valid}")
            print("formula:", " -> ".join(_formula_names(formula)))
            raise SystemExit(1)
        for step, (ref, got) in enumerate(zip(torch_trace, native_trace)):
            if not torch.equal(ref.value, got.value):
                torch.cuda.synchronize()
                flat_idx, ref_v, got_v = _first_diff(ref.value, got.value)
                diff = (ref.value - got.value).abs().max().item()
                print("BITWISE_MISMATCH")
                print(f"formula_index={formula_idx}")
                print("formula:", " -> ".join(_formula_names(formula)))
                print(f"step={step} op={ref.name} max_abs_diff={diff}")
                print(f"first_flat_index={flat_idx} torch={ref_v!r} native={got_v!r}")
                print(f"shape={tuple(ref.value.shape)} symbols={args.symbols} bars={args.bars}")
                if ref.inputs:
                    input_tensor = ref.inputs[0]
                    _, sym, bar = torch.unravel_index(
                        torch.tensor(flat_idx, device=input_tensor.device),
                        ref.value.shape,
                    )
                    sym_i = int(sym.item())
                    bar_i = int(bar.item())
                    start = max(0, bar_i - 6)
                    end = min(input_tensor.shape[-1], bar_i + 2)
                    window = input_tensor[0, sym_i, start:end].detach().cpu().tolist()
                    print(f"input_window bars[{start}:{end}]={window!r}")
                raise SystemExit(1)

        os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "0"
        ref_factors, ref_valid = BatchStackVM3D().execute_batch(formula.view(1, -1), feat)
        os.environ["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "1"
        got_factors, got_valid = BatchStackVM3D().execute_batch(formula.view(1, -1), feat)
        if not torch.equal(ref_valid, got_valid) or not torch.equal(ref_factors, got_factors):
            flat_idx, ref_v, got_v = _first_diff(ref_factors, got_factors)
            diff = (ref_factors - got_factors).abs().max().item()
            print("FINAL_FACTOR_MISMATCH")
            print(f"formula_index={formula_idx}")
            print("formula:", " -> ".join(_formula_names(formula)))
            print(f"max_abs_diff={diff}")
            print(f"first_flat_index={flat_idx} torch={ref_v!r} native={got_v!r}")
            raise SystemExit(1)

    print(f"BITWISE_OK formulas={len(formula_indices)} symbols={args.symbols} bars={args.bars}")


if __name__ == "__main__":
    main()
