"""Find real formula fragments worth fusing into native CUDA chain kernels.

The output is planning evidence only. It reads recent checkpoints, extracts
formula tokens, and counts consecutive operator runs that can be executed by a
native policy. It does not modify checkpoints or start training.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys
from collections.abc import Iterable

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.kernel_planner import classify_operator
from model_core.native_backend import (
    AGGRESSIVE_DISABLED_OPS,
    NUMERICALLY_STABLE_NATIVE_OPS,
    STRICT_DISABLED_OPS,
)
from model_core.vocab import FORMULA_VOCAB
from scripts.analyze_kernel_coverage import formulas_from_checkpoint


def _policy_supported_ops(policy: str) -> set[str]:
    policy = policy.strip().lower()
    if policy in {"strict", "verified"}:
        return set(NUMERICALLY_STABLE_NATIVE_OPS) - set(STRICT_DISABLED_OPS)
    if policy == "aggressive":
        return set(NUMERICALLY_STABLE_NATIVE_OPS) - set(AGGRESSIVE_DISABLED_OPS)
    raise ValueError(f"unknown policy: {policy}")


def _op_name(token: int) -> str | None:
    token = int(token)
    if token < FORMULA_VOCAB.operator_offset:
        return None
    if token >= len(FORMULA_VOCAB.token_names):
        return None
    return FORMULA_VOCAB.token_names[token]


def _operator_runs(formula: Iterable[int], supported_ops: set[str]) -> list[tuple[str, ...]]:
    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in formula:
        name = _op_name(int(token))
        if name is None or name not in supported_ops:
            if current:
                runs.append(tuple(current))
                current = []
            continue
        current.append(name)
    if current:
        runs.append(tuple(current))
    return runs


def _ngrams(run: tuple[str, ...], min_len: int, max_len: int) -> Iterable[tuple[str, ...]]:
    max_len = min(max_len, len(run))
    for size in range(min_len, max_len + 1):
        for start in range(0, len(run) - size + 1):
            yield run[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--policy", default="aggressive", choices=["aggressive", "strict", "verified"])
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--glob",
        action="append",
        default=["checkpoints/ckpt_*.pt", "checkpoints/ga/ckpt_*.pt"],
    )
    args = parser.parse_args()

    supported_ops = _policy_supported_ops(args.policy)
    paths: list[pathlib.Path] = []
    for pattern in args.glob:
        paths.extend(ROOT.glob(pattern))
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)[: args.limit]

    chain_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    run_len_counts: collections.Counter[int] = collections.Counter()
    op_counts: collections.Counter[str] = collections.Counter()
    formulas_seen = 0
    formulas_with_chain = 0
    total_ops = 0
    supported_ops_seen = 0

    for path in paths:
        formulas = formulas_from_checkpoint(path)
        file_chains = 0
        for formula in formulas:
            formulas_seen += 1
            formula_has_chain = False
            for token in formula:
                name = _op_name(int(token))
                if name is None:
                    continue
                total_ops += 1
                if name in supported_ops:
                    supported_ops_seen += 1
                    op_counts[name] += 1
            for run in _operator_runs(formula, supported_ops):
                run_len_counts[len(run)] += 1
                if len(run) >= args.min_len:
                    formula_has_chain = True
                    file_chains += 1
                for chain in _ngrams(run, args.min_len, args.max_len):
                    chain_counts[chain] += 1
            if formula_has_chain:
                formulas_with_chain += 1
        print(
            f"{path.relative_to(ROOT)} formulas={len(formulas)} "
            f"chain_runs>={args.min_len}:{file_chains}"
        )

    print("")
    print(f"policy={args.policy}")
    print(f"supported_ops={sorted(supported_ops)}")
    print(f"formulas={formulas_seen} formulas_with_chain={formulas_with_chain}")
    print(
        "op_coverage="
        f"{(supported_ops_seen / total_ops if total_ops else 0.0):.1%} "
        f"supported_ops_seen={supported_ops_seen} total_ops={total_ops}"
    )
    print(f"run_len_counts={sorted(run_len_counts.items())}")
    print("top_ops=" + repr(op_counts.most_common(args.top)))
    print("top_chains=")
    for chain, count in chain_counts.most_common(args.top):
        families = tuple(classify_operator(name).value for name in chain)
        print(f"  count={count:4d} chain={' -> '.join(chain)} families={families}")


if __name__ == "__main__":
    main()
