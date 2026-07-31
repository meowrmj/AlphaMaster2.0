"""Analyze which saved formulas can be handled by a candidate fused VM kernel.

This is a planning/verification tool. It does not start training and does not
modify checkpoints. Use it before adding Triton/CUDA operators so we optimize
the operators that actually appear in recent runs.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys
from typing import Iterable

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.vocab import FORMULA_VOCAB
from model_core.kernel_planner import KernelPlanner


DEFAULT_KERNEL_OPS = {
    "ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "SIGN",
    "MIN", "MAX", "POWER", "SIGNED_POWER_2", "SIGNED_LOG",
    "SQRT", "CLIP", "SIGMOID", "TANH_SQUASH",
}


def _iter_formulas(obj) -> Iterable[list[int]]:
    if isinstance(obj, list):
        if obj and all(isinstance(x, int) for x in obj):
            yield [int(x) for x in obj]
            return
        for item in obj:
            yield from _iter_formulas(item)
    elif isinstance(obj, tuple):
        if len(obj) >= 3 and isinstance(obj[2], list):
            yield from _iter_formulas(obj[2])


def formulas_from_checkpoint(path: pathlib.Path) -> list[list[int]]:
    ckpt = torch.load(path, map_location="cpu")
    formulas: list[list[int]] = []
    for key in (
        "best_formula",
        "population",
        "archive",
        "elite_pool",
        "incubation_pool",
        "training_history",
    ):
        if key in ckpt:
            formulas.extend(_iter_formulas(ckpt[key]))
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for f in formulas:
        key = tuple(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--ops",
        default=",".join(sorted(DEFAULT_KERNEL_OPS)),
        help="Comma-separated operator names supported by the candidate kernel.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=["checkpoints/ckpt_*.pt", "checkpoints/ga/ckpt_*.pt"],
        help="Checkpoint glob to inspect. Can be passed multiple times.",
    )
    args = parser.parse_args()

    supported_ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    names = FORMULA_VOCAB.token_names
    op_offset = FORMULA_VOCAB.operator_offset

    paths: list[pathlib.Path] = []
    for pattern in args.glob:
        paths.extend(ROOT.glob(pattern))
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)[: args.limit]

    total = 0
    supported = 0
    naive_launches = 0
    bucketed_launches = 0
    unsupported_bucketed_launches = 0
    unsupported_ops = collections.Counter()
    all_ops = collections.Counter()
    planner = KernelPlanner()

    for path in paths:
        formulas = formulas_from_checkpoint(path)
        kernel_plan = planner.plan(formulas)
        naive_launches += kernel_plan.naive_launches
        bucketed_launches += kernel_plan.bucketed_launches
        unsupported_bucketed_launches += kernel_plan.unsupported_bucketed_launches
        file_total = len(formulas)
        file_supported = 0
        file_ops = collections.Counter()
        for formula in formulas:
            total += 1
            ok = True
            for token in formula:
                token = int(token)
                if token < op_offset:
                    continue
                name = names[token]
                all_ops[name] += 1
                file_ops[name] += 1
                if name not in supported_ops:
                    ok = False
                    unsupported_ops[name] += 1
            if ok:
                supported += 1
                file_supported += 1
        coverage = file_supported / file_total if file_total else 0.0
        print(f"{path.relative_to(ROOT)} formulas={file_total} kernel_cover={coverage:.1%}")
        print(
            "  launches="
            f"naive:{kernel_plan.naive_launches} "
            f"bucketed:{kernel_plan.bucketed_launches} "
            f"reduction:{kernel_plan.launch_reduction:.1%} "
            f"unsupported_buckets:{kernel_plan.unsupported_bucketed_launches}"
        )
        print(
            "  top_buckets="
            f"{[(b.stage, b.op_name, b.family.value, b.size) for b in kernel_plan.top_buckets(8)]}"
        )
        print(f"  top_ops={file_ops.most_common(10)}")

    print("")
    print(f"TOTAL formulas={total} kernel_cover={(supported / total if total else 0):.1%}")
    launch_reduction = 1.0 - (bucketed_launches / naive_launches) if naive_launches else 0.0
    print(
        f"TOTAL launches naive={naive_launches} bucketed={bucketed_launches} "
        f"reduction={launch_reduction:.1%} unsupported_buckets={unsupported_bucketed_launches}"
    )
    print(f"supported_ops={sorted(supported_ops)}")
    print(f"top_all_ops={all_ops.most_common(20)}")
    print(f"top_missing_ops={unsupported_ops.most_common(20)}")


if __name__ == "__main__":
    main()
