"""Benchmark kernel plan construction and cache behavior.

This benchmark does not execute formulas and does not start training. It only
measures the overhead of producing bucketed kernel plans from saved formulas.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_core.kernel_planner import KernelPlanner
from model_core.kernel_backends import DryRunKernelBackend
from scripts.analyze_kernel_coverage import formulas_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument(
        "--glob",
        action="append",
        default=["checkpoints/ckpt_*.pt", "checkpoints/ga/ckpt_*.pt"],
    )
    args = parser.parse_args()

    paths: list[pathlib.Path] = []
    for pattern in args.glob:
        paths.extend(ROOT.glob(pattern))
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)[: args.limit]

    formulas: list[list[int]] = []
    for path in paths:
        formulas.extend(formulas_from_checkpoint(path))
    formulas = formulas[: args.batch_size]
    if not formulas:
        raise SystemExit("no formulas found")

    cold_planner = KernelPlanner(cache_size=1)
    t0 = time.perf_counter()
    cold_plan = cold_planner.plan(formulas)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    backend_report = DryRunKernelBackend().analyze(cold_plan)

    hot_planner = KernelPlanner(cache_size=16)
    hot_planner.plan(formulas)
    hot_times: list[float] = []
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        hot_planner.plan(formulas)
        hot_times.append((time.perf_counter() - t0) * 1000.0)

    print(f"formulas={len(formulas)} token_steps={cold_plan.token_steps}")
    print(
        f"launches naive={cold_plan.naive_launches} "
        f"bucketed={cold_plan.bucketed_launches} "
        f"reduction={cold_plan.launch_reduction:.1%} "
        f"unsupported_buckets={cold_plan.unsupported_bucketed_launches}"
    )
    print(f"cold_plan_ms={cold_ms:.4f}")
    print(
        f"dry_backend executable={backend_report.executable} "
        f"planned={backend_report.planned_launches} "
        f"executable_launches={backend_report.executable_launches} "
        f"fallback_launches={backend_report.fallback_launches} "
        f"reason={backend_report.reason!r}"
    )
    print(
        f"hot_plan_ms avg={statistics.mean(hot_times):.6f} "
        f"p50={statistics.median(hot_times):.6f} "
        f"max={max(hot_times):.6f} "
        f"cache_hits={hot_planner.cache.hits} "
        f"cache_misses={hot_planner.cache.misses}"
    )
    print(
        "top_buckets="
        f"{[(b.stage, b.op_name, b.family.value, b.size) for b in cold_plan.top_buckets(12)]}"
    )


if __name__ == "__main__":
    main()
