"""Kernel planning helpers for formula VM optimization.

The planner is intentionally execution-free. It converts a batch of formula
tokens into cacheable kernel buckets so a future CUDA/Triton/C++ backend can
reduce launch overhead without changing scoring semantics.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .formula_ir import FormulaCompiler, FormulaIR


class KernelFamily(str, Enum):
    ELEMENTWISE = "elementwise"
    SHIFT = "shift"
    ROLLING = "rolling"
    BRANCH = "branch"
    CROSS_SECTIONAL = "cross_sectional"
    REDUCTION = "reduction"
    UNSUPPORTED = "unsupported"


ELEMENTWISE_OPS = {
    "ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "SIGN", "MIN", "MAX",
    "MAX3", "POWER", "SIGNED_POWER_2", "SIGNED_LOG", "SQRT", "CLIP",
    "SIGMOID", "TANH_SQUASH", "WINSORIZE",
}
SHIFT_OPS = {"DELAY", "DELTA", "DELTA_5", "MOMENTUM_5", "MOMENTUM_10"}
ROLLING_OPS = {
    "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
    "TS_STD_5", "TS_STD_10", "TS_STD_20",
    "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
    "TS_MIN_10", "TS_MIN_20",
    "TS_MAX_10", "TS_MAX_20",
    "TS_RANK_5", "TS_RANK_10", "TS_RANK_20",
    "TS_ZSCORE_10", "TS_ZSCORE_20",
    "TS_CORR_10", "TS_SKEW_10", "TS_QUANTILE_10",
    "COVARIANCE_10",
    "WMA", "EMA_5", "EMA_20", "DECAY", "DECAY_LINEAR_5",
    "TS_DECAY_EXP_5", "PRODUCT_5",
}
BRANCH_OPS = {"IF_GT", "GATE", "JUMP"}
CROSS_SECTIONAL_OPS = {"CS_RANK", "CS_SCALE", "CS_NEUTRALIZE"}
REDUCTION_OPS = {"CORR_GATE"}


@dataclass(frozen=True)
class KernelBucket:
    """One kernel-launch candidate inside a token execution stage."""

    stage: int
    family: KernelFamily
    op_name: str
    arity: int
    formula_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.formula_indices)


@dataclass(frozen=True)
class KernelExecutionPlan:
    """Launch-reduction plan for a batch of formulas."""

    batch_key: tuple[tuple[int, ...], ...]
    valid_count: int
    invalid_count: int
    token_steps: int
    naive_launches: int
    bucketed_launches: int
    buckets: tuple[KernelBucket, ...]

    @property
    def launch_reduction(self) -> float:
        if self.naive_launches <= 0:
            return 0.0
        return 1.0 - (self.bucketed_launches / self.naive_launches)

    @property
    def supported_bucketed_launches(self) -> int:
        return sum(1 for b in self.buckets if b.family != KernelFamily.UNSUPPORTED)

    @property
    def unsupported_bucketed_launches(self) -> int:
        return sum(1 for b in self.buckets if b.family == KernelFamily.UNSUPPORTED)

    def top_buckets(self, limit: int = 10) -> list[KernelBucket]:
        return sorted(self.buckets, key=lambda b: (b.size, b.op_name), reverse=True)[:limit]


class KernelPlanCache:
    """Small LRU cache for repeated formula batches."""

    def __init__(self, max_size: int = 256):
        self.max_size = max(1, int(max_size))
        self._items: OrderedDict[tuple[tuple[int, ...], ...], KernelExecutionPlan] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[tuple[int, ...], ...]) -> KernelExecutionPlan | None:
        plan = self._items.get(key)
        if plan is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return plan

    def put(self, plan: KernelExecutionPlan) -> None:
        self._items[plan.batch_key] = plan
        self._items.move_to_end(plan.batch_key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


class KernelPlanner:
    """Build grouped execution plans for stack-formula batches."""

    def __init__(self, cache_size: int = 256):
        self.compiler = FormulaCompiler()
        self.cache = KernelPlanCache(cache_size)

    def plan(self, formulas: Iterable[Iterable[int]]) -> KernelExecutionPlan:
        batch_key = tuple(tuple(int(t) for t in f) for f in formulas)
        cached = self.cache.get(batch_key)
        if cached is not None:
            return cached

        irs = tuple(self.compiler.parse(f) for f in batch_key)
        token_steps = max((len(f) for f in batch_key), default=0)
        buckets: list[KernelBucket] = []
        naive_launches = 0

        for stage in range(token_steps):
            groups: dict[tuple[KernelFamily, str, int], list[int]] = {}
            for idx, ir in enumerate(irs):
                if not ir.valid or stage >= len(ir.tokens):
                    continue
                token = ir.tokens[stage]
                if token < self.compiler.feat_offset:
                    continue
                op_node = _operator_node_at_stage(ir, stage)
                if op_node is None:
                    continue
                naive_launches += 1
                family = classify_operator(op_node.name)
                groups.setdefault((family, op_node.name, len(op_node.inputs)), []).append(idx)

            for (family, op_name, arity), indices in groups.items():
                buckets.append(
                    KernelBucket(
                        stage=stage,
                        family=family,
                        op_name=op_name,
                        arity=arity,
                        formula_indices=tuple(indices),
                    )
                )

        plan = KernelExecutionPlan(
            batch_key=batch_key,
            valid_count=sum(1 for ir in irs if ir.valid),
            invalid_count=sum(1 for ir in irs if not ir.valid),
            token_steps=token_steps,
            naive_launches=naive_launches,
            bucketed_launches=len(buckets),
            buckets=tuple(buckets),
        )
        self.cache.put(plan)
        return plan


def classify_operator(op_name: str) -> KernelFamily:
    if op_name in ELEMENTWISE_OPS:
        return KernelFamily.ELEMENTWISE
    if op_name in SHIFT_OPS:
        return KernelFamily.SHIFT
    if op_name in ROLLING_OPS:
        return KernelFamily.ROLLING
    if op_name in BRANCH_OPS:
        return KernelFamily.BRANCH
    if op_name in CROSS_SECTIONAL_OPS:
        return KernelFamily.CROSS_SECTIONAL
    if op_name in REDUCTION_OPS:
        return KernelFamily.REDUCTION
    return KernelFamily.UNSUPPORTED


def _operator_node_at_stage(ir: FormulaIR, stage: int):
    for node in ir.nodes:
        if node.kind == "operator" and node.position == stage:
            return node
    return None
