"""Pluggable formula evaluation engines.

The training loop must keep its semantics: same formulas in, same scores out.
These classes only choose how the formulas are evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import torch

from .formula_ir import BackendRegistry, FormulaCompiler, FormulaIR, FormulaPlan


EvalFn = Callable[
    [int, list[int], torch.Tensor, torch.Tensor, list[dict], bool, list],
    dict[str, Any],
]
BatchEvalFn = Callable[
    [list[list[int]], torch.Tensor, torch.Tensor, list[dict], bool, list],
    list[dict[str, Any]],
]


@dataclass
class EvalGuardReport:
    checked: int = 0
    passed: bool = True
    max_reward_diff: float = 0.0
    max_val_diff: float = 0.0
    max_oos_sortino_diff: float = 0.0
    max_factor_diff: float = 0.0
    top_reward_stable: bool = True
    top_val_stable: bool = True
    reason: str = ""
    elapsed_ms: float = 0.0


class FormulaEvaluator:
    name = "base"

    def supports(self, ir: FormulaIR) -> bool:
        return ir.valid

    def evaluate(
        self,
        formulas: list[list[int]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def evaluate_plan(
        self,
        plan: FormulaPlan,
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict[str, Any]]:
        formulas = [list(f) for f in plan.formulas]
        return self.evaluate(formulas, feat, t_ret, folds, use_wf, factor_pool_snapshot)


class StandardFormulaEvaluator(FormulaEvaluator):
    """Reference path: evaluate formulas one by one with the scalar VM."""

    name = "standard"

    def __init__(self, scalar_eval: EvalFn):
        self.scalar_eval = scalar_eval

    def supports(self, ir: FormulaIR) -> bool:
        return True

    def evaluate(
        self,
        formulas: list[list[int]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict[str, Any]]:
        return [
            self.scalar_eval(i, fml, feat, t_ret, folds, use_wf, factor_pool_snapshot)
            for i, fml in enumerate(formulas)
        ]


class FastBatchFormulaEvaluator(FormulaEvaluator):
    """Current low-risk fast path.

    This wraps the already validated batch evaluator. Future CUDA graph / op
    bucketing work should replace this internals while keeping this contract.
    """

    name = "fast_batch"

    def __init__(self, batch_eval: BatchEvalFn):
        self.batch_eval = batch_eval
        self.last_plan: FormulaPlan | None = None
        self.last_invalid: list[FormulaIR] = []

    def supports(self, ir: FormulaIR) -> bool:
        return ir.valid

    def evaluate(
        self,
        formulas: list[list[int]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict[str, Any]]:
        return self.batch_eval(formulas, feat, t_ret, folds, use_wf, factor_pool_snapshot)

    def evaluate_plan(
        self,
        plan: FormulaPlan,
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict[str, Any]]:
        self.last_plan = plan
        self.last_invalid = [ir for ir in plan.irs if not ir.valid]
        return self.evaluate([list(f) for f in plan.formulas], feat, t_ret, folds, use_wf, factor_pool_snapshot)


class ScoreGuard:
    """Compare a fast evaluator against the scalar reference without changing training results."""

    def __init__(
        self,
        reference: StandardFormulaEvaluator,
        sample_size: int = 8,
        every_n_steps: int = 50,
        score_tol: float = 1e-4,
        factor_tol: float = 1e-4,
        top_k: int = 8,
    ):
        self.reference = reference
        self.sample_size = max(1, int(sample_size))
        self.every_n_steps = max(1, int(every_n_steps))
        self.score_tol = float(score_tol)
        self.factor_tol = float(factor_tol)
        self.top_k = max(1, int(top_k))
        self.last_report = EvalGuardReport()

    def should_check(self, step: int) -> bool:
        return step <= 2 or (step % self.every_n_steps == 0)

    def check(
        self,
        *,
        step: int,
        formulas: list[list[int]],
        fast_results: list[dict[str, Any]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> EvalGuardReport:
        if not formulas or not self.should_check(step):
            return self.last_report
        t0 = time.perf_counter()
        sample_idx = self._sample_indices(formulas, fast_results)
        count = len(sample_idx)

        ref_formulas = [formulas[i] for i in sample_idx]
        ref_results = self.reference.evaluate(
            ref_formulas, feat, t_ret, folds, use_wf, factor_pool_snapshot
        )

        max_reward = 0.0
        max_val = 0.0
        max_oos_sortino = 0.0
        max_factor = 0.0
        reason = ""
        ref_by_idx: dict[int, dict[str, Any]] = {}
        fast_by_idx: dict[int, dict[str, Any]] = {}
        for ref_pos, original_idx in enumerate(sample_idx):
            ref = ref_results[ref_pos]
            fast = fast_results[original_idx]
            ref_by_idx[original_idx] = ref
            fast_by_idx[original_idx] = fast
            if ref.get("status") != fast.get("status"):
                reason = f"status mismatch idx={original_idx}: {ref.get('status')} vs {fast.get('status')}"
                break
            max_reward = max(max_reward, _num_diff(ref.get("reward"), fast.get("reward")))
            max_val = max(max_val, _num_diff(ref.get("val_score"), fast.get("val_score")))
            if "oos_sortino" in ref or "oos_sortino" in fast:
                max_oos_sortino = max(max_oos_sortino, _num_diff(ref.get("oos_sortino"), fast.get("oos_sortino")))
            ref_res = ref.get("res")
            fast_res = fast.get("res")
            if torch.is_tensor(ref_res) and torch.is_tensor(fast_res):
                diff = (ref_res.detach() - fast_res.detach()).abs().max().item()
                max_factor = max(max_factor, float(diff))

        top_reward_stable = True
        top_val_stable = True
        if not reason:
            top_reward_stable = _top_choice_stable(ref_by_idx, fast_by_idx, "reward", self.score_tol)
            top_val_stable = _top_choice_stable(ref_by_idx, fast_by_idx, "val_score", self.score_tol)

        passed = (
            not reason
            and max_reward <= self.score_tol
            and max_val <= self.score_tol
            and max_oos_sortino <= self.score_tol
            and max_factor <= self.factor_tol
            and top_reward_stable
            and top_val_stable
        )
        if not passed and not reason:
            reason = (
                f"diff too high reward={max_reward:.6g} val={max_val:.6g} "
                f"oos_sortino={max_oos_sortino:.6g} factor={max_factor:.6g} "
                f"top_reward_stable={top_reward_stable} top_val_stable={top_val_stable}"
            )
        self.last_report = EvalGuardReport(
            checked=count,
            passed=passed,
            max_reward_diff=max_reward,
            max_val_diff=max_val,
            max_oos_sortino_diff=max_oos_sortino,
            max_factor_diff=max_factor,
            top_reward_stable=top_reward_stable,
            top_val_stable=top_val_stable,
            reason=reason,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return self.last_report

    def _sample_indices(self, formulas: list[list[int]], fast_results: list[dict[str, Any]]) -> list[int]:
        if not formulas:
            return []
        count = min(self.sample_size, len(formulas))
        if count == len(formulas):
            base = set(range(count))
        else:
            stride = max(1, len(formulas) // count)
            base = set(range(0, len(formulas), stride))
            base = set(sorted(base)[:count])
        base.update(_top_indices(fast_results, "reward", self.top_k))
        base.update(_top_indices(fast_results, "val_score", self.top_k))
        return sorted(i for i in base if 0 <= i < len(formulas))


class EvaluatorRouter:
    """Route formula evaluation while preserving the training contract."""

    def __init__(
        self,
        *,
        standard: StandardFormulaEvaluator,
        fast: FormulaEvaluator,
        guard: ScoreGuard | None = None,
        mode: str = "auto",
    ):
        self.standard = standard
        self.fast = fast
        self.guard = guard
        self.mode = (mode or "auto").strip().lower()
        self.last_engine = "standard"
        self.last_guard_report = EvalGuardReport()
        self.compiler = FormulaCompiler()
        self.registry = BackendRegistry()
        self.registry.register(self.fast)
        self.registry.register(self.standard)
        self.last_plan: FormulaPlan | None = None
        self.last_fast_coverage = 0.0
        self.last_invalid_count = 0

    def evaluate(
        self,
        *,
        step: int,
        formulas: list[list[int]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
        prefer_fast: bool,
    ) -> list[dict[str, Any]]:
        plan = self.compiler.compile_batch(formulas)
        plan = self.registry.annotate(plan)
        self.last_plan = plan
        self.last_fast_coverage = plan.coverage(self.fast.name)
        self.last_invalid_count = plan.invalid_count

        if self.mode == "standard" or not prefer_fast:
            self.last_engine = self.standard.name
            return self.standard.evaluate_plan(plan, feat, t_ret, folds, use_wf, factor_pool_snapshot)

        fast_supported = self.last_fast_coverage >= 1.0
        if not fast_supported:
            self.last_engine = f"{self.fast.name}->standard_unsupported_fallback"
            return self.standard.evaluate_plan(plan, feat, t_ret, folds, use_wf, factor_pool_snapshot)

        try:
            results = self.fast.evaluate_plan(plan, feat, t_ret, folds, use_wf, factor_pool_snapshot)
        except Exception as exc:
            raise RuntimeError(f"formula evaluation engine failed: {self.fast.name}") from exc

        self.last_engine = self.fast.name
        if self.guard is not None:
            report = self.guard.check(
                step=step,
                formulas=formulas,
                fast_results=results,
                feat=feat,
                t_ret=t_ret,
                folds=folds,
                use_wf=use_wf,
                factor_pool_snapshot=factor_pool_snapshot,
            )
            self.last_guard_report = report
            if not report.passed:
                raise RuntimeError(f"formula evaluation guard failed: {report.reason}")
        return results


def _num_diff(a: Any, b: Any) -> float:
    try:
        fa = float(a)
        fb = float(b)
    except Exception:
        return math.inf
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return 0.0 if fa == fb else math.inf
    return abs(fa - fb)


def _score_value(result: dict[str, Any], key: str) -> float:
    if result.get("status") != "ok":
        return -math.inf
    try:
        value = float(result.get(key, -math.inf))
    except Exception:
        return -math.inf
    return value if math.isfinite(value) else -math.inf


def _top_indices(results: list[dict[str, Any]], key: str, k: int) -> list[int]:
    ranked = sorted(
        ((i, _score_value(result, key)) for i, result in enumerate(results)),
        key=lambda item: item[1],
        reverse=True,
    )
    return [i for i, value in ranked[:k] if math.isfinite(value)]


def _top_choice_stable(
    ref_by_idx: dict[int, dict[str, Any]],
    fast_by_idx: dict[int, dict[str, Any]],
    key: str,
    tol: float,
) -> bool:
    if not ref_by_idx or not fast_by_idx:
        return True
    ref_top_idx, ref_top = max(
        ((idx, _score_value(result, key)) for idx, result in ref_by_idx.items()),
        key=lambda item: item[1],
    )
    fast_top_idx, _fast_top = max(
        ((idx, _score_value(result, key)) for idx, result in fast_by_idx.items()),
        key=lambda item: item[1],
    )
    if ref_top_idx == fast_top_idx:
        return True
    ref_score_for_fast_top = _score_value(ref_by_idx[fast_top_idx], key)
    return abs(ref_top - ref_score_for_fast_top) <= tol
