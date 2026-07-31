"""Kernel backend contracts for formula VM optimization.

These classes define how fused execution engines plug into the planner. They do
not implement CUDA kernels yet and are not used by live training by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .kernel_planner import KernelExecutionPlan, KernelFamily


@dataclass(frozen=True)
class KernelBackendReport:
    backend: str
    executable: bool
    reason: str = ""
    planned_launches: int = 0
    executable_launches: int = 0
    fallback_launches: int = 0


class KernelBackend(Protocol):
    name: str

    def can_execute_family(self, family: KernelFamily) -> bool:
        ...

    def analyze(self, plan: KernelExecutionPlan) -> KernelBackendReport:
        ...

    def execute_factors(
        self,
        plan: KernelExecutionPlan,
        formulas: torch.Tensor,
        feat_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class DryRunKernelBackend:
    """A non-executing backend used to validate coverage before writing kernels."""

    name = "dry_run_kernel"

    def __init__(self, families: set[KernelFamily] | None = None):
        self.families = families or {
            KernelFamily.ELEMENTWISE,
            KernelFamily.SHIFT,
            KernelFamily.ROLLING,
            KernelFamily.BRANCH,
            KernelFamily.CROSS_SECTIONAL,
        }

    def can_execute_family(self, family: KernelFamily) -> bool:
        return family in self.families

    def analyze(self, plan: KernelExecutionPlan) -> KernelBackendReport:
        executable = sum(1 for b in plan.buckets if self.can_execute_family(b.family))
        fallback = plan.bucketed_launches - executable
        return KernelBackendReport(
            backend=self.name,
            executable=(fallback == 0 and plan.invalid_count == 0),
            reason="" if fallback == 0 else f"{fallback} buckets need fallback",
            planned_launches=plan.bucketed_launches,
            executable_launches=executable,
            fallback_launches=fallback,
        )

    def execute_factors(
        self,
        plan: KernelExecutionPlan,
        formulas: torch.Tensor,
        feat_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("DryRunKernelBackend only analyzes plans; it never executes formulas")
