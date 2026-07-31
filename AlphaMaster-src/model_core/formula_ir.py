"""Formula IR and backend routing primitives.

This module is deliberately small and semantics-neutral: it describes formula
structure and backend selection, but it does not change how formulas are scored.
The reference VM remains the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .vocab import FORMULA_VOCAB


@dataclass(frozen=True)
class FormulaNode:
    """A node in the stack-formula intermediate representation."""

    id: int
    position: int
    kind: str
    token: int
    name: str
    inputs: tuple[int, ...] = ()


@dataclass(frozen=True)
class FormulaIR:
    """Parsed formula structure shared by all execution backends."""

    tokens: tuple[int, ...]
    nodes: tuple[FormulaNode, ...]
    root_id: int | None
    valid: bool
    reason: str = ""
    op_names: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        """Shape fingerprint used by future compiled-plan caches."""
        return tuple((n.kind, n.name, len(n.inputs)) for n in self.nodes)


@dataclass(frozen=True)
class FormulaPlan:
    """A backend-independent plan for a batch of formulas."""

    formulas: tuple[tuple[int, ...], ...]
    irs: tuple[FormulaIR, ...]
    support: dict[str, bool] = field(default_factory=dict)

    def coverage(self, backend_name: str) -> float:
        if not self.irs:
            return 0.0
        covered = sum(1 for ir in self.irs if self.support.get(_support_key(backend_name, ir), False))
        return covered / len(self.irs)

    @property
    def invalid_count(self) -> int:
        return sum(1 for ir in self.irs if not ir.valid)


class FormulaBackend(Protocol):
    """Common contract for standard, PyTorch batch, and future fused backends."""

    name: str

    def supports(self, ir: FormulaIR) -> bool:
        ...

    def evaluate(
        self,
        plan: FormulaPlan,
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict[str, Any]]:
        ...


class FormulaCompiler:
    """Parse token formulas into IR and build backend-neutral plans."""

    def __init__(self):
        self.vocab = FORMULA_VOCAB
        self.names = self.vocab.token_names
        self.feat_offset = self.vocab.operator_offset

    def parse(self, formula: list[int] | tuple[int, ...]) -> FormulaIR:
        stack: list[int] = []
        nodes: list[FormulaNode] = []
        op_names: list[str] = []
        feature_names: list[str] = []
        tokens = tuple(int(t) for t in formula)

        for position, token in enumerate(tokens):
            if token < 0 or token >= len(self.names):
                return FormulaIR(tokens, tuple(nodes), None, False, f"unknown token {token}")

            name = self.names[token]
            node_id = len(nodes)
            if token < self.feat_offset:
                nodes.append(FormulaNode(node_id, position, "feature", token, name))
                stack.append(node_id)
                feature_names.append(name)
                continue

            arity = _operator_arity(token)
            if arity is None:
                return FormulaIR(tokens, tuple(nodes), None, False, f"unknown operator {name}")
            if len(stack) < arity:
                return FormulaIR(tokens, tuple(nodes), None, False, f"stack underflow at {name}")

            inputs = tuple(stack[-arity:])
            del stack[-arity:]
            nodes.append(FormulaNode(node_id, position, "operator", token, name, inputs))
            stack.append(node_id)
            op_names.append(name)

        if len(stack) != 1:
            return FormulaIR(tokens, tuple(nodes), None, False, f"stack ended with {len(stack)} values")

        return FormulaIR(
            tokens=tokens,
            nodes=tuple(nodes),
            root_id=stack[0],
            valid=True,
            op_names=tuple(op_names),
            feature_names=tuple(feature_names),
        )

    def compile_batch(self, formulas: list[list[int]] | tuple[tuple[int, ...], ...]) -> FormulaPlan:
        irs = tuple(self.parse(f) for f in formulas)
        return FormulaPlan(formulas=tuple(tuple(int(t) for t in f) for f in formulas), irs=irs)


class BackendSelector:
    """Choose the first backend that supports a formula plan."""

    def __init__(self, backends: list[FormulaBackend]):
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends

    def select(self, plan: FormulaPlan, prefer_fast: bool = True) -> FormulaBackend:
        ordered = self.backends if prefer_fast else list(reversed(self.backends))
        for backend in ordered:
            if all(backend.supports(ir) for ir in plan.irs):
                return backend
        return self.backends[-1]


class BackendRegistry:
    """Ordered backend registry used as a responsibility chain.

    Backends are registered from fastest/most specific to slowest/most general.
    The final backend should normally be the standard interpreter.
    """

    def __init__(self):
        self._backends: list[FormulaBackend] = []

    def register(self, backend: FormulaBackend) -> None:
        if any(b.name == backend.name for b in self._backends):
            raise ValueError(f"backend already registered: {backend.name}")
        self._backends.append(backend)

    @property
    def backends(self) -> tuple[FormulaBackend, ...]:
        return tuple(self._backends)

    def annotate(self, plan: FormulaPlan) -> FormulaPlan:
        return annotate_support(plan, list(self._backends))

    def select(self, plan: FormulaPlan, prefer_fast: bool = True) -> FormulaBackend:
        return BackendSelector(list(self._backends)).select(plan, prefer_fast=prefer_fast)


def annotate_support(plan: FormulaPlan, backends: list[FormulaBackend]) -> FormulaPlan:
    support: dict[str, bool] = {}
    for backend in backends:
        for ir in plan.irs:
            support[_support_key(backend.name, ir)] = backend.supports(ir)
    return FormulaPlan(formulas=plan.formulas, irs=plan.irs, support=support)


def _support_key(backend_name: str, ir: FormulaIR) -> str:
    token_key = ",".join(str(t) for t in ir.tokens)
    return f"{backend_name}:{token_key}"


def _operator_arity(token: int) -> int | None:
    from .ops import OPS_CONFIG

    idx = token - FORMULA_VOCAB.operator_offset
    if idx < 0 or idx >= len(OPS_CONFIG):
        return None
    return int(OPS_CONFIG[idx][2])
