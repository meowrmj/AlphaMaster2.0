"""Behavior-correlation deduplication for formula archives."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .formula_diversity import formula_core_signature

FormulaKey = tuple[int, ...]
ReplayLikeEntry = tuple[float, int, list[int], int]


def formula_key(formula: Iterable[int]) -> FormulaKey:
    return tuple(int(t) for t in formula)


def behavior_vector_from_factor(
    factor: torch.Tensor,
    train_slice: tuple[int, int] | None,
    size: int,
) -> torch.Tensor:
    vecs = behavior_vectors_from_factors(factor.unsqueeze(0), train_slice, size)
    return vecs[0]


def behavior_vectors_from_factors(
    factors: torch.Tensor,
    train_slice: tuple[int, int] | None,
    size: int,
) -> torch.Tensor:
    if factors.ndim == 2:
        factors = factors.unsqueeze(0)
    if factors.ndim != 3:
        raise ValueError(f"expected factors [B,N,T], got shape={tuple(factors.shape)}")

    size = max(8, int(size))
    if train_slice is not None:
        start, end = int(train_slice[0]), int(train_slice[1])
        start = max(0, min(start, factors.shape[-1]))
        end = max(start, min(end, factors.shape[-1]))
        x = factors[:, :, start:end]
    else:
        x = factors

    flat = torch.nan_to_num(x.detach().float().reshape(x.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
    width = flat.shape[1]
    if width <= 0:
        return torch.zeros((factors.shape[0], size), dtype=torch.float32, device=factors.device)
    if width > size:
        idx = torch.linspace(0, width - 1, steps=size, device=flat.device).long()
        flat = flat.index_select(1, idx)
    elif width < size:
        flat = torch.nn.functional.pad(flat, (0, size - width))

    flat = flat - flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True, unbiased=False)
    stable = std >= 1e-8
    return torch.where(stable, flat / (std + 1e-8), torch.zeros_like(flat))


def normalize_behavior(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().flatten().cpu()
        if tensor.numel() == 0:
            return None
        return [float(x) for x in tensor.tolist()]
    if isinstance(value, (list, tuple)):
        try:
            out = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
        return out or None
    return None


def serialize_behavior_map(behavior_by_formula: dict[FormulaKey, list[float]]) -> list[dict[str, Any]]:
    return [
        {"formula": list(key), "behavior": list(vec)}
        for key, vec in behavior_by_formula.items()
        if key and vec
    ]


def load_behavior_map(raw: Any) -> dict[FormulaKey, list[float]]:
    out: dict[FormulaKey, list[float]] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            key = formula_key(item.get("formula") or [])
            vec = normalize_behavior(item.get("behavior"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            key = formula_key(item[0] or [])
            vec = normalize_behavior(item[1])
        else:
            continue
        if key and vec:
            out[key] = vec
    return out


def prune_behavior_map(
    behavior_by_formula: dict[FormulaKey, list[float]],
    pools: Iterable[Iterable[ReplayLikeEntry]],
) -> dict[FormulaKey, list[float]]:
    alive: set[FormulaKey] = set()
    for pool in pools:
        for _score, _counter, toks, _birth in pool:
            alive.add(formula_key(toks))
    return {key: vec for key, vec in behavior_by_formula.items() if key in alive}


def dedupe_entries_by_behavior(
    entries: list[ReplayLikeEntry],
    behavior_by_formula: dict[FormulaKey, list[float]],
    threshold: float,
    core_threshold: float | None = None,
) -> list[ReplayLikeEntry]:
    threshold = float(threshold)
    core_threshold = threshold if core_threshold is None else float(core_threshold)
    if not entries or threshold >= 1.0:
        return entries

    kept: list[ReplayLikeEntry] = []
    kept_items: list[tuple[torch.Tensor, tuple[int, ...]]] = []
    for entry in sorted(entries, key=lambda x: (float(x[0]), int(x[1])), reverse=True):
        key = formula_key(entry[2])
        vec = normalize_behavior(behavior_by_formula.get(key))
        if not vec:
            kept.append(entry)
            continue
        tensor = torch.tensor(vec, dtype=torch.float32)
        norm = torch.linalg.vector_norm(tensor)
        if not torch.isfinite(norm) or float(norm) < 1e-8:
            kept.append(entry)
            continue
        tensor = tensor / norm
        core = formula_core_signature(entry[2])
        if kept_items:
            mat = torch.stack([item[0] for item in kept_items])
            corr_values = torch.abs(mat @ tensor)
            reject = False
            for idx, corr in enumerate(corr_values.tolist()):
                limit = core_threshold if kept_items[idx][1] == core else threshold
                if float(corr) >= limit:
                    reject = True
                    break
            if reject:
                continue
        kept.append(entry)
        kept_items.append((tensor, core))
    return kept


def filter_new_entries_by_behavior(
    existing_entries: list[ReplayLikeEntry],
    new_entries: list[ReplayLikeEntry],
    behavior_by_formula: dict[FormulaKey, list[float]],
    threshold: float,
    core_threshold: float | None = None,
) -> list[ReplayLikeEntry]:
    threshold = float(threshold)
    core_threshold = threshold if core_threshold is None else float(core_threshold)
    if not new_entries or threshold >= 1.0:
        return new_entries

    kept_items: list[tuple[torch.Tensor, tuple[int, ...]]] = []
    for entry in existing_entries:
        key = formula_key(entry[2])
        vec = normalize_behavior(behavior_by_formula.get(key))
        if not vec:
            continue
        tensor = torch.tensor(vec, dtype=torch.float32)
        norm = torch.linalg.vector_norm(tensor)
        if torch.isfinite(norm) and float(norm) >= 1e-8:
            kept_items.append((tensor / norm, formula_core_signature(entry[2])))

    accepted: list[ReplayLikeEntry] = []
    for entry in sorted(new_entries, key=lambda x: (float(x[0]), int(x[1])), reverse=True):
        key = formula_key(entry[2])
        vec = normalize_behavior(behavior_by_formula.get(key))
        if not vec:
            accepted.append(entry)
            continue
        tensor = torch.tensor(vec, dtype=torch.float32)
        norm = torch.linalg.vector_norm(tensor)
        if not torch.isfinite(norm) or float(norm) < 1e-8:
            accepted.append(entry)
            continue
        tensor = tensor / norm
        core = formula_core_signature(entry[2])
        reject = False
        if kept_items:
            mat = torch.stack([item[0] for item in kept_items])
            corr_values = torch.abs(mat @ tensor)
            for idx, corr in enumerate(corr_values.tolist()):
                limit = core_threshold if kept_items[idx][1] == core else threshold
                if float(corr) >= limit:
                    reject = True
                    break
        if reject:
            continue
        accepted.append(entry)
        kept_items.append((tensor, core))
    return accepted
