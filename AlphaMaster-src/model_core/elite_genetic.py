"""Tree-level genetic candidate generation for elite formula archives."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from .config import ModelConfig
from .formula_diversity import (
    formula_core_signature,
    formula_niche_key,
    formula_start_token,
    is_too_similar,
)


@dataclass
class FormulaNode:
    token: int
    children: list["FormulaNode"]


def _clone_node(node: FormulaNode) -> FormulaNode:
    return FormulaNode(int(node.token), [_clone_node(child) for child in node.children])


def _parse_postfix(tokens: list[int], sampler) -> FormulaNode | None:
    stack: list[FormulaNode] = []
    for raw in tokens:
        token = int(raw)
        if token < 0 or token >= sampler.vocab_size:
            return None
        arity = 0 if token < sampler.feat_offset else int(sampler.arity_map.get(token, 1))
        if arity <= 0:
            stack.append(FormulaNode(token, []))
            continue
        if len(stack) < arity:
            return None
        children = stack[-arity:]
        del stack[-arity:]
        stack.append(FormulaNode(token, children))
    return stack[0] if len(stack) == 1 else None


def _to_postfix(node: FormulaNode) -> list[int]:
    out: list[int] = []
    for child in node.children:
        out.extend(_to_postfix(child))
    out.append(int(node.token))
    return out


def _paths(node: FormulaNode, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    out = [prefix]
    for index, child in enumerate(node.children):
        out.extend(_paths(child, prefix + (index,)))
    return out


def _get_subtree(node: FormulaNode, path: tuple[int, ...]) -> FormulaNode:
    cur = node
    for index in path:
        cur = cur.children[index]
    return cur


def _replace_subtree(node: FormulaNode, path: tuple[int, ...], replacement: FormulaNode) -> FormulaNode:
    if not path:
        return _clone_node(replacement)
    root = _clone_node(node)
    cur = root
    for index in path[:-1]:
        cur = cur.children[index]
    cur.children[path[-1]] = _clone_node(replacement)
    return root


class EliteGeneticEmitter:
    """Produce candidate formulas from elite parents without touching policy loss."""

    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def propose(
        self,
        *,
        step: int,
        k: int,
        elite_pool: list[tuple[float, int, list[int], int]],
    ) -> tuple[list[list[int]], dict[str, Any]]:
        if k <= 0 or len(elite_pool) < 2:
            return [], {"planned": int(max(0, k)), "produced": 0, "parents": len(elite_pool), "attempts": 0}
        ranked = self._diverse_parent_pool(
            sorted(elite_pool, key=lambda item: (float(item[0]), int(item[1])), reverse=True)
        )
        niches = self._build_parent_niches(ranked)
        if len(ranked) < 2:
            return [], {"planned": int(k), "produced": 0, "parents": len(elite_pool), "attempts": 0}
        produced: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        max_attempts = max(k * 12, 24)
        random_immigrants = min(
            k,
            int(round(k * float(getattr(ModelConfig, "GA_RANDOM_IMMIGRANT_FRAC", 0.12)))),
        )
        while len(produced) < k and attempts < max_attempts:
            attempts += 1
            if len(produced) < random_immigrants:
                child = self._random_formula(step + attempts)
            else:
                p1, p2, _cross_niche = self._select_parent_pair(ranked, niches)
                child = self._crossover_or_mutate(p1, p2, step + attempts)
            key = tuple(child)
            threshold = float(getattr(ModelConfig, "GA_CHILD_SIMILARITY_MAX", 0.82))
            if child and key not in seen and not is_too_similar(child, produced, threshold):
                seen.add(key)
                produced.append(child)
        return produced, {
            "planned": int(k),
            "produced": len(produced),
            "parents": len(ranked),
            "parent_niches": len(niches),
            "attempts": attempts,
        }

    def _diverse_parent_pool(
        self,
        ranked: list[tuple[float, int, list[int], int]],
    ) -> list[tuple[float, int, list[int], int]]:
        core_cap = max(1, int(getattr(ModelConfig, "GA_PARENT_CORE_CAP", 4)))
        start_cap = max(1, int(getattr(ModelConfig, "GA_PARENT_START_TOKEN_CAP", 8)))
        by_core: dict[tuple[int, ...], list[tuple[float, int, list[int], int]]] = {}
        for entry in ranked:
            by_core.setdefault(formula_core_signature(entry[2]), []).append(entry)
        core_limited: list[tuple[float, int, list[int], int]] = []
        for entries in by_core.values():
            core_limited.extend(entries[:core_cap])
        by_start: dict[int, list[tuple[float, int, list[int], int]]] = {}
        for entry in sorted(core_limited, key=lambda item: (float(item[0]), int(item[1])), reverse=True):
            by_start.setdefault(formula_start_token(entry[2]), []).append(entry)
        out: list[tuple[float, int, list[int], int]] = []
        for entries in by_start.values():
            out.extend(entries[:start_cap])
        return sorted(out, key=lambda item: (float(item[0]), int(item[1])), reverse=True)

    def _select_parent(self, ranked: list[tuple[float, int, list[int], int]]) -> list[int]:
        k = min(max(2, int(getattr(ModelConfig, "GA_TOURNAMENT_K", 4))), len(ranked))
        return list(max(random.sample(ranked, k), key=lambda item: float(item[0]))[2])

    def _build_parent_niches(
        self,
        ranked: list[tuple[float, int, list[int], int]],
    ) -> dict[tuple, list[tuple[float, int, list[int], int]]]:
        niches: dict[tuple, list[tuple[float, int, list[int], int]]] = {}
        for entry in ranked:
            niches.setdefault(formula_niche_key(entry[2]), []).append(entry)
        return {key: entries for key, entries in niches.items() if entries}

    def _select_parent_pair(
        self,
        ranked: list[tuple[float, int, list[int], int]],
        niches: dict[tuple, list[tuple[float, int, list[int], int]]],
    ) -> tuple[list[int], list[int], bool]:
        if len(niches) >= 2 and random.random() < float(getattr(ModelConfig, "GA_CROSS_NICHE_RATE", 0.85)):
            keys = list(niches.keys())
            k1, k2 = random.sample(keys, 2)
            return (
                self._select_parent(niches[k1]),
                self._select_parent(niches[k2]),
                True,
            )
        return self._select_parent(ranked), self._select_parent(ranked), False

    def _crossover_or_mutate(self, p1: list[int], p2: list[int], seed: int) -> list[int]:
        root1 = _parse_postfix(p1, self.sampler)
        root2 = _parse_postfix(p2, self.sampler)
        if root1 is None:
            return self._repair_with_prefix([], seed)
        if root2 is None:
            root2 = root1
        if random.random() < float(getattr(ModelConfig, "GA_CROSSOVER_RATE", 0.70)):
            child_root = self._subtree_crossover(root1, root2)
        else:
            child_root = _clone_node(root1)
        if random.random() < float(getattr(ModelConfig, "GA_MUTATION_RATE", 0.45)):
            child_root = self._subtree_mutation(child_root, seed)
        return self._repair_with_prefix(_to_postfix(child_root), seed)

    def _subtree_crossover(self, root1: FormulaNode, root2: FormulaNode) -> FormulaNode:
        path1 = random.choice(_paths(root1))
        path2 = random.choice(_paths(root2))
        return _replace_subtree(root1, path1, _get_subtree(root2, path2))

    def _subtree_mutation(self, root: FormulaNode, seed: int) -> FormulaNode:
        donor = _parse_postfix(self._random_formula(seed), self.sampler)
        if donor is None:
            return root
        path = random.choice(_paths(root))
        donor_path = random.choice(_paths(donor))
        return _replace_subtree(root, path, _get_subtree(donor, donor_path))

    def _random_formula(self, seed: int) -> list[int]:
        return self._repair_with_prefix([], seed)

    def _repair_with_prefix(self, desired: list[int], seed: int) -> list[int]:
        depth = 0
        prev: int | None = None
        infected = 0
        out: list[int] = []
        device = torch.device("cpu")
        rng = random.Random(random.randrange(1 << 30) + int(seed))
        max_len = int(ModelConfig.MAX_FORMULA_LEN)
        for si in range(max_len):
            mask = self.sampler.valid_mask(
                depth,
                si,
                max_len,
                device,
                prev_token=prev,
                infected_chain_len=infected,
            ).cpu().tolist()
            preferred = int(desired[si]) if si < len(desired) else None
            if preferred is not None and 0 <= preferred < len(mask) and mask[preferred]:
                token = preferred
            else:
                token = int(rng.choice([idx for idx, ok in enumerate(mask) if ok]))
            out.append(token)
            depth += self.sampler.delta[token]
            prev = token
            infected = self.sampler.update_infection(token, infected)
        return out
