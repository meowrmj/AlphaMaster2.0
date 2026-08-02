"""Tree genetic emitter for elite replay formulas."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from .config import ModelConfig
from .vocab import FORMULA_VOCAB


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
    """Produce tree-level genetic offspring from replay elite entries."""

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
            return [], {"planned": 0, "produced": 0, "parents": len(elite_pool)}
        ranked = sorted(elite_pool, key=lambda item: (float(item[0]), int(item[1])), reverse=True)
        produced: list[list[int]] = []
        attempts = 0
        max_attempts = max(k * 8, 16)
        while len(produced) < k and attempts < max_attempts:
            attempts += 1
            p1 = self._select_parent(ranked)
            p2 = self._select_parent(ranked)
            child = self._crossover_or_mutate(p1, p2, step + attempts)
            if child and child not in produced:
                produced.append(child)
        return produced, {
            "planned": int(k),
            "produced": len(produced),
            "parents": len(elite_pool),
            "attempts": attempts,
        }

    def _select_parent(self, ranked: list[tuple[float, int, list[int], int]]) -> list[int]:
        k = min(max(2, int(getattr(ModelConfig, "ELITE_GENETIC_TOURNAMENT_K", 4))), len(ranked))
        return list(max(random.sample(ranked, k), key=lambda item: float(item[0]))[2])

    def _crossover_or_mutate(self, p1: list[int], p2: list[int], seed: int) -> list[int]:
        root1 = _parse_postfix(p1, self.sampler)
        root2 = _parse_postfix(p2, self.sampler)
        if root1 is None or root2 is None:
            return self._repair_with_prefix(p1[:], seed)
        if random.random() < float(getattr(ModelConfig, "ELITE_GENETIC_CROSSOVER_RATE", 0.70)):
            child_root = self._subtree_crossover(root1, root2)
        else:
            child_root = _clone_node(root1)
        if random.random() < float(getattr(ModelConfig, "ELITE_GENETIC_MUTATION_RATE", 0.35)):
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
                infected_chain_len=infected,
            ).cpu().tolist()
            preferred = int(desired[si]) if si < len(desired) else None
            if preferred is not None and 0 <= preferred < len(mask) and mask[preferred]:
                token = preferred
            else:
                token = int(rng.choice([idx for idx, ok in enumerate(mask) if ok]))
            out.append(token)
            depth += self.sampler.delta[token]
            infected = self.sampler.update_infection(token, infected)
        return out
