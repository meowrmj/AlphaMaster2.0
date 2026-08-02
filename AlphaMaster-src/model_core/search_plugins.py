"""Optional candidate-generation plugins for alpha formula search.

Replay policies decide which known formulas are replayed into the policy loss.
Search plugins propose extra fresh formulas. Plugin proposals are evaluated and
can update archives/best strategy, but they do not have policy log-probs and do
not directly pull the neural generator in V1.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import torch

from .config import ModelConfig
from .elite_genetic import EliteGeneticEmitter
from .formula_diversity import formula_core_signature
from .replay_policies import ReplayEntry, formula_bucket_key
from .vocab import FORMULA_VOCAB


@dataclass
class SearchBatch:
    formulas: list[list[int]]
    origins: list[str]
    info: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.formulas)


class SearchPluginManager:
    def __init__(self, sampler) -> None:
        self.sampler = sampler
        self.elite_genetic = EliteGeneticEmitter(sampler)
        self.modules = _load_search_modules()
        self.archive: list[ReplayEntry] = []
        self.counter = 0
        self.anneal_state: dict[str, Any] = {
            "current_formula": None,
            "current_score": None,
            "accepted": 0,
            "tested": 0,
        }
        self._last_genetic_info: dict[str, Any] = {
            "planned": 0,
            "produced": 0,
            "parents": 0,
            "parent_source": "none",
        }

    @property
    def active(self) -> bool:
        return any(self.modules.values())

    def plan(
        self,
        *,
        step: int,
        candidate_slots: int,
        best_formula: list[int] | None,
        elite_pool: list[ReplayEntry] | None = None,
    ) -> SearchBatch:
        if candidate_slots <= 0 or not self.active:
            return SearchBatch([], [], self.metrics())
        frac = max(0.0, min(0.6, float(getattr(ModelConfig, "SEARCH_PLUGIN_FRAC", 0.20))))
        total = min(candidate_slots, int(round(ModelConfig.BATCH_SIZE * frac)))
        if total <= 0:
            return SearchBatch([], [], self.metrics())

        active = [name for name, enabled in self.modules.items() if enabled]
        formulas: list[list[int]] = []
        origins: list[str] = []
        per_module = max(1, math.ceil(total / len(active)))
        for name in active:
            remaining = total - len(formulas)
            if remaining <= 0:
                break
            k = min(per_module, remaining)
            if name == "annealing":
                batch = self._propose_annealing(step, k, best_formula)
            elif name == "genetic":
                batch = self._propose_genetic(step, k, best_formula, elite_pool or [])
            else:
                batch = []
            formulas.extend(batch)
            origins.extend([name] * len(batch))
        return SearchBatch(formulas, origins, self.metrics() | {"planned": len(formulas)})

    def observe(self, *, step: int, results: list[dict], origins: list[str]) -> None:
        for r, origin in zip(results, origins):
            if not origin:
                continue
            status = r.get("status", "error")
            if status in {"none", "const", "error"}:
                score = float(r.get("val_score", -1e9))
                formula = r.get("fml") or []
            else:
                score = float(r.get("val_score", -1e9))
                formula = [int(t) for t in (r.get("fml") or [])]
            if not formula:
                continue
            self._add_archive(score, formula, step)
            if origin == "annealing":
                self._observe_annealing(score, formula)

    def metrics(self) -> dict[str, Any]:
        names = [name for name, enabled in self.modules.items() if enabled]
        return {
            "search_plugins": names,
            "search_archive_size": len(self.archive),
            "search_archive_cells": len({formula_bucket_key(toks) for _sc, _cnt, toks, _birth in self.archive}),
            "anneal_accept_rate": self._anneal_accept_rate(),
            "genetic_planned": int(self._last_genetic_info.get("planned") or 0),
            "genetic_produced": int(self._last_genetic_info.get("produced") or 0),
            "genetic_parent_count": int(self._last_genetic_info.get("parents") or 0),
            "genetic_parent_source": str(self._last_genetic_info.get("parent_source") or "none"),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "modules": self.modules,
            "archive": self.archive,
            "counter": self.counter,
            "anneal_state": self.anneal_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        mods = state.get("modules")
        if isinstance(mods, dict):
            self.modules = {key: bool(mods.get(key, False)) for key in _KNOWN_MODULES}
        self.archive = _rebalance_archive(state.get("archive") or [])
        self.counter = int(state.get("counter") or 0)
        anneal = state.get("anneal_state")
        if isinstance(anneal, dict):
            self.anneal_state.update(anneal)

    def _propose_annealing(self, step: int, k: int, best_formula: list[int] | None) -> list[list[int]]:
        seed = self.anneal_state.get("current_formula") or self._archive_seed(best_formula)
        if not seed:
            seed = self._random_formula()
        out: list[list[int]] = []
        for _ in range(k):
            out.append(self._mutate_formula(seed, step))
        return out

    def _observe_annealing(self, score: float, formula: list[int]) -> None:
        self.anneal_state["tested"] = int(self.anneal_state.get("tested") or 0) + 1
        current_score = self.anneal_state.get("current_score")
        accept = current_score is None or score >= float(current_score)
        if not accept:
            temp = max(
                float(getattr(ModelConfig, "ANNEAL_TEMP_MIN", 0.03)),
                float(getattr(ModelConfig, "ANNEAL_TEMP", 0.35))
                * float(getattr(ModelConfig, "ANNEAL_DECAY", 0.997)) ** int(self.anneal_state["tested"]),
            )
            accept = random.random() < math.exp((score - float(current_score)) / max(temp, 1e-6))
        if accept:
            self.anneal_state["current_formula"] = [int(t) for t in formula]
            self.anneal_state["current_score"] = float(score)
            self.anneal_state["accepted"] = int(self.anneal_state.get("accepted") or 0) + 1

    def _propose_genetic(
        self,
        step: int,
        k: int,
        best_formula: list[int] | None,
        elite_pool: list[ReplayEntry],
    ) -> list[list[int]]:
        parent_pool = elite_pool if len(elite_pool) >= 2 else self.archive
        formulas, info = self.elite_genetic.propose(step=step, k=k, elite_pool=parent_pool)
        if formulas:
            self._last_genetic_info = info | {
                "parent_source": "elite_pool" if len(elite_pool) >= 2 else "search_archive",
            }
            return formulas
        self._last_genetic_info = info | {"parent_source": "none"}
        return []

    def _add_archive(self, score: float, formula: list[int], step: int) -> None:
        entry = (float(score), self.counter, [int(t) for t in formula], int(step))
        self.counter += 1
        self.archive = _rebalance_archive(self.archive + [entry])

    def _archive_seed(self, best_formula: list[int] | None) -> list[int] | None:
        if self.archive and random.random() < 0.85:
            return list(random.choice(self.archive[: min(12, len(self.archive))])[2])
        return list(best_formula) if best_formula else None

    def _select_parent(self, best_formula: list[int] | None) -> list[int] | None:
        if not self.archive:
            return list(best_formula) if best_formula else None
        k = min(max(2, int(getattr(ModelConfig, "GA_TOURNAMENT_K", 4))), len(self.archive))
        return list(max(random.sample(self.archive, k), key=lambda e: e[0])[2])

    def _random_formula(self) -> list[int]:
        return self._repair_with_prefix([], 0)

    def _mutate_formula(self, formula: list[int], step: int) -> list[int]:
        if not formula:
            return self._random_formula()
        cut = random.randrange(0, min(len(formula), ModelConfig.MAX_FORMULA_LEN))
        prefix = [int(t) for t in formula[:cut]]
        return self._repair_with_prefix(prefix, step)

    def _crossover(self, p1: list[int], p2: list[int], step: int) -> list[int]:
        if not p1 or not p2:
            return self._random_formula()
        cut1 = random.randrange(0, min(len(p1), ModelConfig.MAX_FORMULA_LEN))
        cut2 = random.randrange(0, min(len(p2), ModelConfig.MAX_FORMULA_LEN))
        prefix = [int(t) for t in p1[:cut1] + p2[cut2:]][: ModelConfig.MAX_FORMULA_LEN]
        return self._repair_with_prefix(prefix, step)

    def _repair_with_prefix(self, desired: list[int], step_seed: int) -> list[int]:
        depth = 0
        prev: int | None = None
        infected = 0
        out: list[int] = []
        device = torch.device("cpu")
        rng = random.Random(random.randrange(1 << 30) + int(step_seed))
        for si in range(ModelConfig.MAX_FORMULA_LEN):
            mask = self.sampler.valid_mask(
                depth, si, ModelConfig.MAX_FORMULA_LEN, device,
                prev_token=prev, infected_chain_len=infected,
            ).cpu().tolist()
            preferred = desired[si] if si < len(desired) else None
            if preferred is not None and 0 <= preferred < len(mask) and mask[preferred]:
                token = int(preferred)
            else:
                choices = [i for i, ok in enumerate(mask) if ok]
                token = int(rng.choice(choices))
            out.append(token)
            depth += self.sampler.delta[token]
            prev = token
            infected = self.sampler.update_infection(token, infected)
        return out

    def _anneal_accept_rate(self) -> float:
        tested = int(self.anneal_state.get("tested") or 0)
        if tested <= 0:
            return 0.0
        return int(self.anneal_state.get("accepted") or 0) / tested


_KNOWN_MODULES = ("annealing", "genetic")


def _load_search_modules() -> dict[str, bool]:
    raw = os.getenv("ALPHAMASTER_SEARCH_CONFIG") or os.getenv("ALPHAMASTER_SEARCH_PLUGINS") or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                modules = parsed.get("modules", parsed)
                if isinstance(modules, dict):
                    return {key: bool(modules.get(key, False)) for key in _KNOWN_MODULES}
        except json.JSONDecodeError:
            parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
            return {key: key in parts for key in _KNOWN_MODULES}
    legacy = os.getenv("ALPHAMASTER_SEARCH_PLUGINS", "")
    parts = {p.strip().lower() for p in legacy.split(",") if p.strip()}
    return {key: key in parts for key in _KNOWN_MODULES}


def _rebalance_archive(pool: list[ReplayEntry]) -> list[ReplayEntry]:
    cap = max(1, int(getattr(ModelConfig, "SEARCH_ARCHIVE_SIZE", 96)))
    bucket_cap = max(1, int(getattr(ModelConfig, "SEARCH_BUCKET_CAP", 4)))
    core_cap = max(1, int(getattr(ModelConfig, "SEARCH_CORE_CAP", 12)))
    by_formula: dict[tuple[int, ...], ReplayEntry] = {}
    for sc, cnt, toks, birth in pool:
        key = tuple(int(t) for t in toks)
        entry = (float(sc), int(cnt), [int(t) for t in toks], int(birth))
        if key not in by_formula or entry[0] > by_formula[key][0]:
            by_formula[key] = entry
    buckets: dict[tuple, list[ReplayEntry]] = {}
    for entry in by_formula.values():
        buckets.setdefault(formula_bucket_key(entry[2]), []).append(entry)
    kept: list[ReplayEntry] = []
    for entries in buckets.values():
        kept.extend(sorted(entries, key=lambda x: (x[0], x[1]), reverse=True)[:bucket_cap])
    by_core: dict[tuple[int, ...], list[ReplayEntry]] = {}
    for entry in sorted(kept, key=lambda x: (x[0], x[1]), reverse=True):
        by_core.setdefault(formula_core_signature(entry[2]), []).append(entry)
    diverse: list[ReplayEntry] = []
    for entries in by_core.values():
        diverse.extend(entries[:core_cap])
    return sorted(diverse, key=lambda x: (x[0], x[1]), reverse=True)[:cap]
