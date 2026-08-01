"""Replay policy strategies for Alpha formula training.

The engine owns the training loop; replay policies own memory, sampling and
update rules for historical formulas. This keeps elite/QD/incubation behavior
swappable without threading more conditionals through the trainer.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .vocab import FORMULA_VOCAB

ReplayEntry = tuple[float, int, list[int], int]


@dataclass
class ReplayBatch:
    formulas: list[list[int]]
    n_elite: int = 0
    n_incubation: int = 0
    elite_info: dict[str, Any] | None = None
    incubation_info: dict[str, Any] | None = None
    elite_frac_effective: float = 0.0

    @property
    def n_memory(self) -> int:
        return len(self.formulas)


def formula_bucket_key(formula: list[int]) -> tuple:
    op_offset = getattr(FORMULA_VOCAB, "operator_offset", 0)
    names = FORMULA_VOCAB.token_names
    first = int(formula[0]) if formula else -1
    feat_cnt = sum(1 for t in formula if t < op_offset)
    ts_cnt = arith_cnt = norm_cnt = nonlinear_cnt = 0
    for t in formula:
        name = names[t] if 0 <= t < len(names) else ""
        if name.startswith("TS_") or name in {"DELAY", "DELTA", "DECAY_LINEAR_5", "PRODUCT_5"}:
            ts_cnt += 1
        if name in {"ADD", "SUB", "MUL", "DIV", "NEG"}:
            arith_cnt += 1
        if "ZSCORE" in name or "RANK" in name or "SCALE" in name or "NORMALIZE" in name:
            norm_cnt += 1
        if name in {"SIGNED_LOG", "TANH_SQUASH", "SIGMOID", "ABS", "SQRT"}:
            nonlinear_cnt += 1
    return (
        first,
        min(feat_cnt, 3),
        min(ts_cnt, 3),
        min(arith_cnt, 2),
        min(norm_cnt, 2),
        min(nonlinear_cnt, 2),
    )


class ReplayPolicy:
    name = "none"

    def plan(self, *, step: int, batch_size: int, last_restart_step: int) -> tuple[int, ReplayBatch]:
        return batch_size, ReplayBatch(formulas=[])

    def observe(self, *, score: float, formula: list[int], step: int, is_new: bool, last_restart_step: int) -> None:
        return

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return

    def metrics(self) -> dict[str, Any]:
        return {
            "elite_pool_size": 0,
            "elite_archive_cells": 0,
            "incubation_pool_size": 0,
            "incubation_archive_cells": 0,
        }


class QDIncubationReplayPolicy(ReplayPolicy):
    name = "qd_incubation"

    def __init__(self, *, enable_qd: bool = True, enable_incubation: bool = True) -> None:
        self.enable_qd = bool(enable_qd)
        self.enable_incubation = bool(enable_incubation)
        if self.enable_qd and self.enable_incubation:
            self.name = "qd_incubation"
        elif self.enable_qd:
            self.name = "qd"
        elif self.enable_incubation:
            self.name = "incubation"
        else:
            self.name = "none"
        self.elite_pool: list[ReplayEntry] = []
        self.elite_counter = 0
        self.incubation_pool: list[ReplayEntry] = []
        self.incubation_counter = 0

    @staticmethod
    def _rebalance_elite_pool(pool: list[ReplayEntry]) -> list[ReplayEntry]:
        bucket_cap = max(1, int(getattr(ModelConfig, "ELITE_BUCKET_CAP", 3)))
        global_cap = max(1, int(ModelConfig.ELITE_POOL_SIZE))
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
        return sorted(kept, key=lambda x: (x[0], x[1]), reverse=True)[:global_cap]

    @staticmethod
    def _rebalance_incubation_pool(pool: list[ReplayEntry], step: int) -> list[ReplayEntry]:
        max_age = max(1, int(getattr(ModelConfig, "INCUBATION_CAPTURE_STEPS", 180)))
        bucket_cap = max(1, int(getattr(ModelConfig, "INCUBATION_BUCKET_CAP", 2)))
        global_cap = max(1, int(getattr(ModelConfig, "INCUBATION_POOL_SIZE", 36)))
        min_score = float(getattr(ModelConfig, "INCUBATION_MIN_SCORE", -0.5))
        fresh = [
            (float(sc), int(cnt), [int(t) for t in toks], int(birth))
            for sc, cnt, toks, birth in pool
            if step - int(birth) <= max_age and float(sc) >= min_score
        ]
        by_formula: dict[tuple[int, ...], ReplayEntry] = {}
        for entry in fresh:
            key = tuple(entry[2])
            if key not in by_formula or entry[0] > by_formula[key][0]:
                by_formula[key] = entry
        buckets: dict[tuple, list[ReplayEntry]] = {}
        for entry in by_formula.values():
            buckets.setdefault(formula_bucket_key(entry[2]), []).append(entry)
        kept: list[ReplayEntry] = []
        for entries in buckets.values():
            kept.extend(sorted(entries, key=lambda x: (x[0], -x[3], x[1]), reverse=True)[:bucket_cap])
        return sorted(kept, key=lambda x: (x[0], -x[3], x[1]), reverse=True)[:global_cap]

    def _sample_elite(self, step: int, k: int) -> tuple[list[list[int]], dict[str, Any]]:
        if not self.elite_pool or k <= 0:
            return [], {"avg_decay": 0.0, "max_age": 0, "age_list": [], "scores": [], "cells": 0}
        buckets: dict[tuple, list[tuple[float, int, list[int], int, float]]] = {}
        ages: list[int] = []
        decays: list[float] = []
        for sc, cnt, toks, birth in self.elite_pool:
            age = max(0, step - birth)
            decay = 1.0
            if ModelConfig.ELITE_DECAY:
                half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                decay = 0.5 ** (age / half)
            ages.append(age)
            decays.append(decay)
            buckets.setdefault(formula_bucket_key(toks), []).append((sc, cnt, toks, birth, decay))
        formulas: list[list[int]] = []
        scores: list[float] = []
        for key in random.choices(list(buckets.keys()), k=k):
            entries = buckets[key]
            ps = [e[0] for e in entries]
            ps_min = min(ps)
            ps_max = max(ps)
            if ps_max > ps_min:
                normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
            else:
                normalized = [1.0] * len(ps)
            temp = 0.7
            weights = [entries[i][4] * (2.0 ** (normalized[i] / temp)) for i in range(len(entries))]
            chosen = random.choices(entries, weights=weights, k=1)[0]
            formulas.append(list(chosen[2]))
            scores.append(float(chosen[0]))
        return formulas, {
            "avg_decay": sum(decays) / len(decays) if decays else 0.0,
            "max_age": max(ages) if ages else 0,
            "age_list": sorted(ages),
            "scores": scores,
            "cells": len(buckets),
        }

    def _sample_incubation(self, step: int, k: int) -> tuple[list[list[int]], dict[str, Any]]:
        self.incubation_pool = self._rebalance_incubation_pool(self.incubation_pool, step)
        if not self.incubation_pool or k <= 0:
            return [], {"cells": 0, "scores": [], "max_age": 0}
        buckets: dict[tuple, list[ReplayEntry]] = {}
        ages: list[int] = []
        for entry in self.incubation_pool:
            ages.append(max(0, step - entry[3]))
            buckets.setdefault(formula_bucket_key(entry[2]), []).append(entry)
        formulas: list[list[int]] = []
        scores: list[float] = []
        for key in random.choices(list(buckets.keys()), k=k):
            entries = buckets[key]
            floor = min(0.0, min(e[0] for e in entries))
            weights = [max(0.01, e[0] - floor + 0.01) for e in entries]
            chosen = random.choices(entries, weights=weights, k=1)[0]
            formulas.append(list(chosen[2]))
            scores.append(float(chosen[0]))
        return formulas, {"cells": len(buckets), "scores": scores, "max_age": max(ages) if ages else 0}

    def plan(self, *, step: int, batch_size: int, last_restart_step: int) -> tuple[int, ReplayBatch]:
        steps_since_restart = step - last_restart_step
        cooldown = max(0, int(getattr(ModelConfig, "ELITE_REPLAY_COOLDOWN_STEPS", 0)))
        recovery = max(1, int(getattr(ModelConfig, "ELITE_REPLAY_RECOVERY_STEPS", 1)))
        if steps_since_restart < cooldown:
            elite_frac_eff = 0.0
        elif steps_since_restart < cooldown + recovery:
            elite_frac_eff = ModelConfig.ELITE_REPLAY_FRAC * ((steps_since_restart - cooldown) / recovery)
        else:
            elite_frac_eff = ModelConfig.ELITE_REPLAY_FRAC

        incubation_active = (
            self.enable_incubation
            and steps_since_restart < max(0, int(getattr(ModelConfig, "INCUBATION_REPLAY_STEPS", 260)))
        )
        n_incubation = (
            int(batch_size * float(getattr(ModelConfig, "INCUBATION_REPLAY_FRAC", 0.08)))
            if incubation_active and self.incubation_pool
            else 0
        )
        n_elite = int(batch_size * elite_frac_eff) if self.enable_qd and self.elite_pool else 0
        n_elite = min(n_elite, max(0, batch_size - n_incubation))
        incubation_formulas, incubation_info = self._sample_incubation(step, n_incubation)
        n_incubation = len(incubation_formulas)
        elite_formulas, elite_info = self._sample_elite(step, n_elite)
        n_elite = len(elite_formulas)
        replay = ReplayBatch(
            formulas=incubation_formulas + elite_formulas,
            n_elite=n_elite,
            n_incubation=n_incubation,
            elite_info=elite_info,
            incubation_info=incubation_info,
            elite_frac_effective=elite_frac_eff,
        )
        return batch_size - replay.n_memory, replay

    def observe(self, *, score: float, formula: list[int], step: int, is_new: bool, last_restart_step: int) -> None:
        entry = (float(score), self.elite_counter, [int(t) for t in formula], int(step))
        self.elite_counter += 1
        if self.enable_qd:
            self.elite_pool = self._rebalance_elite_pool(self.elite_pool + [entry])
        capture_steps = max(0, int(getattr(ModelConfig, "INCUBATION_CAPTURE_STEPS", 180)))
        if self.enable_incubation and is_new and step - last_restart_step < capture_steps:
            inc = (float(score), self.incubation_counter, [int(t) for t in formula], int(step))
            self.incubation_counter += 1
            self.incubation_pool = self._rebalance_incubation_pool(self.incubation_pool + [inc], step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enable_qd": self.enable_qd,
            "enable_incubation": self.enable_incubation,
            "elite_pool": self.elite_pool,
            "elite_counter": self.elite_counter,
            "incubation_pool": self.incubation_pool,
            "incubation_counter": self.incubation_counter,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state_name = str(state.get("name") or self.name).strip().lower()
        if "enable_qd" in state:
            self.enable_qd = bool(state.get("enable_qd"))
        else:
            self.enable_qd = state_name in {"qd", "qd_incubation", "hybrid"}
        if "enable_incubation" in state:
            self.enable_incubation = bool(state.get("enable_incubation"))
        else:
            self.enable_incubation = state_name in {"incubation", "qd_incubation", "hybrid"}
        if self.enable_qd and self.enable_incubation:
            self.name = "qd_incubation"
        elif self.enable_qd:
            self.name = "qd"
        elif self.enable_incubation:
            self.name = "incubation"
        else:
            self.name = "none"
        self.elite_pool = self._rebalance_elite_pool(state.get("elite_pool") or []) if self.enable_qd else []
        self.elite_counter = int(state.get("elite_counter") or 0)
        incubation = state.get("incubation_pool") or []
        restore_step = max((int(entry[3]) for entry in incubation), default=0)
        self.incubation_pool = self._rebalance_incubation_pool(incubation, restore_step) if self.enable_incubation else []
        self.incubation_counter = int(state.get("incubation_counter") or 0)

    def metrics(self) -> dict[str, Any]:
        return {
            "elite_pool_size": len(self.elite_pool),
            "elite_archive_cells": len({formula_bucket_key(toks) for _sc, _cnt, toks, _birth in self.elite_pool}),
            "incubation_pool_size": len(self.incubation_pool),
            "incubation_archive_cells": len({formula_bucket_key(toks) for _sc, _cnt, toks, _birth in self.incubation_pool}),
        }


class NoReplayPolicy(ReplayPolicy):
    name = "none"


def _modules_from_config(config: dict[str, Any]) -> dict[str, bool]:
    modules = config.get("modules") if isinstance(config, dict) else None
    if isinstance(modules, dict):
        return {
            "qd": bool(modules.get("qd", True)),
            "incubation": bool(modules.get("incubation", True)),
        }
    return {"qd": True, "incubation": True}


def _modules_from_legacy_name(name: str) -> dict[str, bool]:
    mode = name.strip().lower()
    return {
        "qd": mode in {"qd", "qd_incubation", "hybrid"},
        "incubation": mode in {"incubation", "qd_incubation", "hybrid"},
    }


def _load_replay_config(value: str | dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raw = value if value is not None else os.getenv("ALPHAMASTER_REPLAY_CONFIG", "")
    if raw:
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    legacy = str(value or getattr(ModelConfig, "REPLAY_POLICY", "qd_incubation") or "qd_incubation")
    return {"modules": _modules_from_legacy_name(legacy)}


def build_replay_policy(name: str | dict[str, Any] | None = None) -> ReplayPolicy:
    config = _load_replay_config(name)
    modules = _modules_from_config(config)
    if not modules["qd"] and not modules["incubation"]:
        return NoReplayPolicy()
    if isinstance(name, str):
        mode = name.strip().lower()
    else:
        mode = str(getattr(ModelConfig, "REPLAY_POLICY", "qd_incubation")).strip().lower()
    if mode in {"none", "off", "disabled"}:
        return NoReplayPolicy()
    return QDIncubationReplayPolicy(enable_qd=modules["qd"], enable_incubation=modules["incubation"])
