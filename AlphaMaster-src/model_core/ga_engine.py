"""Independent genetic algorithm engine for formula mining."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import ModelConfig
from .engine import AlphaEngine, _build_walk_forward_folds, _artifact_suffix
from .replay_policies import formula_bucket_key
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError

_GA_CHECKPOINT_DIR = Path("checkpoints") / "ga"


class GeneticAlphaEngine(AlphaEngine):
    """A standalone GA search engine.

    It shares data loading, VM execution, walk-forward scoring, and strategy
    persistence with AlphaEngine, but it owns a separate population/checkpoint
    state and does not run neural policy gradients.
    """

    algorithm_mode = "ga"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, use_lord_regularization=False, **kwargs)
        self.population: list[list[int]] = []
        self.fitness: list[float] = []
        self.archive: list[tuple[float, int, list[int], int]] = []
        self.archive_counter = 0
        self.generation = 0
        self._feature_dim = None
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "val_score": [],
            "batch_best_val_score": [],
            "new_candidate_best_val_score": [],
            "ga_population_size": [],
            "ga_archive_size": [],
            "ga_archive_cells": [],
            "ga_elite_kept": [],
            "ga_random_injected": [],
            "ga_mutation_rate": [],
            "ga_crossover_rate": [],
            "timing_total_ms": [],
            "timing_eval_ms": [],
        }

    def train(self, start_step: int = 0, end_step: int | None = None, migration_hook=None, verbose_header: bool = True):
        if self.data_manager is None:
            raise RuntimeError("GeneticAlphaEngine requires a data_manager.")
        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS

        raw = self.data_manager.raw_dict
        feat = torch.stack([
            raw[k].to(ModelConfig.DEVICE)
            for k in ("open", "high", "low", "close", "volume")
        ], dim=1)
        self._feature_dim = int(feat.shape[1])
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)
        T = feat.shape[-1]
        folds = _build_walk_forward_folds(T, self.n_folds, ModelConfig.WF_GAP)
        use_wf = len(folds) > 0
        if not use_wf:
            raise RuntimeError("GA engine requires walk-forward folds for strict validation.")

        pop_size = max(8, int(ModelConfig.GA_POPULATION_SIZE))
        if not self.population:
            self.population = [self._random_formula() for _ in range(pop_size)]
        else:
            self.population = self.population[:pop_size]
            while len(self.population) < pop_size:
                self.population.append(self._random_formula())

        if verbose_header:
            print("GA engine: standalone population search")
            print(f"   population={pop_size} elite_frac={ModelConfig.GA_ELITE_FRAC:.3f} "
                  f"mutation={ModelConfig.GA_MUTATION_RATE:.3f} crossover={ModelConfig.GA_CROSSOVER_RATE:.3f}")
            print(f"   acceleration={'GPU batch' if str(ModelConfig.DEVICE).startswith('cuda') else 'CPU batch'} "
                  f"batch_eval={ModelConfig.GPU_BATCH_EVAL}")

        pbar = tqdm(
            range(start_step, end_step),
            total=end_step,
            initial=start_step,
            disable=not os.isatty(2),
            leave=False,
            mininterval=5.0,
        )

        for gen in pbar:
            import time
            t0 = time.perf_counter()
            eval0 = time.perf_counter()
            results = self._evaluate_population(self.population, feat, t_ret, folds, use_wf)
            eval_ms = (time.perf_counter() - eval0) * 1000.0

            scores = [float(r.get("val_score", -5.0)) for r in results]
            rewards = [float(r.get("reward", -5.0)) for r in results]
            self.fitness = scores
            valid_indices = [i for i, r in enumerate(results) if r.get("status") == "ok" and r.get("fml")]
            best_idx = max(valid_indices, key=lambda i: scores[i]) if valid_indices else None
            best_score = scores[best_idx] if best_idx is not None else max(scores, default=-5.0)
            best_formula = list(results[best_idx].get("fml")) if best_idx is not None else None

            if best_formula is not None and best_score > self.best_score:
                old = self.best_score
                self.best_score = best_score
                self.best_formula = best_formula
                self._best_update_step = gen
                self._save_strategy_live()
                tqdm.write(
                    f"[GA] 新冠军 @ 第{gen}代: 验证={best_score:.3f} "
                    f"(原 {old:.3f}, +{best_score-old:.3f}) | {self._decode_formula(best_formula)}"
                )

            for r in results:
                if r.get("status") == "ok":
                    self._archive_add(float(r.get("val_score", -5.0)), list(r.get("fml") or []), gen)

            avg_reward = sum(rewards) / max(1, len(rewards))
            avg_val = sum(scores) / max(1, len(scores))
            elite_count = max(1, int(pop_size * max(0.01, min(0.5, ModelConfig.GA_ELITE_FRAC))))
            random_count = max(1, int(pop_size * max(0.0, min(0.5, ModelConfig.GA_RANDOM_INJECT_FRAC))))
            self._append_history(gen, avg_reward, avg_val, best_score, elite_count, random_count, eval_ms, (time.perf_counter() - t0) * 1000.0)

            if gen % 5 == 0:
                tqdm.write(
                    f"[GA {gen+1}/{end_step}] 种群={pop_size} 平均验证={avg_val:.3f} "
                    f"本代最高={best_score:.3f} 历史冠军={self.best_score:.3f} 有效={len(valid_indices)} "
                    f"档案={len(self.archive)} 类型={len({formula_bucket_key(x[2]) for x in self.archive})} "
                    f"评估={eval_ms:.0f}ms"
                )

            self.population = self._next_generation(results, pop_size, elite_count, random_count)
            self.generation = gen + 1

            if gen % 5 == 0:
                self._save_training_history_live()
            if gen > 0 and gen % 20 == 0:
                self.save_checkpoint(gen)

        self._save_training_history_live()
        self.save_checkpoint(max(start_step, end_step - 1))

    def _evaluate_population(self, population, feat, t_ret, folds, use_wf):
        snapshot = list(self.factor_pool)
        if ModelConfig.GPU_BATCH_EVAL and use_wf:
            return self._eval_formula_batch_tasks(self.generation, population, feat, t_ret, folds, use_wf, snapshot)
        if self._eval_pool is not None and self._eval_workers > 1 and len(population) > 1:
            futures = [
                self._eval_pool.submit(self._eval_formula_task, i, fml, feat, t_ret, folds, use_wf, snapshot)
                for i, fml in enumerate(population)
            ]
            by_idx = {}
            for fut in futures:
                r = fut.result()
                by_idx[r["idx"]] = r
            return [by_idx[i] for i in range(len(population))]
        return [
            self._eval_formula_task(i, fml, feat, t_ret, folds, use_wf, snapshot)
            for i, fml in enumerate(population)
        ]

    def _next_generation(self, results, pop_size: int, elite_count: int, random_count: int) -> list[list[int]]:
        ranked = sorted(results, key=lambda r: float(r.get("val_score", -5.0)), reverse=True)
        next_pop = [list(r.get("fml") or []) for r in ranked[:elite_count] if r.get("fml")]
        target_children = max(0, pop_size - elite_count - random_count)
        while len(next_pop) < elite_count + target_children:
            p1 = self._select_parent(ranked)
            p2 = self._select_parent(ranked)
            if random.random() < ModelConfig.GA_CROSSOVER_RATE:
                child = self._crossover(p1, p2)
            else:
                child = list(p1)
            if random.random() < ModelConfig.GA_MUTATION_RATE:
                child = self._mutate(child)
            next_pop.append(child)
        while len(next_pop) < pop_size:
            next_pop.append(self._random_formula())
        random.shuffle(next_pop)
        return next_pop[:pop_size]

    def _select_parent(self, ranked_results) -> list[int]:
        valid = [r for r in ranked_results if r.get("fml")]
        if not valid:
            return self._random_formula()
        k = min(max(2, ModelConfig.GA_TOURNAMENT_K), len(valid))
        return list(max(random.sample(valid, k), key=lambda r: float(r.get("val_score", -5.0))).get("fml"))

    def _crossover(self, p1: list[int], p2: list[int]) -> list[int]:
        cut1 = random.randrange(0, len(p1)) if p1 else 0
        cut2 = random.randrange(0, len(p2)) if p2 else 0
        return self._repair_formula((p1[:cut1] + p2[cut2:])[: ModelConfig.MAX_FORMULA_LEN])

    def _mutate(self, formula: list[int]) -> list[int]:
        if not formula:
            return self._random_formula()
        cut = random.randrange(0, len(formula))
        return self._repair_formula(formula[:cut])

    def _random_formula(self) -> list[int]:
        return self._repair_formula([])

    def _repair_formula(self, desired: list[int]) -> list[int]:
        depth = 0
        prev = None
        infected = 0
        out = []
        device = torch.device("cpu")
        for si in range(ModelConfig.MAX_FORMULA_LEN):
            mask = self.sampler.valid_mask(
                depth, si, ModelConfig.MAX_FORMULA_LEN, device,
                prev_token=prev, infected_chain_len=infected,
            ).cpu().tolist()
            if self._feature_dim is not None:
                for tid in range(int(self._feature_dim), FORMULA_VOCAB.operator_offset):
                    mask[tid] = False
            preferred = desired[si] if si < len(desired) else None
            if preferred is not None and 0 <= int(preferred) < len(mask) and mask[int(preferred)]:
                token = int(preferred)
            else:
                token = random.choice([i for i, ok in enumerate(mask) if ok])
            out.append(token)
            depth += self.sampler.delta[token]
            prev = token
            infected = self.sampler.update_infection(token, infected)
        return out

    def _archive_add(self, score: float, formula: list[int], generation: int) -> None:
        if not formula:
            return
        entry = (float(score), self.archive_counter, [int(t) for t in formula], int(generation))
        self.archive_counter += 1
        self.archive = _rebalance_archive(self.archive + [entry])

    def _append_history(self, gen, avg_reward, avg_val, gen_best, elite_count, random_count, eval_ms, total_ms):
        cells = len({formula_bucket_key(toks) for _sc, _cnt, toks, _birth in self.archive})
        self.training_history.setdefault("step", []).append(gen)
        self.training_history.setdefault("avg_reward", []).append(avg_reward)
        self.training_history.setdefault("best_score", []).append(self.best_score)
        self.training_history.setdefault("val_score", []).append(avg_val)
        self.training_history.setdefault("batch_best_val_score", []).append(gen_best)
        self.training_history.setdefault("new_candidate_best_val_score", []).append(gen_best)
        self.training_history.setdefault("ga_population_size", []).append(len(self.population))
        self.training_history.setdefault("ga_archive_size", []).append(len(self.archive))
        self.training_history.setdefault("ga_archive_cells", []).append(cells)
        self.training_history.setdefault("ga_elite_kept", []).append(elite_count)
        self.training_history.setdefault("ga_random_injected", []).append(random_count)
        self.training_history.setdefault("ga_mutation_rate", []).append(ModelConfig.GA_MUTATION_RATE)
        self.training_history.setdefault("ga_crossover_rate", []).append(ModelConfig.GA_CROSSOVER_RATE)
        self.training_history.setdefault("timing_total_ms", []).append(total_ms)
        self.training_history.setdefault("timing_eval_ms", []).append(eval_ms)

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _GA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            suffix = _artifact_suffix(self.target_symbol, getattr(self, "timeframe", None))
            path = str(_GA_CHECKPOINT_DIR / f"ckpt_ga_{suffix}_gen_{step:04d}.pt")
        ckpt = {
            "algorithm_mode": "ga",
            "generation": step,
            "vocab_version": VOCAB_VERSION,
            "population": self.population,
            "fitness": self.fitness,
            "archive": self.archive,
            "archive_counter": self.archive_counter,
            "best_score": self.best_score,
            "best_formula": self.best_formula,
            "training_history": self.training_history,
        }
        tmp_path = f"{path}.{os.getpid()}.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)
        if ckpt.get("algorithm_mode") not in {None, "ga"}:
            raise RuntimeError(f"checkpoint is not a GA checkpoint: {path}")
        version = ckpt.get("vocab_version")
        if version is None:
            raise VocabVersionMismatchError(f"GA checkpoint missing vocab_version: {path}")
        FORMULA_VOCAB.verify(version)
        self.population = [[int(t) for t in f] for f in (ckpt.get("population") or [])]
        self.fitness = [float(x) for x in (ckpt.get("fitness") or [])]
        self.archive = _rebalance_archive(ckpt.get("archive") or [])
        self.archive_counter = int(ckpt.get("archive_counter") or 0)
        self.best_score = float(ckpt.get("best_score", -float("inf")))
        self.best_formula = ckpt.get("best_formula")
        for key, value in (ckpt.get("training_history") or {}).items():
            self.training_history[key] = value
        completed = int(ckpt.get("generation") or 0)
        tqdm.write(f"[GA检查点] 已恢复 {path} generation={completed} best={self.best_score:.4f}")
        return completed


def _rebalance_archive(pool: list[tuple[float, int, list[int], int]]) -> list[tuple[float, int, list[int], int]]:
    cap = max(1, int(getattr(ModelConfig, "SEARCH_ARCHIVE_SIZE", 96)))
    bucket_cap = max(1, int(getattr(ModelConfig, "SEARCH_BUCKET_CAP", 4)))
    by_formula: dict[tuple[int, ...], tuple[float, int, list[int], int]] = {}
    for sc, cnt, toks, birth in pool:
        key = tuple(int(t) for t in toks)
        entry = (float(sc), int(cnt), [int(t) for t in toks], int(birth))
        if key not in by_formula or entry[0] > by_formula[key][0]:
            by_formula[key] = entry
    buckets: dict[tuple, list[tuple[float, int, list[int], int]]] = {}
    for entry in by_formula.values():
        buckets.setdefault(formula_bucket_key(entry[2]), []).append(entry)
    kept = []
    for entries in buckets.values():
        kept.extend(sorted(entries, key=lambda e: (e[0], e[1]), reverse=True)[:bucket_cap])
    return sorted(kept, key=lambda e: (e[0], e[1]), reverse=True)[:cap]
