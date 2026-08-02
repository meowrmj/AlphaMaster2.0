import copy
import heapq
import json
import math
import os
import pathlib
import random
import sys
import time

import torch

# ── 并行评估线程配置（必须在第一个张量操作前设置）──────────────────────────
# torch.set_num_interop_threads 一旦有任何张量 op 就锁定，所以紧跟 import torch。
# 策略：与 MT5_AlphaGPT 一致——保持 intra_threads=full cores，让 PyTorch 内部
# 多线程处理大张量；同时开 8 workers 并行评估不同公式。
# PyTorch 的 intra-op 线程在执行期间会释放 GIL，允许多个 worker 真正并行。
_PHYS_CORES = os.cpu_count() or 4
_EVAL_WORKERS = min(_PHYS_CORES, 8)
_INTRA = _PHYS_CORES  # 保持满线程，不做 phys // workers
try:
    torch.set_num_interop_threads(_EVAL_WORKERS)
except RuntimeError:
    pass  # 已被锁定
torch.set_num_threads(_INTRA)

import torch.nn.functional as F
from tqdm import tqdm

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MT5Backtest, estimate_periods_per_year
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2
from .replay_policies import build_replay_policy, formula_bucket_key
from .search_plugins import SearchPluginManager
from .behavior_dedup import behavior_vector_from_factor, behavior_vectors_from_factors
from .training_control import acknowledge_checkpoint_stop, read_checkpoint_stop_request

# P3：冠军在场时间稳健性校验所需
try:
    from strategy_manager.signal import compute_target_positions_stateless
except ImportError:
    # 兼容无 strategy_manager 的测试环境
    def compute_target_positions_stateless(factors):  # type: ignore
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE  = _RootConfig.STRATEGY_FILE
    _CHECKPOINT_DIR = pathlib.Path(getattr(_RootConfig, 'CHECKPOINT_DIR', 'checkpoints'))
except ImportError:
    _STRATEGY_FILE  = "best_mt5_strategy.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _categorical_stats_from_logits(
    logits: torch.Tensor,
    tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    if tokens is None:
        tokens = torch.multinomial(probs, 1).squeeze(1)
    log_prob = log_probs.gather(1, tokens[:, None]).squeeze(1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return tokens, log_prob, entropy


def _safe_artifact_tag(value: str | None) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "").strip())


def _artifact_suffix(symbol: str | None, timeframe: str | None = None) -> str:
    suffix = _safe_artifact_tag(symbol)
    tf = _safe_artifact_tag(timeframe)
    if suffix and tf:
        return f"{suffix}_{tf}"
    return suffix


def _strategy_file_for_symbol(
    symbol: str | None,
    timeframe: str | None = None,
    algorithm_mode: str | None = None,
) -> str:
    """返回该品种对应的策略文件路径。

    单品种训练时使用 strategies/best_{symbol}.json，
    多品种/未指定品种时回退到默认路径。
    """
    suffix = _artifact_suffix(symbol, timeframe)
    if suffix:
        mode = str(algorithm_mode or "rl").strip().lower()
        if mode not in {"rl", "ga", "hybrid"}:
            mode = "rl"
        return str(pathlib.Path("strategies") / "champions" / mode / f"best_{mode}_{suffix}.json")
    return _STRATEGY_FILE


def _fallback_data_file_for_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Read web_settings.json last_data_file when strategy JSON lacks data_file."""
    settings_path = pathlib.Path("web_settings.json")
    if not settings_path.exists():
        return None, None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        last = str(settings.get("last_data_file") or "").strip()
    except (json.JSONDecodeError, OSError):
        return None, None
    if not last:
        return None, None
    p = pathlib.Path(last)
    if not p.exists():
        return None, None
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(p.resolve()))
    except Exception:
        return str(p.resolve()), None
    if info.get("symbol") != symbol:
        return None, None
    return str(p.resolve()), info.get("timeframe")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 折叠构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_walk_forward_folds(T: int, n_folds: int = 5, gap: int = 20) -> list[dict]:
    """构建 Walk-Forward 折叠。

    改为 rolling window（train_start = (k-1)*fold_size）以避免 expanding window
    导致早期折的 val 数据被后续折的 train 切片包含，造成 val_score 不再严格 OOS。

    同时修正最后一折 val_end = min(val_start + fold_size, T)，避免最后一折 val
    大小不均导致均值被该折主导。
    """
    fold_size = T // n_folds
    if fold_size < 2:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    total_required = fold_size * n_folds + gap * (n_folds - 1)
    if total_required > T:
        gap = max(0, (T - fold_size * n_folds) // n_folds)
    folds = []
    for k in range(1, n_folds):
        # rolling window：每折 train 起点前移，避免包含早期折的 val 切片
        train_start = (k - 1) * fold_size
        train_end   = k * fold_size
        val_start   = train_end + gap
        # 最后一折 val_end 用 min 避免超出 T，且与其他折大小一致
        val_end     = min(val_start + fold_size, T)
        if val_start >= T or val_end <= val_start:
            break
        folds.append({"train_start": train_start, "train_end": train_end,
                      "val_start": val_start, "val_end": val_end, "gap": gap})
    if not folds:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    return folds


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保证 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedSampler:
    def __init__(self, vocab_size: int, feat_offset: int, arity_map: dict[int, int],
                 positive_only_ids: set[int] | None = None):
        self.vocab_size  = vocab_size
        self.feat_offset = feat_offset
        self.arity_map   = arity_map
        self.delta: dict[int, int] = {}
        for tid in range(vocab_size):
            if tid < feat_offset:
                self.delta[tid] = 1
            else:
                a = arity_map.get(tid, 1)
                self.delta[tid] = 1 - a
        # 恒正算子 token id 集合（用于算子链约束）
        self.positive_only_ids = positive_only_ids or set()
        # 构建感染传播/恢复算子 id 集合
        from .vm import INFECTED_PROPAGATING_OPS, SIGN_RESTORE_OPS
        from .ops import OPS_CONFIG as _ops
        self.infected_propagating_ids = set()
        self.sign_restore_ids = set()
        for i, cfg in enumerate(_ops):
            tid = i + feat_offset
            if cfg[0] in INFECTED_PROPAGATING_OPS:
                self.infected_propagating_ids.add(tid)
            if cfg[0] in SIGN_RESTORE_OPS:
                self.sign_restore_ids.add(tid)
        self._valid_mask_cache: dict[tuple[str, int, int, int, int | None, int], torch.Tensor] = {}
        self._constraint_tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _constraint_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._constraint_tensor_cache.get(key)
        if cached is not None:
            return cached
        delta = torch.tensor(
            [self.delta[tid] for tid in range(self.vocab_size)],
            dtype=torch.long,
            device=device,
        )
        infected = torch.tensor(
            [tid in self.infected_propagating_ids for tid in range(self.vocab_size)],
            dtype=torch.bool,
            device=device,
        )
        positive = torch.tensor(
            [tid in self.positive_only_ids for tid in range(self.vocab_size)],
            dtype=torch.bool,
            device=device,
        )
        restore = torch.tensor(
            [tid in self.sign_restore_ids for tid in range(self.vocab_size)],
            dtype=torch.bool,
            device=device,
        )
        cached = (delta, infected, positive, restore)
        self._constraint_tensor_cache[key] = cached
        return cached

    def valid_mask(self, stack_depth: int, step_idx: int,
                   total_steps: int, device: torch.device,
                   prev_token: int | None = None,
                   infected_chain_len: int = 0) -> torch.Tensor:
        remaining = total_steps - step_idx
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        for tid in range(self.vocab_size):
            d         = self.delta[tid]
            new_depth = stack_depth + d
            if new_depth < 1:
                mask[tid] = False;  continue
            min_future = new_depth + (remaining - 1) * (-2)
            max_future = new_depth + (remaining - 1) * 1
            if 1 < min_future or 1 > max_future:
                mask[tid] = False
            # ── 算子链约束（感染模型）──────────────────────────────
            # 如果已感染且感染链 >= 2，禁止再使用传播算子
            # （允许恢复算子和非传播算子如 ADD/SUB/MUL）
            if infected_chain_len >= 2 and tid in self.infected_propagating_ids:
                mask[tid] = False
            # 如果已感染且感染链 >= 3，禁止所有算子（强制恢复或结束）
            # 实际上不禁止恢复算子，只禁止传播和恒正算子
            if infected_chain_len >= 3:
                if tid in self.infected_propagating_ids or tid in self.positive_only_ids:
                    mask[tid] = False
        if not mask.any():
            for tid in range(self.vocab_size):
                if stack_depth + self.delta[tid] >= 1:
                    mask[tid] = True
        return mask

    def apply_mask_to_logits(self, logits: torch.Tensor, stack_depths: list[int],
                              step_idx: int, total_steps: int,
                              prev_tokens: list[int | None] | None = None,
        infected_chain_lens: list[int] | None = None) -> torch.Tensor:
        device = logits.device
        if logits.shape[0] == 0:
            return logits
        delta, infected, positive, _restore = self._constraint_tensors(device)
        if isinstance(stack_depths, torch.Tensor):
            depths = stack_depths.to(device=device, dtype=torch.long).view(-1, 1)
        else:
            depths = torch.tensor(stack_depths, dtype=torch.long, device=device).view(-1, 1)
        remaining = total_steps - step_idx
        new_depth = depths + delta.view(1, -1)
        mask = new_depth >= 1
        min_future = new_depth + (remaining - 1) * (-2)
        max_future = new_depth + (remaining - 1)
        mask = mask & (min_future <= 1) & (max_future >= 1)

        if infected_chain_lens is not None and len(infected_chain_lens) > 0:
            if isinstance(infected_chain_lens, torch.Tensor):
                infected_lens = infected_chain_lens.to(device=device, dtype=torch.long).view(-1, 1)
            else:
                infected_lens = torch.tensor(
                    infected_chain_lens,
                    dtype=torch.long,
                    device=device,
                ).view(-1, 1)
            infected_row = infected.view(1, -1)
            mask = mask & ~((infected_lens >= 2) & infected_row)
            mask = mask & ~((infected_lens >= 3) & (infected_row | positive.view(1, -1)))

        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any()):
            mask[empty_rows] = (depths[empty_rows] + delta.view(1, -1)) >= 1
        return logits.masked_fill(~mask, -1e9)

    def update_infection(self, token: int, infected_chain_len: int) -> int:
        """更新感染链长度，返回新的感染链长度。"""
        if token in self.positive_only_ids:
            return infected_chain_len + 1
        elif token in self.sign_restore_ids:
            return 0
        elif token in self.infected_propagating_ids:
            if infected_chain_len > 0:
                return infected_chain_len + 1
            return 0
        return infected_chain_len  # 非传播/非恢复算子，不改变状态


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEngine — __init__ 与静态辅助方法
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5, n_folds: int = 5,
                 target_symbol: str | None = None):
        self.data_manager  = data_manager
        self.n_folds       = n_folds
        self.target_symbol = target_symbol   # None = 多品种模式，str = 单品种模式
        self.model   = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["in_proj", "out_proj", "qk_norm"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MT5Backtest()
        self._batch_pipeline = None
        self._evaluator_router = None

        from .vocab import FORMULA_VOCAB as _v
        self.sampler = ConstrainedSampler(
            vocab_size=_v.size, feat_offset=_v.operator_offset,
            arity_map=self.vm.arity_map,
            positive_only_ids=self.vm.positive_only_ids
        )

        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [],
            'batch_best_val_score': [], 'new_candidate_best_val_score': [],
            'stable_rank': []
        }
        self._restart_count      = 0
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0
        self._incubation_pool: list[tuple[float, int, list[int], int]] = []
        self._incubation_counter = 0
        self.replay_policy = build_replay_policy()
        self.search_plugins = SearchPluginManager(self.sampler)
        self._last_restart_step = -10**9

        # 自适应噪声：记录 best 刷新步数
        self._best_update_step = 0
        self._stagnation_steps = 0

        # Fix 3: EMA reward baseline
        self._reward_ema: float | None = None
        self._reward_ema_step: int = 0

        # ── 并行评估线程池（CPU 利用率优化）──────────────────────────────
        self._eval_pool = None
        self._eval_workers = 1
        if ModelConfig.PARALLEL_EVAL:
            self._init_parallel_eval()

    # ── 并行评估初始化 ──────────────────────────────────────────────────────

    def _init_parallel_eval(self):
        """初始化 ThreadPoolExecutor 用于并行公式评估。

        PyTorch CPU 算子会释放 GIL，多个 worker 线程可以真正并行执行
        vm.execute / bt.evaluate_fold 等纯张量计算。

        线程配置（与 MT5_AlphaGPT 一致）：
        - intra_threads = physical_cores（保持满，让大张量操作快）
        - inter_threads = workers（允许多 worker 并行调度）
        PyTorch 的 intra-op 线程池会自适应负载，不会真的 8×8=64 全跑满。
        """
        from concurrent.futures import ThreadPoolExecutor

        phys = _PHYS_CORES
        workers = ModelConfig.EVAL_WORKERS
        if workers <= 0:
            workers = min(phys, 8)
        intra = ModelConfig.EVAL_INTRA_THREADS
        if intra <= 0:
            intra = phys  # 保持满线程
        # 线程已在模块 import 时设置，这里只确认
        try:
            torch.set_num_threads(intra)
        except RuntimeError:
            pass

        self._eval_workers = workers
        self._eval_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="alpha-eval",
        )
        if ModelConfig.GPU_BATCH_EVAL:
            print(
                f"[EvalMode] batch evaluator enabled device={ModelConfig.DEVICE} "
                f"intra_threads={intra}; legacy formula pool is bypassed during walk-forward eval",
                flush=True,
            )
        else:
            print(f"[ParallelEval] workers={workers} intra_threads={intra} "
                  f"(physical_cores={phys})  pool={'ON' if workers > 1 else 'OFF'}",
                  flush=True)

    # ── 单条公式评估任务（线程安全）───────────────────────────────────────

    def _eval_formula_task(
        self,
        idx: int,
        fml: list[int],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> dict:
        """评估单条公式（线程安全，可被多个 worker 并发调用）。

        所有共享对象（self.vm, self.bt）在评估路径上都是只读的；
        factor_pool 通过 snapshot 传入只读快照。
        """
        try:
            with torch.no_grad():
                res = self.vm.execute(fml, feat)

            if res is None:
                return {'idx': idx, 'status': 'none', 'reward': -5.0,
                        'val_score': -5.0, 'fml': fml}
            if res.std() < 1e-4:
                return {'idx': idx, 'status': 'const', 'reward': -2.0,
                        'val_score': -2.0, 'fml': fml}

            with torch.no_grad():
                if use_wf:
                    fold_tr, fold_vl, fold_ic = [], [], []
                    for fold in folds:
                        tr_sc, vl_sc = self.bt.evaluate_fold(
                            res, t_ret,
                            fold["train_start"], fold["train_end"],
                            fold["val_start"],   fold["val_end"],
                        )
                        ic_m, _ = AlphaEngine._compute_ic(
                            res[:, fold["train_start"]:fold["train_end"]],
                            t_ret[:, fold["train_start"]:fold["train_end"]],
                        )
                        tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                        fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                        ic_v, _ = AlphaEngine._compute_ic(
                            res[:, fold["val_start"]:fold["val_end"]],
                            t_ret[:, fold["val_start"]:fold["val_end"]],
                        )
                        vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
                        fold_vl.append(vl_adj)
                        fold_ic.append(ic_m.item())
                    train_score = torch.stack(fold_tr).mean()
                    val_score = torch.stack(fold_vl).mean()
                    ic_i = sum(fold_ic) / len(fold_ic)
                else:
                    T_total = res.shape[1]
                    split_pt = max(int(T_total * 0.8), T_total - 100)
                    train_score, _ = self.bt.evaluate(res, {}, t_ret)
                    ic_m0, _ = AlphaEngine._compute_ic(res, t_ret)
                    train_score = AlphaEngine._apply_ic_gate(
                        ModelConfig.REWARD_ALPHA * train_score, ic_m0
                    )
                    if split_pt < T_total - 1:
                        vl_sc, _ = self.bt.evaluate_fold(
                            res, t_ret, 0, split_pt, split_pt, T_total,
                        )
                        ic_v0, _ = AlphaEngine._compute_ic(
                            res[:, split_pt:], t_ret[:, split_pt:],
                        )
                        val_score = AlphaEngine._apply_ic_gate(vl_sc, ic_v0)
                    else:
                        val_score = train_score
                    ic_i = ic_m0.item()
                ic_full, ic_stab_full = AlphaEngine._compute_ic(res, t_ret)

            # 惩罚（纯函数，线程安全）
            reward = train_score
            val_score_out = val_score

            # 重复惩罚
            rp = _repetition_penalty(fml)
            if rp > 0:
                reward = reward - rp
                val_score_out = val_score_out - rp

            # 相关性惩罚（用 step 起始快照）
            if use_wf:
                _corr_slice = (folds[0]["train_start"], folds[0]["train_end"])
            else:
                _corr_slice = (0, max(int(res.shape[1] * 0.8), res.shape[1] - 100))
            reward = self._apply_corr_penalty(reward, res, _corr_slice)
            val_score_out = self._apply_corr_penalty(val_score_out, res, _corr_slice)
            behavior = None
            if bool(getattr(ModelConfig, "BEHAVIOR_DEDUP_ENABLED", True)):
                behavior = behavior_vector_from_factor(
                    res,
                    _corr_slice,
                    int(getattr(ModelConfig, "BEHAVIOR_VECTOR_SIZE", 512)),
                ).detach().cpu().tolist()

            return {
                'idx': idx, 'status': 'ok',
                'reward': reward.item() if isinstance(reward, torch.Tensor) else float(reward),
                'val_score': val_score_out.item() if isinstance(val_score_out, torch.Tensor) else float(val_score_out),
                'ic_full': ic_full.item(), 'ic_stab': ic_stab_full.item(),
                'ic_i': ic_i, 'res': res, 'fml': fml, 'behavior': behavior,
            }
        except Exception as e:
            return {'idx': idx, 'status': 'error', 'reward': -5.0,
                    'val_score': -5.0, 'fml': fml,
                    'error': f'{type(e).__name__}: {e}'}

    # ── IC computation ────────────────────────────────────────────────────────

    def _get_batch_pipeline(self):
        if self._batch_pipeline is None:
            from .gpu_batch import BatchFormulaPipeline
            self._batch_pipeline = BatchFormulaPipeline(
                cost_rate=self.bt.cost_rate,
                periods_per_year=self.bt.periods_per_year,
            )
        else:
            self._batch_pipeline.bt.cost_rate = self.bt.cost_rate
            self._batch_pipeline.bt.periods_per_year = self.bt.periods_per_year
        return self._batch_pipeline

    def _get_evaluator_router(self):
        from .evaluation_engines import (
            EvaluatorRouter,
            FastBatchFormulaEvaluator,
            ScoreGuard,
            StandardFormulaEvaluator,
        )
        if self._evaluator_router is None:
            standard = StandardFormulaEvaluator(self._eval_formula_task)
            fast = FastBatchFormulaEvaluator(self._eval_formula_batch_tasks_impl)
            guard = None
            if ModelConfig.EVALUATOR_GUARD and not ModelConfig.GPU_BATCH_EVAL:
                guard = ScoreGuard(
                    standard,
                    sample_size=ModelConfig.EVALUATOR_GUARD_SAMPLE,
                    every_n_steps=ModelConfig.EVALUATOR_GUARD_EVERY,
                    score_tol=ModelConfig.EVALUATOR_SCORE_TOL,
                    factor_tol=ModelConfig.EVALUATOR_FACTOR_TOL,
                )
            self._evaluator_router = EvaluatorRouter(
                standard=standard,
                fast=fast,
                guard=guard,
                mode=ModelConfig.EVALUATOR_ENGINE,
            )
        return self._evaluator_router

    def _eval_formula_batch_tasks_impl(
        self,
        formulas: list[list[int]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict]:
        """Batch VM/backtest path with the same public result contract."""
        if not use_wf:
            raise RuntimeError("batch evaluator currently requires walk-forward folds")
        try:
            pipe = self._get_batch_pipeline()
            with torch.no_grad():
                factors, valid = pipe.vm.execute_batch(formulas, feat)

            train_scores = torch.zeros(len(formulas), device=feat.device)
            val_scores = torch.zeros(len(formulas), device=feat.device)
            ic_sum = torch.zeros(len(formulas), device=feat.device)
            ic_count = torch.zeros(len(formulas), device=feat.device)

            with torch.no_grad():
                for fold in folds:
                    scores = pipe.bt.evaluate_fold_batch(
                        factors,
                        t_ret,
                        fold["train_start"], fold["train_end"],
                        fold["val_start"], fold["val_end"],
                    )
                    ic_m, ic_m_valid = AlphaEngine._compute_ic_mean_batch(
                        factors[:, :, fold["train_start"]:fold["train_end"]],
                        t_ret[:, fold["train_start"]:fold["train_end"]],
                    )
                    ic_v, _ = AlphaEngine._compute_ic_mean_batch(
                        factors[:, :, fold["val_start"]:fold["val_end"]],
                        t_ret[:, fold["val_start"]:fold["val_end"]],
                    )
                    active = valid
                    train_scores += torch.where(
                        valid,
                        ModelConfig.REWARD_ALPHA * AlphaEngine._apply_ic_gate(scores.train_scores, ic_m),
                        torch.zeros_like(train_scores),
                    )
                    val_scores += torch.where(
                        valid,
                        AlphaEngine._apply_ic_gate(scores.val_scores, ic_v),
                        torch.zeros_like(val_scores),
                    )
                    ic_sum += torch.where(active, ic_m, torch.zeros_like(ic_m))
                    ic_count += active.to(ic_count.dtype)
                train_scores = train_scores / max(1, len(folds))
                val_scores = val_scores / max(1, len(folds))
                ic_full_batch, ic_stab_batch = AlphaEngine._compute_ic_batch(factors, t_ret)
                factor_std = factors.reshape(factors.shape[0], -1).std(dim=1)
                valid_cpu = valid.detach().cpu().tolist()
                const_cpu = (factor_std < 1e-4).detach().cpu().tolist()

            corr_slice = (folds[0]["train_start"], folds[0]["train_end"])
            corr_penalty_cpu = self._corr_penalty_mask_batch(factors, corr_slice).detach().cpu().tolist()
            behavior_cpu = None
            if bool(getattr(ModelConfig, "BEHAVIOR_DEDUP_ENABLED", True)):
                behavior_cpu = behavior_vectors_from_factors(
                    factors,
                    corr_slice,
                    int(getattr(ModelConfig, "BEHAVIOR_VECTOR_SIZE", 512)),
                ).detach().cpu().tolist()
            results: list[dict] = []
            for i, fml in enumerate(formulas):
                if not valid_cpu[i]:
                    results.append({'idx': i, 'status': 'none', 'reward': -5.0,
                                    'val_score': -5.0, 'fml': fml})
                    continue

                res = factors[i]
                if const_cpu[i]:
                    results.append({'idx': i, 'status': 'const', 'reward': -2.0,
                                    'val_score': -2.0, 'fml': fml})
                    continue

                reward = train_scores[i]
                val_score_out = val_scores[i]
                rp = _repetition_penalty(fml)
                if rp > 0:
                    reward = reward - rp
                    val_score_out = val_score_out - rp

                if corr_penalty_cpu[i]:
                    reward = reward * ModelConfig.CORR_PENALTY
                    val_score_out = val_score_out * ModelConfig.CORR_PENALTY
                ic_i = (ic_sum[i] / ic_count[i].clamp(min=1)).item()
                results.append({
                    'idx': i, 'status': 'ok',
                    'reward': reward.item() if isinstance(reward, torch.Tensor) else float(reward),
                    'val_score': val_score_out.item() if isinstance(val_score_out, torch.Tensor) else float(val_score_out),
                    'ic_full': ic_full_batch[i].item(), 'ic_stab': ic_stab_batch[i].item(),
                    'ic_i': ic_i, 'res': res, 'fml': fml,
                    'behavior': behavior_cpu[i] if behavior_cpu is not None else None,
                })
            return results
        except Exception:
            if ModelConfig.GPU_BATCH_EVAL_STRICT:
                raise
            return [
                self._eval_formula_task(i, fml, feat, t_ret, folds, use_wf, factor_pool_snapshot)
                for i, fml in enumerate(formulas)
            ]

    def _eval_formula_batch_tasks(
        self,
        step: int,
        formulas: list[list[int]],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> list[dict]:
        """Evaluate one training step through the pluggable evaluator router."""
        router = self._get_evaluator_router()
        return router.evaluate(
            step=step,
            formulas=formulas,
            feat=feat,
            t_ret=t_ret,
            folds=folds,
            use_wf=use_wf,
            factor_pool_snapshot=factor_pool_snapshot,
            prefer_fast=bool(ModelConfig.GPU_BATCH_EVAL and use_wf),
        )

    @staticmethod
    def _compute_ic(factor: torch.Tensor, target_ret: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """时序 IC（每品种内部 factor[t] vs ret[t+1]）的均值与稳定性。

        对 5 品种宇宙，时序 IC 比横截面 IC 统计意义更强。
        """
        N, T = factor.shape
        if T < 2:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_list = []
        for n in range(N):
            x  = factor[n, :-1]
            y  = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_tensor = torch.stack(ic_list)
        ic_mean   = ic_tensor.mean()
        ic_stab   = (ic_mean / (ic_tensor.std(unbiased=False) + 1e-6)
                     if ic_tensor.numel() >= 2
                     else torch.zeros(1, device=factor.device))
        return ic_mean, ic_stab

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _compute_ic_batch(
        factors: torch.Tensor,
        target_ret: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch version of _compute_ic for factors shaped [B,N,T]."""
        if factors.ndim != 3:
            raise ValueError(f"factors must be [B,N,T], got {tuple(factors.shape)}")
        bsz = factors.shape[0]
        if factors.shape[2] < 2:
            z = torch.zeros(bsz, device=factors.device, dtype=factors.dtype)
            return z, z

        x = factors[:, :, :-1]
        y = target_ret[None, :, 1:].expand_as(x)
        xm = x - x.mean(dim=2, keepdim=True)
        ym = y - y.mean(dim=2, keepdim=True)
        sx = xm.square().mean(dim=2).sqrt()
        sy = ym.square().mean(dim=2).sqrt()
        valid = (sx >= 1e-6) & (sy >= 1e-6)
        ic = (xm * ym).mean(dim=2) / (sx * sy + 1e-8)
        ic = torch.where(valid, ic, torch.zeros_like(ic))
        denom = valid.sum(dim=1).clamp(min=1)
        ic_mean = ic.sum(dim=1) / denom
        centered = torch.where(valid, ic - ic_mean[:, None], torch.zeros_like(ic))
        ic_std = (centered.square().sum(dim=1) / denom).sqrt()
        ic_stab = torch.where(
            valid.sum(dim=1) >= 2,
            ic_mean / (ic_std + 1e-6),
            torch.zeros_like(ic_mean),
        )
        return ic_mean, ic_stab

    @staticmethod
    def _compute_ic_mean_batch(
        factors: torch.Tensor,
        target_ret: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batch IC mean and whether each formula had any valid symbol."""
        if factors.ndim != 3:
            raise ValueError(f"factors must be [B,N,T], got {tuple(factors.shape)}")
        bsz = factors.shape[0]
        if factors.shape[2] < 2:
            z = torch.zeros(bsz, device=factors.device, dtype=factors.dtype)
            return z, torch.zeros(bsz, device=factors.device, dtype=torch.bool)
        x = factors[:, :, :-1]
        y = target_ret[None, :, 1:].expand_as(x)
        xm = x - x.mean(dim=2, keepdim=True)
        ym = y - y.mean(dim=2, keepdim=True)
        sx = xm.square().mean(dim=2).sqrt()
        sy = ym.square().mean(dim=2).sqrt()
        valid = (sx >= 1e-6) & (sy >= 1e-6)
        ic = (xm * ym).mean(dim=2) / (sx * sy + 1e-8)
        ic = torch.where(valid, ic, torch.zeros_like(ic))
        denom = valid.sum(dim=1).clamp(min=1)
        return ic.sum(dim=1) / denom, valid.any(dim=1)

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """IC 门控：用 IC 符号而非量值调整 reward，完全规避量纲问题。
        IC > thresh  → reward × IC_GATE_MULT  (正向预测，奖励)
        IC < -thresh → reward × IC_NEG_MULT   (反向预测，惩罚)
        |IC| ≤ thresh→ 不修改                  (噪声区)
        """
        if isinstance(ic_mean, torch.Tensor) and ic_mean.numel() > 1:
            t = ModelConfig.IC_GATE_THRESH
            return torch.where(
                ic_mean > t,
                reward * ModelConfig.IC_GATE_MULT,
                torch.where(ic_mean < -t, reward * ModelConfig.IC_NEG_MULT, reward),
            )
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return reward * ModelConfig.IC_GATE_MULT
        elif ic_val < -t:
            return reward * ModelConfig.IC_NEG_MULT
        return reward


    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _elite_bucket_key(formula: list[int]) -> tuple:
        """Group similar formulas so the elite pool keeps several directions."""
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
        return (first, min(feat_cnt, 3), min(ts_cnt, 3), min(arith_cnt, 2), min(norm_cnt, 2), min(nonlinear_cnt, 2))

    @classmethod
    def _rebalance_elite_pool(
        cls,
        pool: list[tuple[float, int, list[int], int]],
    ) -> list[tuple[float, int, list[int], int]]:
        bucket_cap = max(1, int(getattr(ModelConfig, "ELITE_BUCKET_CAP", 3)))
        global_cap = max(1, int(ModelConfig.ELITE_POOL_SIZE))
        by_formula: dict[tuple[int, ...], tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = tuple(int(t) for t in toks)
            entry = (float(sc), int(cnt), [int(t) for t in toks], int(birth))
            if key not in by_formula or entry[0] > by_formula[key][0]:
                by_formula[key] = entry
        buckets: dict[tuple, list[tuple[float, int, list[int], int]]] = {}
        for entry in by_formula.values():
            buckets.setdefault(cls._elite_bucket_key(entry[2]), []).append(entry)
        kept: list[tuple[float, int, list[int], int]] = []
        for entries in buckets.values():
            kept.extend(sorted(entries, key=lambda x: (x[0], x[1]), reverse=True)[:bucket_cap])
        kept = sorted(kept, key=lambda x: (x[0], x[1]), reverse=True)[:global_cap]
        heapq.heapify(kept)
        return kept

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        return AlphaEngine._rebalance_elite_pool(pool)

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        entry = (float(val_score), self._elite_counter, [int(t) for t in formula], int(step))
        self._elite_counter += 1
        before = {(sc, tuple(toks), birth) for sc, _cnt, toks, birth in self._elite_pool}
        rebalanced = self._rebalance_elite_pool(self._elite_pool + [entry])
        after = {(sc, tuple(toks), birth) for sc, _cnt, toks, birth in rebalanced}
        if after != before:
            self._elite_pool = rebalanced

    @classmethod
    def _rebalance_incubation_pool(
        cls,
        pool: list[tuple[float, int, list[int], int]],
        step: int,
    ) -> list[tuple[float, int, list[int], int]]:
        max_age = max(1, int(getattr(ModelConfig, "INCUBATION_CAPTURE_STEPS", 180)))
        bucket_cap = max(1, int(getattr(ModelConfig, "INCUBATION_BUCKET_CAP", 2)))
        global_cap = max(1, int(getattr(ModelConfig, "INCUBATION_POOL_SIZE", 36)))
        min_score = float(getattr(ModelConfig, "INCUBATION_MIN_SCORE", -0.5))
        fresh = [
            (float(sc), int(cnt), [int(t) for t in toks], int(birth))
            for sc, cnt, toks, birth in pool
            if step - int(birth) <= max_age and float(sc) >= min_score
        ]
        by_formula: dict[tuple[int, ...], tuple[float, int, list[int], int]] = {}
        for entry in fresh:
            key = tuple(entry[2])
            if key not in by_formula or entry[0] > by_formula[key][0]:
                by_formula[key] = entry
        buckets: dict[tuple, list[tuple[float, int, list[int], int]]] = {}
        for entry in by_formula.values():
            buckets.setdefault(cls._elite_bucket_key(entry[2]), []).append(entry)
        kept: list[tuple[float, int, list[int], int]] = []
        for entries in buckets.values():
            kept.extend(sorted(entries, key=lambda x: (x[0], -x[3], x[1]), reverse=True)[:bucket_cap])
        return sorted(kept, key=lambda x: (x[0], -x[3], x[1]), reverse=True)[:global_cap]

    def _update_incubation_pool(self, val_score: float, formula: list[int], step: int) -> None:
        entry = (float(val_score), self._incubation_counter, [int(t) for t in formula], int(step))
        self._incubation_counter += 1
        self._incubation_pool = self._rebalance_incubation_pool(
            self._incubation_pool + [entry], step
        )

    def _sample_incubation_formulas(self, step: int, k: int) -> tuple[list[list[int]], dict]:
        self._incubation_pool = self._rebalance_incubation_pool(self._incubation_pool, step)
        if not self._incubation_pool or k <= 0:
            return [], {"cells": 0, "scores": [], "max_age": 0}
        buckets: dict[tuple, list[tuple[float, int, list[int], int]]] = {}
        ages: list[int] = []
        for entry in self._incubation_pool:
            ages.append(max(0, step - entry[3]))
            buckets.setdefault(self._elite_bucket_key(entry[2]), []).append(entry)
        keys = list(buckets.keys())
        formulas: list[list[int]] = []
        scores: list[float] = []
        for key in random.choices(keys, k=k):
            entries = buckets[key]
            ps = [max(0.01, e[0] - min(0.0, min(x[0] for x in entries)) + 0.01) for e in entries]
            chosen = random.choices(entries, weights=ps, k=1)[0]
            formulas.append(list(chosen[2]))
            scores.append(float(chosen[0]))
        return formulas, {"cells": len(buckets), "scores": scores, "max_age": max(ages) if ages else 0}

    def _sample_elite_formulas(self, step: int, k: int) -> tuple[list[list[int]], dict]:
        if not self._elite_pool or k <= 0:
            return [], {"avg_decay": 0.0, "max_age": 0, "age_list": [], "scores": [], "cells": 0}

        buckets: dict[tuple, list[tuple[float, int, list[int], int, float]]] = {}
        ages: list[int] = []
        decays: list[float] = []
        for sc, cnt, toks, birth in self._elite_pool:
            age = max(0, step - birth)
            decay = 1.0
            if ModelConfig.ELITE_DECAY:
                half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                decay = 0.5 ** (age / half)
            ages.append(age)
            decays.append(decay)
            buckets.setdefault(self._elite_bucket_key(toks), []).append((sc, cnt, toks, birth, decay))

        keys = list(buckets.keys())
        formulas: list[list[int]] = []
        scores: list[float] = []
        for key in random.choices(keys, k=k):
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

    # ── Factor pool ───────────────────────────────────────────────────────────

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, self._factor_pool_counter, f_gpu)
        self._factor_pool_counter += 1
        if len(self.factor_pool) < k:
            heapq.heappush(self.factor_pool, entry)
        elif val_score > self.factor_pool[0][0]:
            heapq.heapreplace(self.factor_pool, entry)

    def _apply_corr_penalty(
        self,
        reward: torch.Tensor,
        factor: torch.Tensor,
        train_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """相关性惩罚：与因子池中已有因子的相关性超过阈值则惩罚 reward。

        P1-6 修复：相关性只在 train 切片上计算，避免含 val 段数据泄漏。
        train_slice=None 时回退到整段（向后兼容）。
        """
        if not self.factor_pool:
            return reward
        # P1-6: 相关性只在 train 切片上计算，避免 val 信息泄漏
        if train_slice is not None:
            s, e = train_slice
            f = factor.detach()[:, s:e]
        else:
            f = factor.detach()
        f_flat = f.reshape(-1).float()
        if f_flat.std() < 1e-4:
            return reward
        # 因子池中的历史因子也按相同切片取（若形状一致）
        pool_vecs_list = []
        for _, _cnt, pf in self.factor_pool:
            pf_t = pf.detach()
            if train_slice is not None and pf_t.shape[1] >= factor.shape[1]:
                pf_t = pf_t[:, s:e]
            pool_vecs_list.append(pf_t.reshape(-1).float())
        if not pool_vecs_list:
            return reward
        pool_vecs = torch.stack(pool_vecs_list, dim=0)
        f_c  = f_flat - f_flat.mean()
        p_c  = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov  = (p_c * f_c).sum(dim=1)
        sx   = f_c.norm() + 1e-8
        sy   = p_c.norm(dim=1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward * ModelConfig.CORR_PENALTY
        return reward

    def _apply_corr_penalty_batch(
        self,
        rewards: torch.Tensor,
        factors: torch.Tensor,
        train_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Batch version of _apply_corr_penalty for factors shaped [B,N,T]."""
        penalize = self._corr_penalty_mask_batch(factors, train_slice)
        return torch.where(penalize, rewards * ModelConfig.CORR_PENALTY, rewards)

    def _corr_penalty_mask_batch(
        self,
        factors: torch.Tensor,
        train_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Return which batch rows would be penalized by _apply_corr_penalty."""
        if not self.factor_pool:
            return torch.zeros(factors.shape[0], dtype=torch.bool, device=factors.device)
        if train_slice is not None:
            s, e = train_slice
            f = factors.detach()[:, :, s:e]
        else:
            f = factors.detach()
        bsz = f.shape[0]
        f_flat = f.reshape(bsz, -1).float()
        f_c = f_flat - f_flat.mean(dim=1, keepdim=True)
        f_std = f_flat.std(dim=1)

        pool_vecs_list = []
        for _, _cnt, pf in self.factor_pool:
            pf_t = pf.detach()
            if train_slice is not None and pf_t.shape[1] >= factors.shape[2]:
                pf_t = pf_t[:, s:e]
            pool_vecs_list.append(pf_t.reshape(-1).float())
        if not pool_vecs_list:
            return torch.zeros(factors.shape[0], dtype=torch.bool, device=factors.device)

        pool_vecs = torch.stack(pool_vecs_list, dim=0).to(f_flat.device)
        p_c = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov = torch.matmul(f_c, p_c.transpose(0, 1))
        sx = f_c.norm(dim=1, keepdim=True) + 1e-8
        sy = p_c.norm(dim=1).view(1, -1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        penalize = (f_std >= 1e-4) & (corr > ModelConfig.CORR_THRESHOLD).any(dim=1)
        return penalize

    def _distribution_stats(self, prev_dist=None):
        """计算模型初始位置（zero prefix）token 分布的细化指标，用于判断 H 不变时
        分布是否真的在变化。
        """
        vocab_size = FORMULA_VOCAB.size
        with torch.no_grad():
            inp = torch.zeros((1, 1), dtype=torch.long,
                              device=ModelConfig.DEVICE)
            logits, _, _ = self.model(inp)
            logits = self.sampler.apply_mask_to_logits(
                logits, [0], 0, ModelConfig.MAX_FORMULA_LEN
            )
            dist = F.softmax(logits, dim=-1).squeeze(0)
            ent = -(dist * torch.log(dist + 1e-12)).sum().item()
            log_v = math.log(vocab_size)
            kl_uniform = log_v - ent
            top1 = dist.max().item()
            top5 = dist.topk(5, dim=-1).values.sum().item()
            eff_vocab = math.exp(ent)
            prob_std = dist.std(unbiased=False).item()
            kl_prev = 0.0
            if prev_dist is not None:
                kl_prev = (
                    dist * (torch.log(dist + 1e-12) -
                            torch.log(prev_dist.to(dist.device) + 1e-12))
                ).sum().item()
        return {
            'dist': dist.cpu(),
            'entropy': ent,
            'kl_uniform': kl_uniform,
            'top1_prob': top1,
            'top5_prob': top5,
            'eff_vocab': eff_vocab,
            'prob_std': prob_std,
            'kl_prev': kl_prev,
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        if self.data_manager is None:
            raise RuntimeError("AlphaEngine requires a data_manager.")

        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS

        if verbose_header:
            print("开始 Alpha 因子挖掘训练" +
                  ("（含 LoRD 正则化）..." if self.use_lord else "..."))
            print(f"   策略熵: 坍塌阈值={ModelConfig.ENTROPY_COLLAPSE_THRESH}  "
                  f"系数上限={ModelConfig.ENTROPY_COEFF_MAX}  "
                  f"连续坍塌步数={ModelConfig.ENTROPY_COLLAPSE_STEPS}")
            print(f"   精英回放: 比例={ModelConfig.ELITE_REPLAY_FRAC}  "
                  f"池大小={ModelConfig.ELITE_POOL_SIZE}")
            print(f"   IC门控: 阈值±{ModelConfig.IC_GATE_THRESH}  "
                  f"正向×{ModelConfig.IC_GATE_MULT}  负向×{ModelConfig.IC_NEG_MULT}")
            print(f"   最大重启: {ModelConfig.MAX_RESTARTS}  "
                  f"噪声={ModelConfig.RESTART_NOISE}")

        T     = self.data_manager.target_ret.shape[1]
        folds = _build_walk_forward_folds(T, self.n_folds,
                                          gap=getattr(ModelConfig, 'WF_GAP', 20))
        use_wf = len(folds) > 1 and not (
            folds[0]["train_start"] == 0 and folds[0]["train_end"] == T
        )
        if verbose_header:
            if use_wf:
                print(f"   滚动验证: {len(folds)} 折  共 {T} 根K线")
                for k, f in enumerate(folds):
                    print(f"  第{k+1}折: 训练[{f['train_start']},{f['train_end']}) "
                          f"间隔={f['gap']} 验证[{f['val_start']},{f['val_end']})")
            else:
                print(f"   退化为全量评估（共 {T} 根K线）")

        # 因果安全：features.py 的 _robust_norm 已改为滚动因果实现
        # 每个 t 的归一化参数只用 [t-w+1..t]，walk-forward 折叠切片无泄露
        feat  = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)

        # 数据驱动年化因子：按训练数据的实际时间戳估计每年 bar 数，
        # 替代 MT5Backtest 默认的 H1=6240。A 股日线/15min、加密日线等
        # 非 H1 周期不再被按 H1 年化（否则 Sharpe/年化收益被放大数倍）。
        _dm_raw = getattr(self.data_manager, "raw_dict", None) or {}
        _dm_time = _dm_raw.get("time", None)
        if _dm_time is not None:
            try:
                _ppy = estimate_periods_per_year(_dm_time)
                if _ppy != self.bt.periods_per_year:
                    if verbose_header:
                        print(f"   年化因子: {_ppy} bar/年（按数据周期自动估计）")
                    self.bt.periods_per_year = _ppy
            except Exception:
                pass  # 估计失败则保留默认 6240

        bs      = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new   = bs - n_elite

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[训练] 起始步 {start_step} 已达目标步 {end_step}，无需继续训练。")
            return

        # 非交互/重定向输出时关闭 tqdm 进度条，避免进度条刷屏把自定义日志淹掉。
        # tqdm.write 仍然可用，详细 step 日志会继续输出。
        pbar               = tqdm(range(start_step, end_step),
                                  total=end_step,
                                  initial=start_step,
                                  disable=not sys.stderr.isatty(),
                                  leave=False,
                                  mininterval=5.0)
        low_entropy_streak = 0
        prev_init_dist     = None  # 用于计算相邻步分布差异 KL

        for step in pbar:
            timing_step0 = time.perf_counter()
            steps_since_restart = step - self._last_restart_step
            timing_replay0 = time.perf_counter()
            n_new, replay_batch = self.replay_policy.plan(
                step=step,
                batch_size=bs,
                last_restart_step=self._last_restart_step,
            )
            timing_replay_plan_ms = (time.perf_counter() - timing_replay0) * 1000.0
            n_incubate = replay_batch.n_incubation
            n_elite = replay_batch.n_elite
            elite_frac_eff = replay_batch.elite_frac_effective
            timing_search0 = time.perf_counter()
            search_batch = self.search_plugins.plan(
                step=step,
                candidate_slots=n_new,
                best_formula=self.best_formula,
                elite_pool=self.replay_policy.elite_entries(),
            )
            timing_search_plugin_ms = (time.perf_counter() - timing_search0) * 1000.0
            plugin_formulas = search_batch.formulas
            plugin_origins = search_batch.origins
            n_plugin = len(plugin_formulas)
            n_policy = max(0, n_new - n_plugin)
            memory_formulas = replay_batch.formulas
            n_memory = len(memory_formulas)
            # ── Part A: Sample n_new new formulas ────────────────────
            timing_policy_sample0 = time.perf_counter()
            timing_ab_forward_ms = 0.0
            timing_policy_dist_ms = 0.0
            timing_memory_dist_ms = 0.0
            inp_new_full = torch.zeros(
                (n_policy, ModelConfig.MAX_FORMULA_LEN + 1),
                dtype=torch.long,
                device=ModelConfig.DEVICE,
            )
            lp_new, tok_new, ent_new = [], [], []
            lp_elite, ent_elite = [], []
            sd_new = torch.zeros(n_policy, dtype=torch.long, device=ModelConfig.DEVICE)
            infected_chain_new = torch.zeros(n_policy, dtype=torch.long, device=ModelConfig.DEVICE)
            inp_e_full = None
            tok_e_t = None
            sd_e = torch.zeros(0, dtype=torch.long, device=ModelConfig.DEVICE)
            infected_chain_elite = torch.zeros(0, dtype=torch.long, device=ModelConfig.DEVICE)
            if n_memory > 0:
                inp_e_full = torch.zeros(
                    (n_memory, ModelConfig.MAX_FORMULA_LEN + 1),
                    dtype=torch.long,
                    device=ModelConfig.DEVICE,
                )
                tok_e_t = torch.tensor(memory_formulas, dtype=torch.long, device=ModelConfig.DEVICE)
                inp_e_full[:, 1:ModelConfig.MAX_FORMULA_LEN + 1] = tok_e_t
                sd_e = torch.zeros(n_memory, dtype=torch.long, device=ModelConfig.DEVICE)
                infected_chain_elite = torch.zeros(n_memory, dtype=torch.long, device=ModelConfig.DEVICE)

            if n_policy > 0 or n_memory > 0:
                n_ab = n_policy + n_memory
                inp_ab_full = torch.zeros(
                    (n_ab, ModelConfig.MAX_FORMULA_LEN + 1),
                    dtype=torch.long,
                    device=ModelConfig.DEVICE,
                )
                if n_memory > 0 and tok_e_t is not None:
                    inp_ab_full[n_policy:, 1:ModelConfig.MAX_FORMULA_LEN + 1] = tok_e_t
                delta_t, infected_t, positive_t, restore_t = self.sampler._constraint_tensors(ModelConfig.DEVICE)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    if n_policy > 0:
                        inp_ab_full[:n_policy, :si + 1] = inp_new_full[:, :si + 1]
                    inp_ab = inp_ab_full[:, :si + 1].clone()
                    timing_ab_forward0 = time.perf_counter()
                    lg_ab, _, _ = self.model(inp_ab)
                    timing_ab_forward_ms += (time.perf_counter() - timing_ab_forward0) * 1000.0
                    if n_policy > 0:
                        timing_policy_dist0 = time.perf_counter()
                        lg_new = self.sampler.apply_mask_to_logits(
                            lg_ab[:n_policy],
                            sd_new,
                            si,
                            ModelConfig.MAX_FORMULA_LEN,
                            infected_chain_lens=infected_chain_new,
                        )
                        a, lp, ent = _categorical_stats_from_logits(lg_new)
                        lp_new.append(lp)
                        tok_new.append(a)
                        ent_new.append(ent)
                        inp_new_full[:, si + 1] = a
                        sd_new = sd_new + delta_t[a]
                        a_positive = positive_t[a]
                        a_restore = restore_t[a]
                        a_infected = infected_t[a]
                        infected_chain_new = torch.where(
                            a_positive,
                            infected_chain_new + 1,
                            torch.where(
                                a_restore,
                                torch.zeros_like(infected_chain_new),
                                torch.where(
                                    a_infected & (infected_chain_new > 0),
                                    infected_chain_new + 1,
                                    infected_chain_new,
                                ),
                            ),
                        )
                        timing_policy_dist_ms += (time.perf_counter() - timing_policy_dist0) * 1000.0
                    if n_memory > 0 and tok_e_t is not None and inp_e_full is not None:
                        timing_memory_dist0 = time.perf_counter()
                        lg_e = self.sampler.apply_mask_to_logits(
                            lg_ab[n_policy:],
                            sd_e,
                            si,
                            ModelConfig.MAX_FORMULA_LEN,
                            infected_chain_lens=infected_chain_elite,
                        )
                        tk = tok_e_t[:, si]
                        _, lp_e, ent_e = _categorical_stats_from_logits(lg_e, tk)
                        lp_elite.append(lp_e)
                        ent_elite.append(ent_e)
                        sd_e = sd_e + delta_t[tk]
                        tk_positive = positive_t[tk]
                        tk_restore = restore_t[tk]
                        tk_infected = infected_t[tk]
                        infected_chain_elite = torch.where(
                            tk_positive,
                            infected_chain_elite + 1,
                            torch.where(
                                tk_restore,
                                torch.zeros_like(infected_chain_elite),
                                torch.where(
                                    tk_infected & (infected_chain_elite > 0),
                                    infected_chain_elite + 1,
                                    infected_chain_elite,
                                ),
                            ),
                        )
                        timing_memory_dist_ms += (time.perf_counter() - timing_memory_dist0) * 1000.0

            if tok_new:
                seqs_new_list = torch.stack(tok_new, dim=1).tolist()
            else:
                seqs_new_list = []
            timing_policy_sample_ms = timing_ab_forward_ms + timing_policy_dist_ms
            timing_memory_logprob_ms = timing_memory_dist_ms


            # ── Part B: Elite Replay ─────────────────────────────────
            elite_sample_info = replay_batch.elite_info or {"avg_decay": 0.0, "max_age": 0, "age_list": [], "scores": [], "cells": 0}
            incubation_sample_info = replay_batch.incubation_info or {"max_age": 0, "scores": [], "cells": 0}
            if n_elite > 0:
                if step % 100 == 0:
                    tqdm.write(
                        f"[QD优秀池 @ 第{step}步] 类型={elite_sample_info['cells']} "
                        f"回放={n_elite}/{bs} 有效比例={elite_frac_eff:.3f} "
                        f"平均衰减={elite_sample_info['avg_decay']:.3f} "
                        f"最大年龄={elite_sample_info['max_age']} "
                        f"抽样分数=[{', '.join(f'{s:.3f}' for s in elite_sample_info['scores'][:3])}...]"
                    )
            if n_incubate > 0 and step % 50 == 0:
                tqdm.write(
                    f"[新方向孵化池 @ 第{step}步] 类型={incubation_sample_info['cells']} "
                    f"回放={n_incubate}/{bs} 最大年龄={incubation_sample_info['max_age']} "
                    f"抽样分数=[{', '.join(f'{s:.3f}' for s in incubation_sample_info['scores'][:3])}...]"
                )
            if False and self._elite_pool and n_elite > 0:
                ps = []
                pt = []
                weights = []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc)
                    pt.append(toks)
                    weights.append(decay)
                ps_min  = min(ps)
                ps_max  = max(ps)
                # 软温度采样：避免最高分公式垄断
                # 先归一到 [0,1]，再除以温度 T=0.5 后做 softmax
                # T<1 → 高分公式仍被偏好，但不再独占
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp)) for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e   = random.choices(range(len(self._elite_pool)),
                                         weights=probs, k=n_elite)
                elite_formulas = [pt[i] for i in idx_e]

                # 详细日志：Elite Replay 衰减状态（每 100 步打印一次）
                if step % 100 == 0:
                    avg_decay = sum(weights) / len(weights)
                    max_age = max(max(0, step - birth) for _, _, _, birth in self._elite_pool)
                    age_list = sorted([max(0, step - birth) for _, _, _, birth in self._elite_pool])
                    tqdm.write(
                        f"[精英回放 @ 第{step}步] 池大小={len(self._elite_pool)} "
                        f"平均衰减={avg_decay:.3f} 最大龄期={max_age} 龄期列表={age_list} "
                        f"抽样分数=[{', '.join(f'{ps[i]:.3f}' for i in idx_e[:3])}...]"
                    )
            else:
                pass

            timing_ab_ms = (time.perf_counter() - timing_step0) * 1000.0

            # ── Part C: Evaluate all formulas (并行评估) ────────────────
            timing_eval0 = time.perf_counter()
            all_fmls = seqs_new_list + plugin_formulas + memory_formulas
            formula_origins = (["policy"] * len(seqs_new_list)) + plugin_origins + (["memory"] * len(memory_formulas))
            tot      = len(all_fmls)

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf');  step_best_f = None
            new_step_max_val = -float('inf');  new_step_best_f = None
            bic, bis, bsor = [], [], []
            reward_values = [0.0] * tot
            val_score_values = [0.0] * tot
            replay_observations: list[dict[str, Any]] = []

            # factor_pool 快照：所有 worker 看到同一份只读视图
            factor_pool_snapshot = list(self.factor_pool)

            # 并行提交所有公式评估任务
            if ModelConfig.GPU_BATCH_EVAL and use_wf:
                results = self._eval_formula_batch_tasks(
                    step, all_fmls, feat, t_ret, folds, use_wf, factor_pool_snapshot,
                )
            elif self._eval_pool is not None and self._eval_workers > 1 and tot > 1:
                from concurrent.futures import ThreadPoolExecutor
                futures = [
                    self._eval_pool.submit(
                        self._eval_formula_task, i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                    )
                    for i, fml in enumerate(all_fmls)
                ]
                results_by_idx: dict[int, dict] = {}
                for fut in futures:
                    r = fut.result()
                    results_by_idx[r['idx']] = r
                results = [results_by_idx[i] for i in range(tot)]
            else:
                # 串行回退
                results = [
                    self._eval_formula_task(
                        i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                    )
                    for i, fml in enumerate(all_fmls)
                ]
            timing_eval_ms = (time.perf_counter() - timing_eval0) * 1000.0

            # ── 串行后处理：写入 rewards/val_scores，更新冠军/池 ─────────
            for r in results:
                i = r['idx']
                status = r.get('status', 'error')
                reward_values[i] = float(r['reward'])
                val_score_values[i] = float(r['val_score'])

                if status == 'none':
                    none_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue
                if status == 'const':
                    const_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue
                if status == 'error':
                    none_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue

                ok_cnt += 1
                bic.append(r['ic_full']); bis.append(r['ic_stab']); bsor.append(r['val_score'])
                fml = r['fml']
                res = r['res']
                ic_i = r.get('ic_i', 0.0)
                final_val = r['val_score']

                if final_val > step_max_val:
                    step_max_val = final_val; step_best_f = fml
                if i < n_policy + n_plugin and final_val > new_step_max_val:
                    new_step_max_val = final_val; new_step_best_f = fml

                if final_val > self.best_score:
                    # OOS 泛化门控：val_score / train_score < 0.5 说明过拟合
                    train_val = r['reward']
                    if train_val > 0.5 and final_val < train_val * 0.5:
                        tqdm.write(
                            f"[过拟合跳过 @ 第{step}步] 验证={final_val:.3f} "
                            f"训练={train_val:.3f} 比值={final_val/train_val:.2f} | 样本外表现过差"
                        )
                        pass
                    else:
                        pos_check = compute_target_positions_stateless(res)
                        exposure = pos_check.abs().mean().item()
                        if exposure < 0.05:
                            tqdm.write(
                                f"[稀疏跳过 @ 第{step}步] 验证={final_val:.3f} "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | 仓位过稀疏，不更新最优"
                            )
                            pass
                        else:
                            old_best = self.best_score
                            self.best_score   = final_val
                            self.best_formula = fml
                            self._best_snapshot = copy.deepcopy(self.model.state_dict())
                            self._best_update_step = step
                            self._stagnation_steps = 0
                            self._update_factor_pool(final_val, res)
                            self._save_strategy_live()
                            tqdm.write(
                                f"[!] 新最优 @ 第{step}步: 验证={final_val:.3f} "
                                f"(原 {old_best:.3f}，+{final_val-old_best:.3f}) "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | "
                                f"{fml}\n    {self._decode_formula(fml)}"
                            )
                replay_observations.append({
                    "score": final_val,
                    "formula": fml,
                    "is_new": i < n_policy + n_plugin,
                    "behavior": r.get("behavior"),
                })

            rewards = torch.tensor(reward_values, dtype=torch.float32, device=ModelConfig.DEVICE)
            val_scores = torch.tensor(val_score_values, dtype=torch.float32, device=ModelConfig.DEVICE)
            self.replay_policy.observe_many(
                replay_observations,
                step=step,
                last_restart_step=self._last_restart_step,
            )

            plugin_results = [
                results[i] for i, origin in enumerate(formula_origins)
                if origin in {"annealing", "genetic"}
            ]
            plugin_result_origins = [
                origin for origin in formula_origins
                if origin in {"annealing", "genetic"}
            ]
            self.search_plugins.observe(step=step, results=plugin_results, origins=plugin_result_origins)


            # ── Part D: REINFORCE gradient update ────────────────────
            # Fix 3: EMA baseline 替代 batch mean，避免全负 batch 的相对优选问题
            batch_mean = rewards.mean().item()
            timing_grad0 = time.perf_counter()
            timing_loss0 = time.perf_counter()
            batch_std  = rewards.std().clamp(min=0.1)
            if ModelConfig.REWARD_EMA_BASELINE and self._reward_ema_step >= ModelConfig.REWARD_EMA_WARMUP:
                baseline = self._reward_ema
                adv = (rewards - baseline) / (batch_std + 1e-5)
            else:
                adv = (rewards - batch_mean) / (batch_std + 1e-5)
            # 更新 EMA
            if self._reward_ema is None:
                self._reward_ema = batch_mean
            else:
                self._reward_ema = ModelConfig.REWARD_EMA_DECAY * self._reward_ema + (1.0 - ModelConfig.REWARD_EMA_DECAY) * batch_mean
            self._reward_ema_step += 1
            adv_new   = adv[:n_policy]
            adv_elite = adv[n_policy + n_plugin:]

            policy_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            if lp_new:
                lp_new_t = torch.stack(lp_new, dim=0)
                policy_loss = policy_loss + (-(lp_new_t * adv_new.unsqueeze(0))).mean(dim=1).sum()
            if lp_elite and adv_elite.shape[0] > 0:
                lp_elite_t = torch.stack(lp_elite, dim=0)
                if lp_elite_t.shape[1] == adv_elite.shape[0]:
                    policy_loss = policy_loss + (
                        -(lp_elite_t * adv_elite.unsqueeze(0) * ModelConfig.ELITE_REWARD_SCALE)
                    ).mean(dim=1).sum()

            if ent_new:
                mean_ent_new = torch.stack(ent_new).mean()
            else:
                mean_ent_new = torch.zeros(1, device=ModelConfig.DEVICE)
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (
                    mean_ent_new * n_policy + mean_ent_elite * n_memory
                ) / max(1, n_policy + n_memory)
            else:
                mean_ent = mean_ent_new
            ent_val   = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER
            )
            # Fix 1: 熵下限惩罚——当 H < threshold 时加入固定惩罚，确保探索压力不归零
            ent_floor_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            if ModelConfig.ENTROPY_FLOOR and ent_val < ModelConfig.ENTROPY_FLOOR_THRESH:
                floor_gap = ModelConfig.ENTROPY_FLOOR_THRESH - ent_val
                ent_floor_loss = ModelConfig.ENTROPY_FLOOR_LAMBDA * torch.tensor(
                    floor_gap, device=ModelConfig.DEVICE, dtype=mean_ent.dtype
                )
            loss = policy_loss - ent_coeff * mean_ent + ent_floor_loss
            timing_loss_build_ms = (time.perf_counter() - timing_loss0) * 1000.0

            self.opt.zero_grad(set_to_none=True)
            timing_backward0 = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            timing_backward_ms = (time.perf_counter() - timing_backward0) * 1000.0
            timing_optimizer0 = time.perf_counter()
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()
            timing_optimizer_ms = (time.perf_counter() - timing_optimizer0) * 1000.0
            timing_grad_ms = (time.perf_counter() - timing_grad0) * 1000.0

            # ── Part D2: 分布细化指标 ────────────────────────────────
            timing_dist0 = time.perf_counter()
            dst = self._distribution_stats(prev_init_dist)
            prev_init_dist = dst['dist']
            with torch.no_grad():
                if seqs_new_list:
                    seqs_new_tensor = torch.tensor(seqs_new_list, dtype=torch.long, device=ModelConfig.DEVICE)
                    uniq_tokens = seqs_new_tensor.unique().numel()
                    uniq_fmls = torch.unique(seqs_new_tensor, dim=0).shape[0]
                else:
                    uniq_tokens = 0
                    uniq_fmls = 0
                fml_div     = uniq_fmls / max(1, n_new)
            timing_dist_stats_ms = (time.perf_counter() - timing_dist0) * 1000.0

            # ── Part E: Logging & history & checkpoint ───────────────
            avg_rew = rewards.mean().item()
            avg_val = val_scores.mean().item()
            bim  = sum(bic)  / len(bic)  if bic  else 0.0
            bis_ = sum(bis)  / len(bis)  if bis  else 0.0
            bsor_= sum(bsor) / len(bsor) if bsor else 0.0
            timing_total_ms = (time.perf_counter() - timing_step0) * 1000.0
            timing_rest_ms = max(
                0.0,
                timing_total_ms - timing_ab_ms - timing_eval_ms - timing_grad_ms,
            )
            eval_router = self._evaluator_router
            eval_engine_name = getattr(eval_router, "last_engine", "legacy")
            guard_report = getattr(eval_router, "last_guard_report", None)
            guard_passed = getattr(guard_report, "passed", None) if guard_report else None
            guard_checked = getattr(guard_report, "checked", 0) if guard_report else 0
            guard_reward_diff = getattr(guard_report, "max_reward_diff", None) if guard_report else None
            guard_val_diff = getattr(guard_report, "max_val_diff", None) if guard_report else None
            guard_factor_diff = getattr(guard_report, "max_factor_diff", None) if guard_report else None

            self._stagnation_steps = step - self._best_update_step
            replay_metrics = self.replay_policy.metrics()
            search_metrics = self.search_plugins.metrics()
            tqdm.write(
                f"[{step+1}/{end_step}] "
                f"模型={n_policy} 搜索={n_plugin} 孵化={n_incubate} 精英={n_elite} | "
                f"有效={ok_cnt} 无效={none_cnt} 常数={const_cnt} | "
                f"奖励={avg_rew:.3f} 验证={avg_val:.3f} | "
                f"IC={bim:.4f} | 熵={ent_val:.3f}(系数={ent_coeff:.3f}) | "
                f"最优={self.best_score:.3f} 停滞={self._stagnation_steps} "
                f"精英池={replay_metrics['elite_pool_size']} "
                f"孵化池={replay_metrics['incubation_pool_size']} "
                f"搜索池={search_metrics['search_archive_size']} 重启={self._restart_count}"
            )
            tqdm.write(
                f"   分布: 初始熵={dst['entropy']:.3f} KL均匀={dst['kl_uniform']:.3f} "
                f"KL上步={dst['kl_prev']:.4f} 最高概率={dst['top1_prob']:.3f} "
                f"前五概率={dst['top5_prob']:.3f} 有效词汇={dst['eff_vocab']:.2f} "
                f"标准差={dst['prob_std']:.4f} | "
                f"本批: 唯一符号={uniq_tokens}/{FORMULA_VOCAB.size} "
                f"唯一公式={uniq_fmls}/{n_new} 多样性={fml_div:.2f}"
            )
            tqdm.write(
                f"[Timing @{step+1}] total={timing_total_ms:.0f}ms | "
                f"AB(sample+elite)={timing_ab_ms:.0f}ms "
                f"C(eval)={timing_eval_ms:.0f}ms "
                f"D(grad)={timing_grad_ms:.0f}ms "
                f"G(rest)={timing_rest_ms:.0f}ms | "
                f"mode={'batch' if ModelConfig.GPU_BATCH_EVAL else 'legacy'} "
                f"device={ModelConfig.DEVICE} "
                f"engine={eval_engine_name} "
                f"guard={'pass' if guard_passed else ('fail' if guard_passed is False else 'off')}:{guard_checked}"
            )
            if step % 10 == 0:
                tqdm.write(
                    f"[TimingDetail @{step+1}] "
                    f"AB: replay={timing_replay_plan_ms:.0f}ms "
                    f"search={timing_search_plugin_ms:.0f}ms "
                    f"forward={timing_ab_forward_ms:.0f}ms "
                    f"policy={timing_policy_sample_ms:.0f}ms "
                    f"memory={timing_memory_logprob_ms:.0f}ms | "
                    f"D: loss={timing_loss_build_ms:.0f}ms "
                    f"backward={timing_backward_ms:.0f}ms "
                    f"optim={timing_optimizer_ms:.0f}ms | "
                    f"dist={timing_dist_stats_ms:.0f}ms"
                )
            pbar.set_postfix({
                '验证': f"{avg_val:.3f}", '最优': f"{self.best_score:.3f}",
                '熵':   f"{ent_val:.2f}", 'IC':   f"{bim:.4f}",
                '停滞': f"{self._stagnation_steps}",
                '初始熵':  f"{dst['entropy']:.2f}",
                'KL上步': f"{dst['kl_prev']:.3f}",
            })

            if self.use_lord and step % 10 == 0:
                sr = self.rank_monitor.compute()
                self.training_history['stable_rank'].append(sr)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history.setdefault('batch_best_val_score', []).append(
                step_max_val if step_max_val != -float('inf') else None
            )
            self.training_history.setdefault('new_candidate_best_val_score', []).append(
                new_step_max_val if new_step_max_val != -float('inf') else None
            )
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('ic_stability', []).append(bis_)
            self.training_history.setdefault('sortino', []).append(bsor_)
            self.training_history.setdefault('elite_pool_size', []).append(
                replay_metrics["elite_pool_size"])
            self.training_history.setdefault('elite_archive_cells', []).append(
                replay_metrics["elite_archive_cells"])
            self.training_history.setdefault('elite_replay_used', []).append(n_elite)
            self.training_history.setdefault('incubation_replay_used', []).append(n_incubate)
            self.training_history.setdefault('policy_generated_used', []).append(n_policy)
            self.training_history.setdefault('search_plugin_used', []).append(n_plugin)
            self.training_history.setdefault('search_plugin_modules', []).append(
                ",".join(search_metrics.get("search_plugins") or []))
            self.training_history.setdefault('search_archive_size', []).append(
                search_metrics["search_archive_size"])
            self.training_history.setdefault('search_archive_cells', []).append(
                search_metrics["search_archive_cells"])
            self.training_history.setdefault('anneal_accept_rate', []).append(
                search_metrics["anneal_accept_rate"])
            self.training_history.setdefault('genetic_planned', []).append(
                search_metrics.get("genetic_planned", 0))
            self.training_history.setdefault('genetic_produced', []).append(
                search_metrics.get("genetic_produced", 0))
            self.training_history.setdefault('genetic_parent_count', []).append(
                search_metrics.get("genetic_parent_count", 0))
            self.training_history.setdefault('genetic_parent_niches', []).append(
                search_metrics.get("genetic_parent_niches", 0))
            self.training_history.setdefault('genetic_parent_source', []).append(
                search_metrics.get("genetic_parent_source", "none"))
            self.training_history.setdefault('incubation_pool_size', []).append(
                replay_metrics["incubation_pool_size"])
            self.training_history.setdefault('incubation_archive_cells', []).append(
                replay_metrics["incubation_archive_cells"])
            self.training_history.setdefault('elite_replay_frac_effective', []).append(elite_frac_eff)
            self.training_history.setdefault('init_entropy', []).append(dst['entropy'])
            self.training_history.setdefault('kl_uniform', []).append(dst['kl_uniform'])
            self.training_history.setdefault('kl_prev', []).append(dst['kl_prev'])
            self.training_history.setdefault('top1_prob', []).append(dst['top1_prob'])
            self.training_history.setdefault('eff_vocab', []).append(dst['eff_vocab'])
            self.training_history.setdefault('batch_uniq_tokens', []).append(uniq_tokens)
            self.training_history.setdefault('batch_uniq_fmls', []).append(uniq_fmls)
            self.training_history.setdefault('batch_fml_div', []).append(fml_div)
            self.training_history.setdefault('timing_total_ms', []).append(timing_total_ms)
            self.training_history.setdefault('timing_sample_elite_ms', []).append(timing_ab_ms)
            self.training_history.setdefault('timing_eval_ms', []).append(timing_eval_ms)
            self.training_history.setdefault('timing_grad_ms', []).append(timing_grad_ms)
            self.training_history.setdefault('timing_rest_ms', []).append(timing_rest_ms)
            self.training_history.setdefault('timing_replay_plan_ms', []).append(timing_replay_plan_ms)
            self.training_history.setdefault('timing_search_plugin_ms', []).append(timing_search_plugin_ms)
            self.training_history.setdefault('timing_ab_forward_ms', []).append(timing_ab_forward_ms)
            self.training_history.setdefault('timing_policy_sample_ms', []).append(timing_policy_sample_ms)
            self.training_history.setdefault('timing_memory_logprob_ms', []).append(timing_memory_logprob_ms)
            self.training_history.setdefault('timing_loss_build_ms', []).append(timing_loss_build_ms)
            self.training_history.setdefault('timing_backward_ms', []).append(timing_backward_ms)
            self.training_history.setdefault('timing_optimizer_ms', []).append(timing_optimizer_ms)
            self.training_history.setdefault('timing_dist_stats_ms', []).append(timing_dist_stats_ms)
            self.training_history.setdefault('eval_engine', []).append(eval_engine_name)
            self.training_history.setdefault('eval_guard_passed', []).append(guard_passed)
            self.training_history.setdefault('eval_guard_checked', []).append(guard_checked)
            self.training_history.setdefault('eval_guard_reward_diff', []).append(guard_reward_diff)
            self.training_history.setdefault('eval_guard_val_diff', []).append(guard_val_diff)
            self.training_history.setdefault('eval_guard_factor_diff', []).append(guard_factor_diff)

            self._save_training_history_live()

            if (step + 1) % 20 == 0 or (step + 1) == end_step:
                ckpt = self.save_checkpoint(step + 1)
                tqdm.write(f"[检查点] → {ckpt} (最优={self.best_score:.3f})")

            stop_request = read_checkpoint_stop_request(
                symbol=self.target_symbol,
                timeframe=getattr(self, "timeframe", None),
                algorithm_mode="rl",
            )
            if stop_request:
                ckpt = self.save_checkpoint(step + 1)
                acknowledge_checkpoint_stop(
                    symbol=self.target_symbol,
                    timeframe=getattr(self, "timeframe", None),
                    algorithm_mode="rl",
                    step=step + 1,
                    checkpoint_path=ckpt,
                )
                tqdm.write(f"[优雅切换] 已保存 checkpoint → {ckpt}; step={step + 1}")
                return

            # ── Part F: Migration hook（多岛训练时交换精英）────────────
            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                tqdm.write(f"[迁移钩子 @ 第{step+1}步] 调用已注册钩子")
                migration_hook(self, step + 1)

            # ── Part G: Entropy collapse detection & restart ─────────
            if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
                low_entropy_streak += 1
            else:
                low_entropy_streak  = 0

            if low_entropy_streak >= ModelConfig.ENTROPY_COLLAPSE_STEPS:
                # ── 自适应噪声：根据 stagnation 调整 ─────────────────────
                self._stagnation_steps = step - self._best_update_step
                stagnation_ratio = self._stagnation_steps / max(1, ModelConfig.STAGNATION_WINDOW)
                base_noise = ModelConfig.RESTART_NOISE
                if ModelConfig.ADAPTIVE_NOISE:
                    raw_noise = base_noise + ModelConfig.NOISE_BOOST_FACTOR * 0.1 * min(stagnation_ratio, 3.0)
                    noise = max(ModelConfig.NOISE_MIN, min(ModelConfig.NOISE_MAX, raw_noise))
                else:
                    noise = base_noise

                max_r = ModelConfig.MAX_RESTARTS
                if self._restart_count < max_r:
                    self._restart_count  += 1
                    self._last_restart_step = step
                    low_entropy_streak    = 0

                    # Fix 2: 每 N 次重启做一次完全随机初始化，逃离 best_snapshot 吸引子
                    # 深度坍塌 (H < 0.3) 时强制 full reset，不给 best_snapshot 恢复的机会
                    do_full_reset = (
                        self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                        or ent_val < 0.3
                    )

                    if do_full_reset:
                        # 完全重新初始化模型参数
                        for layer in self.model.modules():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=完全重置（脱离最优快照吸引子） "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f}"
                        )
                    elif self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                        with torch.no_grad():
                            if ModelConfig.PARTIAL_RESET:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if any(k in nm for k in ModelConfig.PARTIAL_RESET_LAYERS):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                            else:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if 'ffn' in nm or 'attention' in nm or nm.startswith('blocks'):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式={'部分层' if ModelConfig.PARTIAL_RESET else 'FFN/注意力'} "
                            f"噪声={noise:.4f}(基准={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f} "
                            f"扰动层数={len(perturbed_layers)}"
                        )
                    else:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                p.add_(torch.randn_like(p) * noise)
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=全参数 "
                            f"噪声={noise:.4f}(基准={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f} | 无最优快照"
                        )
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                else:
                    # 训练时间不敏感：超过重启上限后不再 Early Stop 终止，
                    # 改为「全参数强扰动 + 重置流计数」继续探索，直到跑满 TRAIN_STEPS。
                    # 从 best_snapshot 恢复（若有）以保住已发现的最优结构，再加大扰动。
                    low_entropy_streak = 0
                    self._last_restart_step = step
                    hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * hard_noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(
                        f"[强重启 @ 第{step}步] 已达最大重启次数={max_r} "
                        f"熵={ent_val:.3f} 强噪声={hard_noise:.4f} "
                        f"继续训练，不提前停止"
                    )

        # ── End of training ──────────────────────────────────────────
        # 仅当跑满最终步时才保存最终 strategy 和历史
        if end_step == ModelConfig.TRAIN_STEPS:
            if self.best_formula is not None:
                from .vocab import VOCAB_VERSION
                strategy_data = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": self.target_symbol,
                    "formula": self.best_formula,
                    "best_score": self.best_score,
                    "algorithm_mode": str(getattr(self, "algorithm_mode", "rl") or "rl").strip().lower(),
                    "strategy_source": "champion",
                }
                save_path = _strategy_file_for_symbol(
                    self.target_symbol,
                    getattr(self, "timeframe", None),
                    getattr(self, "algorithm_mode", "rl"),
                )
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                # P1-3: 原子写入
                tmp_path = f"{save_path}.{os.getpid()}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as fp:
                    json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
                os.replace(tmp_path, save_path)

            sym_tag = f"[{self.target_symbol}] " if self.target_symbol else ""
            self.training_history.pop('_low_entropy_streak', None)
            hist_path = self._training_history_path()
            # P1-3: 原子写入
            tmp_hist = hist_path + ".tmp"
            with open(tmp_hist, "w", encoding="utf-8") as fp:
                json.dump(self.training_history, fp)
            os.replace(tmp_hist, hist_path)

            print(f"\n[完成] {sym_tag}训练结束！")
            print(f"  最优验证分数 : {self.best_score:.4f}")
            print(f"  最优公式令牌 : {self.best_formula}")
            print(f"  可读公式     : {self._decode_formula(self.best_formula)}")
            print(f"  精英池大小   : {len(self._elite_pool)}")
            print(f"  精英衰减     : 启用={ModelConfig.ELITE_DECAY}，半衰期={ModelConfig.ELITE_DECAY_HALF_LIFE}")
            print(f"  自适应噪声   : 启用={ModelConfig.ADAPTIVE_NOISE}，范围=[{ModelConfig.NOISE_MIN}, {ModelConfig.NOISE_MAX}]")
            print(f"  部分层重置   : 启用={ModelConfig.PARTIAL_RESET}，层={ModelConfig.PARTIAL_RESET_LAYERS}")
            print(f"  重启次数     : {self._restart_count}")
            print(f"  策略已保存   : {save_path}")


    # ── 实时保存最优公式（防进程意外退出丢失）────────────────────────────────
    def _save_training_history_live(self) -> None:
        """周期性写入训练曲线 JSON，供 Web UI 实时展示。

        P1-3 修复：原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断写入
        导致 history 文件损坏。异常打印告警而非静默吞掉。
        """
        if not self.target_symbol:
            return
        try:
            hist_path = self._training_history_path()
            payload = {
                k: v for k, v in self.training_history.items()
                if k != "_low_entropy_streak"
            }
            # 原子写入：先写 tmp，再 os.replace 覆盖（POSIX/Windows 均原子）
            tmp_path = f"{hist_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp)
            os.replace(tmp_path, hist_path)
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会掩盖磁盘满/权限错误，至少打印告警
            try:
                tqdm.write(f"[警告] 训练历史保存失败: {exc}")
            except Exception:
                pass

    def _training_history_path(self) -> str:
        suffix = _artifact_suffix(self.target_symbol, getattr(self, "timeframe", None))
        mode = str(getattr(self, "algorithm_mode", "rl") or "rl").strip().lower()
        if not suffix:
            return "training_history.json"
        if mode == "ga":
            return f"training_history_ga_{suffix}.json"
        if mode == "hybrid":
            return f"training_history_hybrid_{suffix}.json"
        return f"training_history_{suffix}.json"

    def _save_strategy_live(self) -> None:
        """每次 best_formula 更新时立即保存 strategy json。
        即使训练中途进程被杀（OOM/终端回收/Ctrl+C），也能保留最新最优公式。

        P1-3 修复：原子写入（tmp + os.replace），避免写入中途被打断导致
        strategy JSON 截断损坏——既丢新最优也丢旧最优。异常打印告警。
        """
        if self.best_formula is None:
            return
        try:
            from .vocab import VOCAB_VERSION
            save_path = _strategy_file_for_symbol(
                self.target_symbol,
                getattr(self, "timeframe", None),
                getattr(self, "algorithm_mode", "rl"),
            )
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            existing: dict = {}
            p = pathlib.Path(save_path)
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        existing = raw
                except Exception:
                    existing = {}

            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "symbol": self.target_symbol,
                "formula": self.best_formula,
                "best_score": self.best_score,
                "formula_decoded": self._decode_formula(self.best_formula),
                "algorithm_mode": str(getattr(self, "algorithm_mode", "rl") or "rl").strip().lower(),
                "strategy_source": "champion",
            }
            # 保留训练数据路径等元数据，避免 live 保存把 data_file 冲掉
            for key in (
                "timeframe",
                "data_file",
                "mode",
                "train_steps",
                "train_sample",
                "train_ratio",
                "total_bars",
                "train_end_bar",
                "oos_start_bar",
            ):
                val = getattr(self, key, None)
                if val is None:
                    val = existing.get(key)
                if val is not None:
                    strategy_data[key] = val
            if not strategy_data.get("data_file") and self.target_symbol:
                data_file, tf = _fallback_data_file_for_symbol(self.target_symbol)
                if data_file:
                    strategy_data["data_file"] = data_file
                if tf and not strategy_data.get("timeframe"):
                    strategy_data["timeframe"] = tf
                if data_file and not strategy_data.get("mode"):
                    strategy_data["mode"] = "parquet_file"

            existing_score = existing.get("best_score")
            same_symbol = (
                not existing.get("symbol")
                or not self.target_symbol
                or str(existing.get("symbol")) == str(self.target_symbol)
            )
            same_timeframe = (
                not existing.get("timeframe")
                or not strategy_data.get("timeframe")
                or str(existing.get("timeframe")).upper() == str(strategy_data.get("timeframe")).upper()
            )
            if (
                existing_score is not None
                and same_symbol
                and same_timeframe
                and float(existing_score) > float(self.best_score)
            ):
                merged = dict(existing)
                for key in (
                    "timeframe",
                    "data_file",
                    "mode",
                    "train_steps",
                    "train_sample",
                    "train_ratio",
                    "total_bars",
                    "train_end_bar",
                    "oos_start_bar",
                ):
                    if strategy_data.get(key) is not None and not merged.get(key):
                        merged[key] = strategy_data[key]
                if merged != existing:
                    tmp_path = f"{save_path}.{os.getpid()}.tmp"
                    with open(tmp_path, "w", encoding="utf-8") as fp:
                        json.dump(merged, fp, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, save_path)
                return

            # 原子写入：先写 tmp，再 os.replace 覆盖
            tmp_path = f"{save_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
            os.replace(tmp_path, save_path)
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会让用户误以为策略已保存，实则没有
            try:
                tqdm.write(f"[警告] 策略保存失败: {exc}")
            except Exception:
                pass

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            suffix = _artifact_suffix(self.target_symbol, getattr(self, "timeframe", None))
            sym_tag = f"_{suffix}" if suffix else ""
            path = str(_CHECKPOINT_DIR / f"ckpt{sym_tag}_step_{step:04d}.pt")
        ckpt = {
            "step":                 step,
            "vocab_version":        VOCAB_VERSION,   # task 12.2: 版本校验所需
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "replay_policy":        self.replay_policy.state_dict(),
            "search_plugins":       self.search_plugins.state_dict(),
            # Legacy fields kept for older inspection scripts.
            "elite_pool":           self.replay_policy.state_dict().get("elite_pool", []),
            "elite_counter":        self.replay_policy.state_dict().get("elite_counter", 0),
            "incubation_pool":      self.replay_policy.state_dict().get("incubation_pool", []),
            "incubation_counter":   self.replay_policy.state_dict().get("incubation_counter", 0),
            "restart_count":        self._restart_count,
            "last_restart_step":    self._last_restart_step,
            "training_history":     {
                k: v for k, v in self.training_history.items()
                if k != '_low_entropy_streak'
            },
        }
        # P1-3: 原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断导致
        # checkpoint 文件截断损坏——既丢新最优也丢旧最优
        tmp_path = f"{path}.{os.getpid()}.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)

        # ── Task 12.2：版本校验（R3.7）──────────────────────────────────────
        # 从 checkpoint 读取 vocab_version；若字段缺失（旧版 checkpoint），视为
        # 版本不匹配并抛错——拒绝加载、不消费任何 token。
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError(
                f"checkpoint '{path}' 不含 vocab_version 字段（旧版产物），"
                f"当前词表版本 {FORMULA_VOCAB.version!r}；需重新训练后加载"
            )
        # verify() 版本不匹配时抛 VocabVersionMismatchError，拒绝加载
        FORMULA_VOCAB.verify(artifact_version)
        # ── 版本校验通过，继续加载 ────────────────────────────────────────

        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_score          = ckpt.get("best_score",  -float('inf'))
        self.best_formula        = ckpt.get("best_formula", None)
        self._best_snapshot      = ckpt.get("best_snapshot", None)
        self.factor_pool         = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        replay_state = ckpt.get("replay_policy") or {
            "elite_pool": ckpt.get("elite_pool", []),
            "elite_counter": ckpt.get("elite_counter", 0),
            "incubation_pool": ckpt.get("incubation_pool", []),
            "incubation_counter": ckpt.get("incubation_counter", 0),
        }
        self.replay_policy.load_state_dict(replay_state)
        self.search_plugins.load_state_dict(ckpt.get("search_plugins") or {})
        self._elite_pool = replay_state.get("elite_pool", [])
        self._elite_counter = replay_state.get("elite_counter", 0)
        self._incubation_pool = replay_state.get("incubation_pool", [])
        self._incubation_counter = replay_state.get("incubation_counter", 0)
        self._restart_count      = ckpt.get("restart_count", 0)
        self._last_restart_step  = ckpt.get("last_restart_step", -10**9)
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v

        completed = ckpt.get("step", 0)
        replay_metrics = self.replay_policy.metrics()
        tqdm.write(f"[检查点] 已从 {path} 恢复。"
                   f" 当前步={completed}  最优={self.best_score:.4f}"
                   f"  精英池={replay_metrics['elite_pool_size']} "
                   f"孵化池={replay_metrics['incubation_pool_size']}")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "无"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)
