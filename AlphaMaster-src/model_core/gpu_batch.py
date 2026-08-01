"""
Experimental GPU-oriented batch evaluators.

This module is intentionally not wired into the live trainer yet.  It provides a
measurable bridge from the current per-formula CPU path to a true GPU batch path:
many candidate factors enter as one tensor, and rewards come back as one vector.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from torch import Tensor

from strategy_manager.signal import compute_target_positions_stateless
from .batch_ops import BATCH_OPS_CONFIG
from .config import ModelConfig
from .vocab import FORMULA_VOCAB


@dataclass(frozen=True)
class BatchEvalResult:
    train_scores: Tensor
    val_scores: Tensor
    oos_sortino: Tensor


@dataclass(frozen=True)
class BatchPipelineResult:
    factors: Tensor
    valid: Tensor
    train_scores: Tensor
    val_scores: Tensor
    oos_sortino: Tensor


class BatchStackVM3D:
    """Shape-correct batch StackVM for both single-symbol and multi-symbol data."""

    def __init__(self):
        self.feat_offset = FORMULA_VOCAB.operator_offset
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(BATCH_OPS_CONFIG)}
        self.op_name_map = {i + self.feat_offset: cfg[0] for i, cfg in enumerate(BATCH_OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(BATCH_OPS_CONFIG)}
        self.native_ops = None

    def _native(self, device: torch.device):
        enabled = os.getenv("ALPHAMASTER_NATIVE_FORMULA_OPS", "0").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled or device.type != "cuda":
            return None
        if self.native_ops is None:
            from .native_backend import NativeElementwiseOps

            self.native_ops = NativeElementwiseOps(verbose=False)
        return self.native_ops

    def _apply_op(self, op_name: str, op_func, args: list[Tensor], device: torch.device) -> Tensor:
        native = self._native(device)
        if native is not None and native.supports(op_name, len(args)):
            return native.apply(op_name, *args)
        return op_func(*args)

    @staticmethod
    def _normalize_output_batch(x: Tensor) -> Tensor:
        """Normalize [B,N,T] exactly like StackVM does per formula."""
        bsz, n_symbols, n_bars = x.shape
        flat = x.reshape(bsz, -1)
        global_std = flat.std(dim=1)
        const = global_std < 1e-6

        if n_symbols > 1:
            cs_mean = x.mean(dim=1, keepdim=True)
            cs_std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
            cs_z = torch.clamp((x - cs_mean) / cs_std, -3.0, 3.0)
            return torch.where(const[:, None, None], x, cs_z)

        single = x[:, 0, :]
        cnt = torch.arange(1, n_bars + 1, device=x.device, dtype=x.dtype).view(1, n_bars)
        ts_mean = single.cumsum(dim=1) / cnt
        ts_var = ((single * single).cumsum(dim=1) / cnt) - ts_mean * ts_mean
        ts_std = ts_var.clamp(min=1e-8).sqrt()
        ts_z = torch.clamp((single - ts_mean) / ts_std, -3.0, 3.0)
        out = torch.where(const[:, None], single, ts_z)
        return out[:, None, :]

    def execute_batch(self, formulas: Tensor | list[list[int]], feat_tensor: Tensor) -> tuple[Tensor, Tensor]:
        if not torch.is_tensor(formulas):
            formulas = torch.tensor(formulas, dtype=torch.long, device=feat_tensor.device)
        else:
            formulas = formulas.to(device=feat_tensor.device, dtype=torch.long)
        if formulas.ndim != 2:
            raise ValueError(f"formulas must be [B,L], got {tuple(formulas.shape)}")
        if feat_tensor.ndim != 3:
            raise ValueError(f"feat_tensor must be [N,F,T], got {tuple(feat_tensor.shape)}")

        bsz, max_len = formulas.shape
        n_symbols, n_features, n_bars = feat_tensor.shape
        stack = torch.zeros(
            bsz,
            max_len,
            n_symbols,
            n_bars,
            dtype=feat_tensor.dtype,
            device=feat_tensor.device,
        )
        ptr = torch.zeros(bsz, dtype=torch.long, device=feat_tensor.device)
        valid = torch.ones(bsz, dtype=torch.bool, device=feat_tensor.device)
        feat_bank = feat_tensor.permute(1, 0, 2)

        for step in range(max_len):
            tok = formulas[:, step]
            active = valid

            feat_mask = active & (tok < self.feat_offset)
            if feat_mask.any():
                idx = torch.nonzero(feat_mask, as_tuple=False).flatten()
                feat_ids = tok[idx]
                ok = feat_ids < n_features
                if ok.any():
                    idx_ok = idx[ok]
                    stack[idx_ok, ptr[idx_ok], :, :] = feat_bank[feat_ids[ok], :, :]
                    ptr[idx_ok] += 1
                if (~ok).any():
                    valid[idx[~ok]] = False

            op_active = active & (tok >= self.feat_offset)
            if op_active.any():
                op_tokens = torch.unique(tok[op_active])
                for op_tok_t in op_tokens:
                    op_tok = int(op_tok_t.item())
                    idx = torch.nonzero(op_active & (tok == op_tok), as_tuple=False).flatten()
                    if op_tok not in self.op_map:
                        valid[idx] = False
                        continue
                    arity = self.arity_map[op_tok]
                    enough = ptr[idx] >= arity
                    if (~enough).any():
                        valid[idx[~enough]] = False
                    idx = idx[enough]
                    if idx.numel() == 0:
                        continue

                    op_name = self.op_name_map[op_tok]
                    native = self._native(feat_tensor.device)
                    native_supported = native is not None and native.supports(op_name, arity)
                    base = ptr[idx] - arity
                    args = [stack[idx, base + off, :, :] for off in range(arity)]
                    try:
                        res = self._apply_op(
                            op_name,
                            self.op_map[op_tok],
                            args,
                            feat_tensor.device,
                        )
                    except Exception as exc:
                        if native_supported:
                            raise RuntimeError(f"native formula op failed: {op_name}/{arity}") from exc
                        valid[idx] = False
                        continue
                    if res.shape != (idx.numel(), n_symbols, n_bars):
                        valid[idx] = False
                        continue
                    res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack[idx, base, :, :] = res
                    ptr[idx] = base + 1

        valid = valid & (ptr == 1)
        factors = torch.zeros(bsz, n_symbols, n_bars, dtype=feat_tensor.dtype, device=feat_tensor.device)
        if valid.any():
            idx = torch.nonzero(valid, as_tuple=False).flatten()
            factors[idx, :, :] = stack[idx, 0, :, :]
            factors[idx] = self._normalize_output_batch(factors[idx])
        return factors, valid


class BatchBacktestEvaluator:
    """Vectorized scorer for factors shaped [B, N, T].

    The first target is the common single-symbol path used by the current A-share
    runs.  It keeps the same high-level reward ingredients as MT5Backtest, but
    returns one score per candidate formula instead of evaluating candidates one
    by one through Python.
    """

    def __init__(self, cost_rate: float = 0.0003, periods_per_year: int = 6240):
        self.cost_rate = cost_rate
        self.periods_per_year = periods_per_year

    @staticmethod
    def _positions_batch(factors: Tensor) -> Tensor:
        b, n, t = factors.shape
        flat = factors.reshape(b * n, t)
        pos = compute_target_positions_stateless(flat)
        return pos.reshape(b, n, t)

    @staticmethod
    def _sortino_batch(pnl: Tensor, periods_per_year: int, eps: float = 1e-8) -> Tensor:
        flat = pnl.reshape(pnl.shape[0], -1)
        mean_pnl = flat.mean(dim=1)
        downside_mask = flat < 0
        downside_count = downside_mask.sum(dim=1)
        downside = torch.where(downside_mask, flat, torch.zeros_like(flat))
        downside_mean = downside.sum(dim=1) / downside_count.clamp(min=1)
        centered = torch.where(downside_mask, flat - downside_mean[:, None], torch.zeros_like(flat))
        raw_std = torch.sqrt((centered.square().sum(dim=1) / downside_count.clamp(min=1)).clamp(min=0.0))
        raw_std = torch.where(downside_count > 0, raw_std, torch.zeros_like(raw_std))
        full_std = flat.std(dim=1, unbiased=False).clamp(min=eps)
        floor = (full_std * 0.2).clamp(min=eps)
        downside_std = torch.maximum(raw_std, floor)
        score = mean_pnl / downside_std * math.sqrt(periods_per_year)
        return torch.clamp(score, -20.0, 20.0)

    @staticmethod
    def _calmar_batch(pnl: Tensor, periods_per_year: int, eps: float = 1e-8) -> Tensor:
        flat = pnl.reshape(pnl.shape[0], -1)
        ann_ret = flat.mean(dim=1) * periods_per_year
        cum = torch.cumsum(flat, dim=1)
        peak = torch.cummax(cum, dim=1).values
        drawdown = (peak - cum).amax(dim=1).clamp(min=eps)
        return torch.clamp(ann_ret / drawdown, -10.0, 10.0)

    @staticmethod
    def _ts_ic_stability_batch(factors: Tensor, target_ret: Tensor) -> Tensor:
        if factors.shape[-1] < 10:
            return torch.zeros(factors.shape[0], dtype=factors.dtype, device=factors.device)
        x = factors[:, :, :-1]
        y = target_ret[None, :, 1:].expand_as(x)
        xm = x - x.mean(dim=2, keepdim=True)
        ym = y - y.mean(dim=2, keepdim=True)
        sx = xm.square().mean(dim=2).sqrt()
        sy = ym.square().mean(dim=2).sqrt()
        ic = (xm * ym).mean(dim=2) / (sx * sy + 1e-8)
        valid = (sx >= 1e-6) & (sy >= 1e-6)
        ic = torch.where(valid, ic, torch.zeros_like(ic))
        denom = valid.sum(dim=1).clamp(min=1)
        ic_mean = ic.sum(dim=1) / denom
        centered = torch.where(valid, ic - ic_mean[:, None], torch.zeros_like(ic))
        ic_std = torch.sqrt(centered.square().sum(dim=1) / denom)
        return torch.clamp(ic_mean / (ic_std + 1e-6), -3.0, 3.0)

    @staticmethod
    def _turnover_penalty_batch(turnover: Tensor) -> Tensor:
        mean_to = turnover.reshape(turnover.shape[0], -1).mean(dim=1)
        return -torch.clamp((mean_to - 0.2) * 3.0, min=0.0, max=3.0)

    @staticmethod
    def _turnover_quality_batch(position: Tensor) -> Tensor:
        """Replicate MT5Backtest._turnover_quality for each batch row."""
        bsz, n_symbols, n_bars = position.shape
        pos_i = position.to(torch.int32)
        nonzero = pos_i != 0
        prev = torch.roll(pos_i, 1, dims=2)
        prev[:, :, 0] = 0
        run_start = nonzero & (pos_i != prev)
        total_trades = run_start.sum(dim=(1, 2)).to(position.dtype)

        total_bars = n_symbols * n_bars
        target_trades = max(total_bars / 12.0, 1.0)
        actual_ratio = total_trades / target_trades
        freq_score = torch.empty_like(actual_ratio)
        freq_score = torch.where(actual_ratio <= 0, torch.full_like(freq_score, -2.0), freq_score)
        freq_score = torch.where(
            (actual_ratio > 0) & (actual_ratio < 0.05),
            -2.0 + actual_ratio / 0.05,
            freq_score,
        )
        freq_score = torch.where(
            (actual_ratio >= 0.05) & (actual_ratio < 0.5),
            -1.0 + (actual_ratio - 0.05) / 0.45,
            freq_score,
        )
        freq_score = torch.where(
            (actual_ratio >= 0.5) & (actual_ratio <= 2.0),
            torch.exp(-0.5 * (torch.log(actual_ratio) / math.log(2.0)).square()),
            freq_score,
        )
        freq_score = torch.where(
            (actual_ratio > 2.0) & (actual_ratio <= 8.0),
            0.5 - (actual_ratio - 2.0) / 6.0 * 1.5,
            freq_score,
        )
        freq_score = torch.where(actual_ratio > 8.0, torch.full_like(freq_score, -2.0), freq_score)

        # Sum run lengths by counting non-zero bars. This matches the scalar
        # average because runs partition exactly the non-zero position bars.
        nonzero_bars = nonzero.sum(dim=(1, 2)).to(position.dtype)
        avg_hold = nonzero_bars / total_trades.clamp(min=1)
        hold_bonus = torch.minimum(
            torch.full_like(avg_hold, 0.3),
            torch.log(torch.clamp(avg_hold, min=1.0)) / math.log(30.0) * 0.3,
        )
        hold_bonus = torch.where(total_trades > 0, hold_bonus, torch.zeros_like(hold_bonus))
        return freq_score + hold_bonus

    def _symbol_consistency_batch(self, pnl: Tensor, position: Tensor) -> Tensor:
        bsz, n_symbols, n_bars = pnl.shape
        if n_symbols == 0:
            return torch.zeros(bsz, dtype=pnl.dtype, device=pnl.device)
        per_sortino = self._sortino_by_symbol_batch(pnl, self.periods_per_year)
        pos_abs = position.abs()
        diff = (pos_abs[:, :, 1:] - pos_abs[:, :, :-1]).abs()
        trades = (diff > 0.1).sum(dim=2)
        min_trades = max(5, n_bars // 100)
        inactive = trades < min_trades
        inactive_ratio = inactive.float().mean(dim=1)
        any_bad = (per_sortino < -2.0).any(dim=1)
        active = ~inactive
        active_count = active.sum(dim=1)
        positive_active = ((per_sortino > 0) & active).sum(dim=1)
        ratio = positive_active.to(pnl.dtype) / active_count.clamp(min=1).to(pnl.dtype)
        score = torch.where(
            ratio < 0.6,
            (ratio - 0.6) / 0.6,
            (ratio - 0.6) / 0.4,
        )
        score = torch.where(ratio == 1.0, score + 0.5, score)
        score = torch.where(active_count == 0, torch.full_like(score, -3.0), score)
        score = torch.where(any_bad, torch.full_like(score, -2.0), score)
        score = torch.where(inactive_ratio > 0.4, torch.full_like(score, -3.0), score)
        return score

    def _cost_stress_batch(self, position: Tensor, target_ret: Tensor, stress_mult: float = 2.0) -> Tensor:
        prev_pos = torch.roll(position, 1, dims=2)
        prev_pos[:, :, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        stressed_pnl = position * target_ret.unsqueeze(0) - turnover * self.cost_rate * stress_mult
        return torch.clamp(self._sortino_batch(stressed_pnl, self.periods_per_year), -5.0, 5.0)

    @staticmethod
    def _sortino_by_symbol_batch(pnl: Tensor, periods_per_year: int = 6240, eps: float = 1e-8) -> Tensor:
        bsz, n_symbols, _ = pnl.shape
        flat = pnl
        mean_pnl = flat.mean(dim=2)
        downside_mask = flat < 0
        downside_count = downside_mask.sum(dim=2)
        downside = torch.where(downside_mask, flat, torch.zeros_like(flat))
        downside_mean = downside.sum(dim=2) / downside_count.clamp(min=1)
        centered = torch.where(downside_mask, flat - downside_mean[:, :, None], torch.zeros_like(flat))
        raw_std = torch.sqrt((centered.square().sum(dim=2) / downside_count.clamp(min=1)).clamp(min=0.0))
        raw_std = torch.where(downside_count > 0, raw_std, torch.zeros_like(raw_std))
        full_std = flat.std(dim=2, unbiased=False).clamp(min=eps)
        floor = (full_std * 0.2).clamp(min=eps)
        downside_std = torch.maximum(raw_std, floor)
        score = mean_pnl / downside_std * math.sqrt(periods_per_year)
        return torch.clamp(score, -20.0, 20.0)

    @staticmethod
    def _exposure_penalty_batch(position: Tensor) -> Tensor:
        exposure = position.abs().reshape(position.shape[0], -1).mean(dim=1)
        penalty = (exposure / 0.10 - 1.0) * 2.0
        return torch.where(exposure < 0.10, penalty, torch.zeros_like(penalty))

    @staticmethod
    def _beta_neutral_penalty_batch(position: Tensor) -> Tensor:
        flat = position.reshape(position.shape[0], -1)
        long_ratio = (flat > 0.05).float().mean(dim=1)
        short_ratio = (flat < -0.05).float().mean(dim=1)
        max_ratio = torch.maximum(long_ratio, short_ratio)
        heavy = -2.0 * ((max_ratio - 0.85) / 0.15)
        light = -0.5 * ((max_ratio - 0.70) / 0.15)
        return torch.where(
            max_ratio > 0.85,
            heavy,
            torch.where(max_ratio > 0.70, light, torch.zeros_like(max_ratio)),
        )

    def _half_consistency_bonus_batch(self, pnl: Tensor) -> Tensor:
        t = pnl.shape[2]
        if t < 20:
            return torch.zeros(pnl.shape[0], dtype=pnl.dtype, device=pnl.device)
        half = t // 2
        s1 = self._sortino_batch(pnl[:, :, :half], self.periods_per_year)
        s2 = self._sortino_batch(pnl[:, :, half:], self.periods_per_year)
        both_positive = (s1 > 0) & (s2 > 0)
        opposite = (s1 * s2) < 0
        return torch.where(
            both_positive,
            torch.full_like(s1, 0.5),
            torch.where(opposite, torch.full_like(s1, -1.0), torch.zeros_like(s1)),
        )

    def _multi_objective_batch(
        self,
        factors: Tensor,
        target_ret: Tensor,
        pnl: Tensor,
        position: Tensor,
    ) -> Tensor:
        ann_ret = pnl.reshape(pnl.shape[0], -1).mean(dim=1) * self.periods_per_year
        sortino = self._sortino_batch(pnl, self.periods_per_year)
        calmar = self._calmar_batch(pnl, self.periods_per_year)
        ts_ic = self._ts_ic_stability_batch(factors, target_ret)
        tq = self._turnover_quality_batch(position)
        exposure = self._exposure_penalty_batch(position)
        beta = self._beta_neutral_penalty_batch(position)
        consist = self._half_consistency_bonus_batch(pnl)

        if factors.shape[1] > 1:
            sym_cons = self._symbol_consistency_batch(pnl, position)
            cost_s = self._cost_stress_batch(position, target_ret)
            if ModelConfig.REWARD_MODE == "ftmo":
                return 0.75 * ann_ret + 0.05 * sortino + 0.10 * calmar + 0.02 * ts_ic + 0.03 * sym_cons + 0.02 * cost_s + 0.03 * tq + exposure + beta + consist
            return 0.60 * ann_ret + 0.10 * sortino + 0.05 * calmar + 0.10 * ts_ic + 0.05 * sym_cons + 0.05 * cost_s + 0.05 * tq + exposure + beta + consist
        if ModelConfig.REWARD_MODE == "ftmo":
            return 0.80 * ann_ret + 0.05 * sortino + 0.10 * calmar + 0.03 * ts_ic + 0.02 * tq + exposure + beta + consist
        return 0.60 * ann_ret + 0.15 * sortino + 0.10 * calmar + 0.10 * ts_ic + 0.05 * tq + exposure + beta + consist

    def evaluate_fold_batch(
        self,
        factors: Tensor,
        target_ret: Tensor,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> BatchEvalResult:
        if factors.ndim != 3:
            raise ValueError(f"factors must be [B,N,T], got {tuple(factors.shape)}")
        if target_ret.ndim != 2:
            raise ValueError(f"target_ret must be [N,T], got {tuple(target_ret.shape)}")

        position = self._positions_batch(factors)
        prev_pos = torch.roll(position, 1, dims=2)
        prev_pos[:, :, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl = position * target_ret.unsqueeze(0) - turnover * self.cost_rate

        train_score = self._multi_objective_batch(
            factors[:, :, train_start:train_end],
            target_ret[:, train_start:train_end],
            pnl[:, :, train_start:train_end],
            position[:, :, train_start:train_end],
        ) + self._turnover_penalty_batch(turnover[:, :, train_start:train_end])

        base_val = self._multi_objective_batch(
            factors[:, :, val_start:val_end],
            target_ret[:, val_start:val_end],
            pnl[:, :, val_start:val_end],
            position[:, :, val_start:val_end],
        )
        pnl_val = pnl[:, :, val_start:val_end]
        oos_sor = self._sortino_batch(pnl_val, self.periods_per_year)
        mult = torch.where(
            oos_sor <= 0,
            torch.maximum(torch.full_like(oos_sor, 0.1), 0.5 + oos_sor * 0.4),
            torch.minimum(torch.full_like(oos_sor, 1.2), 1.0 + oos_sor * 0.1),
        )
        return BatchEvalResult(train_score, base_val * mult, oos_sor)


class BatchFormulaPipeline:
    """Full single-symbol batch path: formulas -> factors -> scores."""

    def __init__(self, cost_rate: float = 0.0003, periods_per_year: int = 6240):
        self.vm = BatchStackVM3D()
        self.bt = BatchBacktestEvaluator(cost_rate=cost_rate, periods_per_year=periods_per_year)

    def evaluate_fold_batch(
        self,
        formulas: Tensor | list[list[int]],
        feat_tensor: Tensor,
        target_ret: Tensor,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> BatchPipelineResult:
        factors, valid = self.vm.execute_batch(formulas, feat_tensor)
        scores = self.bt.evaluate_fold_batch(
            factors,
            target_ret,
            train_start,
            train_end,
            val_start,
            val_end,
        )
        train_scores = torch.where(valid, scores.train_scores, torch.full_like(scores.train_scores, -5.0))
        val_scores = torch.where(valid, scores.val_scores, torch.full_like(scores.val_scores, -5.0))
        return BatchPipelineResult(
            factors=factors,
            valid=valid,
            train_scores=train_scores,
            val_scores=val_scores,
            oos_sortino=scores.oos_sortino,
        )
