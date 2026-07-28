"""3D batch operator implementations for formula execution.

Shape contract:
    input/output tensors are [B, N, T]
    B = candidate formula batch
    N = symbols
    T = time
"""
from __future__ import annotations

import torch

from .ops import OPS_CONFIG


def _delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return x
    return torch.cat([torch.zeros_like(x[:, :, :d]), x[:, :, :-d]], dim=2)


def _rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    b, n, t = x.shape
    pad = torch.zeros(b, n, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=2).unfold(2, d, 1)


def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    return _rolling(x, d).mean(dim=-1)


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    w = _rolling(x, d)
    m = w.mean(dim=-1, keepdim=True)
    return torch.nan_to_num(((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6, nan=0.0)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    w = _rolling(x, d)
    cur = w[:, :, :, -1:]
    return torch.nan_to_num((w < cur).float().mean(dim=-1), nan=0.0)


def _ts_corr_10(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = 10
    wx = _rolling(x, d)
    wy = _rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    sx = ((wx - mx) ** 2).mean(dim=-1).sqrt()
    sy = ((wy - my) ** 2).mean(dim=-1).sqrt()
    return torch.nan_to_num(cov / (sx * sy + 1e-6), nan=0.0).clamp(-1.0, 1.0)


def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


def _op_jump(x: torch.Tensor) -> torch.Tensor:
    b, n, t = x.shape
    cnt = torch.arange(1, t + 1, device=x.device, dtype=x.dtype).view(1, 1, t)
    mean = x.cumsum(dim=2) / cnt
    var = ((x * x).cumsum(dim=2) / cnt) - mean * mean
    std = var.clamp(min=1e-12).sqrt() + 1e-6
    out = torch.tanh((x - mean) / std - 1.5)
    min_warmup = 5
    if t > min_warmup:
        warmup_mask = torch.arange(t, device=x.device) < min_warmup
        out[:, :, warmup_mask] = 0.0
    return out


def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return (x + 0.8 * _delay(x, 1) + 0.6 * _delay(x, 2)) / 2.4


def _op_wma(x: torch.Tensor) -> torch.Tensor:
    return (3.0 * x + 2.0 * _delay(x, 1) + _delay(x, 2)) / 6.0


def _ema_simple(x: torch.Tensor, span: int) -> torch.Tensor:
    alpha = 2.0 / (span + 1.0)
    if x.shape[2] == 0 or alpha >= 1.0:
        return x.clone()
    out = torch.zeros_like(x)
    out[:, :, 0] = x[:, :, 0]
    for t in range(1, x.shape[2]):
        out[:, :, t] = alpha * x[:, :, t] + (1 - alpha) * out[:, :, t - 1]
    return torch.nan_to_num(out, nan=0.0)


def _ts_quantile(x: torch.Tensor, d: int) -> torch.Tensor:
    w = _rolling(x, d)
    cur = w[:, :, :, -1:]
    return torch.nan_to_num((w < cur).float().mean(dim=-1), nan=0.0)


def _ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    w = _rolling(x, d)
    m = w.mean(dim=-1, keepdim=True)
    std = ((w - m) ** 2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
    return torch.nan_to_num(((w - m) / std).pow(3).mean(dim=-1), nan=0.0).clamp(-5.0, 5.0)


def _delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    return x - _delay(x, d)


def _ts_arg_max(x: torch.Tensor, d: int) -> torch.Tensor:
    return _rolling(x, d).argmax(dim=-1).float() / (d - 1)


def _ts_arg_min(x: torch.Tensor, d: int) -> torch.Tensor:
    return _rolling(x, d).argmin(dim=-1).float() / (d - 1)


def _decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    weights = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (_rolling(x, d) * weights).sum(dim=-1)


def _decay_exp(x: torch.Tensor, d: int, alpha: float = 0.5) -> torch.Tensor:
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(d)], dtype=x.dtype, device=x.device)
    weights = torch.flip(weights, dims=[0])
    weights = weights / weights.sum()
    return (_rolling(x, d) * weights).sum(dim=-1)


def _scale(x: torch.Tensor) -> torch.Tensor:
    return x / (x.abs().cumsum(dim=2) + 1e-6)


def _ts_covariance(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    wx = _rolling(x, d)
    wy = _rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    return torch.nan_to_num(((wx - mx) * (wy - my)).mean(dim=-1), nan=0.0)


def _ts_product(x: torch.Tensor, d: int) -> torch.Tensor:
    x_safe = torch.clamp(x, -0.999, None)
    log_sum = _rolling(torch.log1p(x_safe), d).sum(dim=-1).clamp(-10.0, 10.0)
    return torch.nan_to_num(torch.expm1(log_sum), nan=0.0, posinf=0.0, neginf=0.0)


def _signed_power(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    return torch.nan_to_num((torch.sign(x) * torch.abs(x) ** a).clamp(-1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    b, n, t = x.shape
    if n == 1:
        return torch.nan_to_num(x, nan=0.5, posinf=0.5, neginf=0.5)
    order = x.argsort(dim=1)
    ranks = torch.empty_like(x)
    rank_vals = torch.arange(n, device=x.device, dtype=x.dtype).view(1, n, 1).expand(b, n, t)
    ranks.scatter_(1, order, rank_vals)
    return torch.nan_to_num(ranks / (n - 1), nan=0.5, posinf=0.5, neginf=0.5)


def _cs_scale(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return torch.nan_to_num(x, nan=0.5, posinf=0.5, neginf=0.5)
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    span = mx - mn
    zero_span = span.abs() < 1e-9
    out = (x - mn) / torch.where(zero_span, torch.ones_like(span), span)
    out = torch.where(zero_span.expand_as(out), torch.full_like(out, 0.5), out)
    return torch.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_neutralize(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(x - x.mean(dim=1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0)


def _ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    return _rolling(x, d).sum(dim=-1)


def _power_signed(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    return _signed_power(x, a)


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(torch.sign(x) * torch.log1p(torch.abs(x)), nan=0.0, posinf=0.0, neginf=0.0)


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(torch.sign(x) * torch.sqrt(torch.abs(x)), nan=0.0, posinf=0.0, neginf=0.0)


def _ts_zscore(x: torch.Tensor, w: int) -> torch.Tensor:
    windows = _rolling(x, w)
    m = windows.mean(dim=-1)
    std = windows.std(dim=-1, unbiased=False)
    z = (x - m) / (std + 1e-6)
    return torch.nan_to_num(torch.where(std < 1e-6, torch.zeros_like(z), z), nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(x: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    windows = _rolling(x, 20)
    lower = torch.quantile(windows.float(), lo, dim=-1).to(x.dtype)
    upper = torch.quantile(windows.float(), hi, dim=-1).to(x.dtype)
    span = upper - lower
    safe_lower = torch.where(span < 1e-9, x, lower)
    safe_upper = torch.where(span < 1e-9, x, upper)
    return torch.nan_to_num(torch.clamp(x, safe_lower, safe_upper), nan=0.0, posinf=0.0, neginf=0.0)


def _sigmoid_squash(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.sigmoid(x) - 1


def _tanh_squash(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _if_gt(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(torch.where(x > 0, y, z), nan=0.0, posinf=0.0, neginf=0.0)


BATCH_OPS_CONFIG = [
    ("ADD", lambda x, y: x + y, 2),
    ("SUB", lambda x, y: x - y, 2),
    ("MUL", lambda x, y: x * y, 2),
    ("DIV", lambda x, y: x / (y + 1e-6), 2),
    ("NEG", lambda x: -x, 1),
    ("ABS", torch.abs, 1),
    ("SIGN", torch.sign, 1),
    ("GATE", _op_gate, 3),
    ("JUMP", _op_jump, 1),
    ("DECAY", _op_decay, 1),
    ("DELAY1", lambda x: _delay(x, 1), 1),
    ("MAX3", lambda x: torch.max(x, torch.max(_delay(x, 1), _delay(x, 2))), 1),
    ("TS_MEAN_5", lambda x: _ts_mean(x, 5), 1),
    ("TS_MEAN_10", lambda x: _ts_mean(x, 10), 1),
    ("TS_MEAN_20", lambda x: _ts_mean(x, 20), 1),
    ("TS_STD_5", lambda x: _ts_std(x, 5), 1),
    ("TS_STD_10", lambda x: _ts_std(x, 10), 1),
    ("TS_STD_20", lambda x: _ts_std(x, 20), 1),
    ("TS_RANK_5", lambda x: _ts_rank(x, 5), 1),
    ("TS_RANK_10", lambda x: _ts_rank(x, 10), 1),
    ("TS_RANK_20", lambda x: _ts_rank(x, 20), 1),
    ("TS_CORR_10", _ts_corr_10, 2),
    ("MOMENTUM_5", lambda x: _ts_mean(x, 5) - _ts_mean(x, 20), 1),
    ("MOMENTUM_10", lambda x: _ts_mean(x, 10) - _ts_mean(x, 20), 1),
    ("TS_MAX_10", lambda x: _rolling(x, 10).max(dim=-1).values, 1),
    ("TS_MIN_10", lambda x: _rolling(x, 10).min(dim=-1).values, 1),
    ("WMA", _op_wma, 1),
    ("DELAY4", lambda x: _delay(x, 4), 1),
    ("EMA_5", lambda x: _ema_simple(x, 5), 1),
    ("EMA_20", lambda x: _ema_simple(x, 20), 1),
    ("TS_QUANTILE_10", lambda x: _ts_quantile(x, 10), 1),
    ("TS_SKEW_10", lambda x: _ts_skew(x, 10), 1),
    ("TS_MIN_20", lambda x: _rolling(x, 20).min(dim=-1).values, 1),
    ("TS_MAX_20", lambda x: _rolling(x, 20).max(dim=-1).values, 1),
    ("DELTA", lambda x: _delta(x, 1), 1),
    ("TS_ARG_MAX_5", lambda x: _ts_arg_max(x, 5), 1),
    ("TS_ARG_MIN_5", lambda x: _ts_arg_min(x, 5), 1),
    ("DECAY_LINEAR_5", lambda x: _decay_linear(x, 5), 1),
    ("SCALE", _scale, 1),
    ("COVARIANCE_10", lambda x, y: _ts_covariance(x, y, 10), 2),
    ("PRODUCT_5", lambda x: _ts_product(x, 5), 1),
    ("SIGNED_POWER_2", lambda x: _signed_power(x, 2.0), 1),
    ("TS_DECAY_EXP_5", lambda x: _decay_exp(x, 5, 0.5), 1),
    ("DELTA_5", lambda x: _delta(x, 5), 1),
    ("CS_RANK", _cs_rank, 1),
    ("CS_SCALE", _cs_scale, 1),
    ("CS_NEUTRALIZE", _cs_neutralize, 1),
    ("TS_SUM_5", lambda x: torch.nan_to_num(_ts_sum(x, 5), nan=0.0), 1),
    ("TS_SUM_10", lambda x: torch.nan_to_num(_ts_sum(x, 10), nan=0.0), 1),
    ("TS_SUM_20", lambda x: torch.nan_to_num(_ts_sum(x, 20), nan=0.0), 1),
    ("MIN", lambda x, y: torch.nan_to_num(torch.minimum(x, y), nan=0.0), 2),
    ("MAX", lambda x, y: torch.nan_to_num(torch.maximum(x, y), nan=0.0), 2),
    ("POWER", lambda x: _power_signed(x, 2.0), 1),
    ("SIGNED_LOG", _signed_log, 1),
    ("SQRT", _signed_sqrt, 1),
    ("TS_ZSCORE_10", lambda x: _ts_zscore(x, 10), 1),
    ("TS_ZSCORE_20", lambda x: _ts_zscore(x, 20), 1),
    ("WINSORIZE", _winsorize, 1),
    ("CLIP", lambda x: torch.clamp(x, -3.0, 3.0), 1),
    ("SIGMOID", _sigmoid_squash, 1),
    ("TANH_SQUASH", _tanh_squash, 1),
    ("IF_GT", _if_gt, 3),
]


if [x[0] for x in BATCH_OPS_CONFIG] != [x[0] for x in OPS_CONFIG]:
    raise RuntimeError("BATCH_OPS_CONFIG must match OPS_CONFIG order and names")
