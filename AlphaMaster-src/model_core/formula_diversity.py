"""Structural diversity helpers for formula replay and search archives."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .vocab import FORMULA_VOCAB


PRICE_FEATURE_HINTS = (
    "PRICE",
    "RET",
    "ROC",
    "MA",
    "VWAP",
    "BOLL",
    "KELTNER",
    "DONCHIAN",
    "ICHIMOKU",
    "SAR",
    "STOCH",
    "RSI",
    "DMI",
    "ADX",
    "SUPERTREND",
)
VOLUME_FEATURE_HINTS = ("VOL", "VOLUME", "OBV", "MFI", "AD_LINE", "VWAP")
VOLATILITY_FEATURE_HINTS = ("ATR", "STD", "VOLATILITY", "RANGE", "GK_VOL", "PARKINSON", "JUMP")
TREND_FEATURE_HINTS = ("SLOPE", "TREND", "MOMENTUM", "AROON", "DMI", "ADX", "SUPERTREND")
FLOW_FEATURE_HINTS = ("AD_LINE", "OBV", "MFI", "CMF", "FLOW", "MONEY")

TS_OP_HINTS = ("TS_", "DELAY", "DELTA", "DECAY", "EMA", "WMA", "PRODUCT")
NORM_OP_HINTS = ("ZSCORE", "RANK", "SCALE", "NORMALIZE", "WINSORIZE", "CLIP")
NONLINEAR_OP_HINTS = ("SIGNED_LOG", "TANH", "SIGMOID", "ABS", "SQRT", "POWER", "SIGNED_POWER")
GATE_OP_HINTS = ("GATE", "IF_")
ARITH_OPS = {"ADD", "SUB", "MUL", "DIV", "NEG", "MIN", "MAX", "MAX3"}


def token_name(token: int) -> str:
    names = FORMULA_VOCAB.token_names
    return names[token] if 0 <= int(token) < len(names) else ""


def feature_family(name: str) -> str:
    upper = name.upper()
    if any(hint in upper for hint in FLOW_FEATURE_HINTS):
        return "flow"
    if any(hint in upper for hint in VOLUME_FEATURE_HINTS):
        return "volume"
    if any(hint in upper for hint in VOLATILITY_FEATURE_HINTS):
        return "volatility"
    if any(hint in upper for hint in TREND_FEATURE_HINTS):
        return "trend"
    if any(hint in upper for hint in PRICE_FEATURE_HINTS):
        return "price"
    return "other"


def op_family(name: str) -> str:
    upper = name.upper()
    if any(hint in upper for hint in GATE_OP_HINTS):
        return "gate"
    if any(hint in upper for hint in NORM_OP_HINTS):
        return "norm"
    if any(hint in upper for hint in NONLINEAR_OP_HINTS):
        return "nonlinear"
    if any(hint in upper for hint in TS_OP_HINTS):
        return "ts"
    if upper in ARITH_OPS:
        return "arith"
    return "other"


def formula_behavior_key(formula: list[int]) -> tuple:
    op_offset = FORMULA_VOCAB.operator_offset
    names = FORMULA_VOCAB.token_names
    feature_families: Counter[str] = Counter()
    op_families: Counter[str] = Counter()
    first_feature_family = "none"
    feature_count = 0
    op_count = 0
    for raw in formula:
        token = int(raw)
        name = names[token] if 0 <= token < len(names) else ""
        if token < op_offset:
            family = feature_family(name)
            feature_families[family] += 1
            feature_count += 1
            if first_feature_family == "none":
                first_feature_family = family
        else:
            op_families[op_family(name)] += 1
            op_count += 1
    dominant_features = tuple(sorted(k for k, v in feature_families.items() if v > 0))[:3]
    dominant_ops = tuple(sorted(k for k, v in op_families.items() if v > 0))[:4]
    return (
        first_feature_family,
        dominant_features,
        dominant_ops,
        min(op_families.get("ts", 0), 3),
        min(op_families.get("norm", 0), 2),
        min(op_families.get("nonlinear", 0), 2),
        min(op_families.get("gate", 0), 1),
        min(feature_count, 3),
        min(op_count, 5),
        min(len(formula), 10),
    )


def formula_core_signature(formula: list[int]) -> tuple[int, ...]:
    names = FORMULA_VOCAB.token_names
    op_offset = FORMULA_VOCAB.operator_offset
    salient: list[int] = []
    for raw in formula:
        token = int(raw)
        name = names[token] if 0 <= token < len(names) else ""
        family = feature_family(name) if token < op_offset else op_family(name)
        if family in {"flow", "trend", "volatility", "ts", "norm", "gate"}:
            salient.append(token)
    return tuple(sorted(set(salient)))


def token_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a = {int(x) for x in left}
    b = {int(x) for x in right}
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def is_too_similar(formula: list[int], others: Iterable[list[int]], threshold: float) -> bool:
    if threshold >= 1.0:
        return False
    signature = formula_core_signature(formula)
    for other in others:
        if token_jaccard(signature, formula_core_signature(other)) >= threshold:
            return True
    return False
