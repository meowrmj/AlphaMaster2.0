"""Read training progress from checkpoints and strategy files."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"


def _safe_symbol_tag(symbol: str) -> str:
    return symbol.replace(".", "_")


def _safe_timeframe_tag(timeframe: str | None) -> str:
    tag = str(timeframe or "").strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def strategy_path_for(symbol: str, timeframe: str | None = None) -> Path:
    tag = _safe_symbol_tag(symbol)
    tf_tag = _safe_timeframe_tag(timeframe)
    if tf_tag:
        tag = f"{tag}_{tf_tag}"
    return STRATEGIES_DIR / f"best_{tag}.json"


def checkpoint_glob(symbol: str, timeframe: str | None = None) -> list[Path]:
    tag = _safe_symbol_tag(symbol)
    tf_tag = _safe_timeframe_tag(timeframe)
    if tf_tag:
        patterns = [
            f"ckpt_{symbol}_{tf_tag}_step_*.pt",
            f"ckpt_{tag}_{tf_tag}_step_*.pt",
        ]
    else:
        patterns = [
            f"ckpt_{symbol}_step_*.pt",
            f"ckpt_{tag}_step_*.pt",
        ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(CHECKPOINT_DIR.glob(pattern))
    return sorted(set(found), key=lambda p: p.stat().st_mtime)


def _step_from_name(path: Path) -> int:
    m = re.search(r"_step_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 0


@dataclass
class SymbolProgress:
    symbol: str
    train_steps: int
    current_step: int
    best_score: float | None
    best_formula: list[int] | None
    formula_decoded: str | None
    has_strategy: bool
    strategy_score: float | None
    checkpoint_path: str | None
    checkpoint_mtime: float | None
    history: dict[str, Any] | None

    @property
    def progress_pct(self) -> float:
        if self.train_steps <= 0:
            return 0.0
        return min(100.0, 100.0 * self.current_step / self.train_steps)

    @property
    def status(self) -> str:
        if self.current_step >= self.train_steps and self.has_strategy:
            return "completed"
        if self.current_step > 0:
            return "in_progress"
        if self.has_strategy:
            return "strategy_only"
        return "idle"


_ckpt_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_checkpoint_cache() -> None:
    _ckpt_cache.clear()


def _load_checkpoint_meta(path: Path) -> dict[str, Any]:
    mtime = path.stat().st_mtime
    key = str(path)
    cached = _ckpt_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "step": int(ckpt.get("step", _step_from_name(path))),
        "best_score": ckpt.get("best_score"),
        "best_formula": ckpt.get("best_formula"),
        "training_history": ckpt.get("training_history") or {},
    }
    _ckpt_cache[key] = (mtime, meta)
    return meta


def _decode_formula(tokens: list[int] | None) -> str | None:
    if not tokens:
        return None
    names = FORMULA_VOCAB.token_names
    try:
        return " → ".join(names[t] for t in tokens)
    except (IndexError, TypeError):
        return str(tokens)


def _load_strategy(symbol: str, timeframe: str | None = None) -> dict[str, Any] | None:
    path = strategy_path_for(symbol, timeframe)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _infer_timeframe_from_name(name: str) -> str | None:
    stem = Path(name).stem.lower()
    parts = re.split(r"[^a-z0-9]+", stem)
    aliases = {
        "daily": "D1",
        "day": "D1",
        "d1": "D1",
        "1d": "D1",
        "m15": "M15",
        "15m": "M15",
        "15min": "M15",
        "m5": "M5",
        "5m": "M5",
        "5min": "M5",
        "m30": "M30",
        "30m": "M30",
        "30min": "M30",
        "h1": "H1",
        "1h": "H1",
        "h4": "H4",
        "4h": "H4",
    }
    for part in parts:
        if part in aliases:
            return aliases[part]
    return None


def _pick_training_history(
    file_history: dict[str, Any] | None,
    ckpt_history: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """取步数更多的那份历史，避免旧 checkpoint 覆盖较新的 json 曲线。"""
    if not file_history and not ckpt_history:
        return None
    if not file_history:
        return ckpt_history
    if not ckpt_history:
        return file_history
    file_n = len(file_history.get("step") or [])
    ckpt_n = len(ckpt_history.get("step") or [])
    return file_history if file_n >= ckpt_n else ckpt_history


def get_symbol_progress(symbol: str, timeframe: str | None = None) -> SymbolProgress:
    train_steps = ModelConfig.TRAIN_STEPS
    strategy = _load_strategy(symbol, timeframe)
    ckpts = checkpoint_glob(symbol, timeframe)

    current_step = 0
    best_score = None
    best_formula = None
    history: dict[str, Any] | None = None
    ckpt_path: str | None = None
    ckpt_mtime: float | None = None

    hist_tag = _safe_symbol_tag(symbol)
    tf_tag = _safe_timeframe_tag(timeframe)
    if tf_tag:
        hist_tag = f"{hist_tag}_{tf_tag}"
    hist_file = PROJECT_ROOT / f"training_history_{hist_tag}.json"
    file_history: dict[str, Any] | None = None
    if hist_file.exists():
        try:
            file_history = json.loads(hist_file.read_text(encoding="utf-8"))
            steps = file_history.get("step") or []
            if steps:
                # history 存的是 0 起算的训练步索引，展示与日志 [N/5000] 对齐用 N
                current_step = max(current_step, int(steps[-1]) + 1)
            bests = file_history.get("best_score") or []
            if bests:
                best_score = float(bests[-1])
            history = file_history
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    if ckpts:
        latest = ckpts[-1]
        ckpt_path = str(latest.relative_to(PROJECT_ROOT)).replace("\\", "/")
        ckpt_mtime = latest.stat().st_mtime
        try:
            meta = _load_checkpoint_meta(latest)
            current_step = max(current_step, int(meta["step"]))
            if meta.get("best_score") is not None:
                best_score = float(meta["best_score"])
            best_formula = meta.get("best_formula")
            history = _pick_training_history(file_history, meta.get("training_history"))
        except Exception:
            current_step = max(current_step, _step_from_name(latest))

    if strategy:
        if strategy.get("best_score") is not None:
            strategy_score = float(strategy["best_score"])
            if best_score is None or strategy_score > float(best_score):
                best_score = strategy_score
                if strategy.get("formula"):
                    best_formula = strategy["formula"]
        if best_formula is None and strategy.get("formula"):
            best_formula = strategy["formula"]

    return SymbolProgress(
        symbol=symbol,
        train_steps=train_steps,
        current_step=current_step,
        best_score=best_score,
        best_formula=best_formula,
        formula_decoded=_decode_formula(best_formula),
        has_strategy=strategy is not None,
        strategy_score=float(strategy["best_score"]) if strategy and strategy.get("best_score") is not None else None,
        checkpoint_path=ckpt_path,
        checkpoint_mtime=ckpt_mtime,
        history=history,
    )


def get_strategy_for_export(symbol: str, timeframe: str | None = None) -> dict[str, Any]:
    data = _load_strategy(symbol, timeframe)
    if not data:
        raise FileNotFoundError(f"未找到 {symbol} 的策略，请先完成训练")
    out = dict(data)
    formula = out.get("formula")
    if formula and not out.get("formula_decoded"):
        out["formula_decoded"] = _decode_formula(formula)
    return out


def build_strategy_export_filename(
    symbol: str,
    step: int,
    score: float | None,
    timeframe: str | None = None,
) -> str:
    """e.g. strategy_ADAUSD_step0084_score2.4021.json"""
    safe = symbol.replace(".", "_")
    tf = _safe_timeframe_tag(timeframe)
    if tf:
        safe = f"{safe}_{tf}"
    step_part = f"step{max(0, int(step)):04d}"
    if score is not None:
        return f"strategy_{safe}_{step_part}_score{float(score):.4f}.json"
    return f"strategy_{safe}_{step_part}.json"


def list_strategies() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not STRATEGIES_DIR.exists():
        return rows
    for path in sorted(STRATEGIES_DIR.glob("best_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        formula = data.get("formula")
        symbol = data.get("symbol") or path.stem.replace("best_", "", 1)
        timeframe = data.get("timeframe")
        inferred_timeframe = None if timeframe else _infer_timeframe_from_name(path.name)
        display_timeframe = timeframe or inferred_timeframe
        expected = strategy_path_for(str(symbol), str(timeframe) if timeframe else None).name
        is_canonical = path.name == expected
        rows.append({
            "file": path.name,
            "symbol": symbol,
            "timeframe": timeframe,
            "display_timeframe": display_timeframe,
            "timeframe_source": "json" if timeframe else ("filename" if inferred_timeframe else "unknown"),
            "best_score": data.get("best_score"),
            "formula_decoded": data.get("formula_decoded") or _decode_formula(formula),
            "train_steps": data.get("train_steps"),
            "mode": data.get("mode"),
            "is_canonical": is_canonical,
            "is_legacy": not is_canonical,
        })
    return rows
