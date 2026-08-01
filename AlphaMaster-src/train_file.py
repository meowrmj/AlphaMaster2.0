"""Train AlphaMaster from one local Parquet K-line file."""
from __future__ import annotations

import glob as _glob
import json
import os
import pathlib
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from data_pipeline.data_manager import MT5DataManager
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.ga_engine import GeneticAlphaEngine
from model_core.vocab import VOCAB_VERSION

DEFAULT_TRAIN_RATIO = 0.80


def _safe_artifact_tag(value: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())


def _artifact_suffix(symbol: str, timeframe: str | None = None) -> str:
    tf = _safe_artifact_tag(timeframe)
    return f"{_safe_artifact_tag(symbol)}_{tf}" if tf else _safe_artifact_tag(symbol)


def _strategy_path(
    symbol: str,
    timeframe: str | None = None,
    algorithm_mode: str = "rl",
) -> pathlib.Path:
    mode = algorithm_mode if algorithm_mode in {"rl", "ga", "hybrid"} else "rl"
    return pathlib.Path("strategies") / "champions" / mode / f"best_{mode}_{_artifact_suffix(symbol, timeframe)}.json"


def _algorithm_mode() -> str:
    mode = os.getenv("ALPHAMASTER_ALGORITHM_MODE", ModelConfig.ALGORITHM_MODE).strip().lower()
    return mode if mode in {"rl", "ga", "hybrid"} else "rl"


def _checkpoint_pattern(symbol: str, timeframe: str | None = None, algorithm_mode: str = "rl") -> str:
    suffix = _artifact_suffix(symbol, timeframe)
    if algorithm_mode == "ga":
        return str(pathlib.Path("checkpoints") / "ga" / f"ckpt_ga_{suffix}_gen_*.pt")
    if algorithm_mode == "hybrid":
        return str(pathlib.Path("checkpoints") / "hybrid" / f"ckpt_hybrid_{suffix}_step_*.pt")
    return str(pathlib.Path("checkpoints") / f"ckpt_{suffix}_step_*.pt")


def _history_path(
    symbol: str,
    timeframe: str | None = None,
    algorithm_mode: str = "rl",
) -> pathlib.Path:
    suffix = _artifact_suffix(symbol, timeframe)
    if algorithm_mode == "ga":
        return pathlib.Path(f"training_history_ga_{suffix}.json")
    if algorithm_mode == "hybrid":
        return pathlib.Path(f"training_history_hybrid_{suffix}.json")
    return pathlib.Path(f"training_history_{suffix}.json")


def _same_scope(data: dict, symbol: str, timeframe: str) -> bool:
    file_symbol = data.get("symbol")
    file_timeframe = data.get("timeframe")
    return (
        (not file_symbol or str(file_symbol) == str(symbol))
        and (not file_timeframe or str(file_timeframe).upper() == str(timeframe).upper())
    )


def _metadata(symbol: str, timeframe: str, data_file: str, total_bars: int, train_end: int) -> dict:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "train_steps": ModelConfig.TRAIN_STEPS,
        "train_sample": "train_only",
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "total_bars": total_bars,
        "train_end_bar": train_end,
        "oos_start_bar": train_end,
    }


def _merge_metadata(data: dict, symbol: str, timeframe: str, data_file: str, total_bars: int, train_end: int) -> dict:
    merged = dict(data)
    had_split = bool(merged.get("train_sample") and merged.get("oos_start_bar"))
    for key, val in _metadata(symbol, timeframe, data_file, total_bars, train_end).items():
        if val is not None and not merged.get(key):
            merged[key] = val
    if not had_split:
        merged["metadata_inferred"] = True
        merged["metadata_inferred_reason"] = "legacy_strategy_missing_train_split_fields"
    return merged


def train_from_file(data_file: str, *, from_scratch: bool = False) -> AlphaEngine | None:
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]
    algorithm_mode = _algorithm_mode()
    if algorithm_mode == "hybrid":
        print("[algorithm] Hybrid 框架已预留，但当前版本尚未实现训练逻辑。请先选择 RL 或 GA。")
        return None

    print(f"\n{'=' * 60}")
    print(f"  AlphaMaster file training - {info['filename']}")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  File: {Path(data_file).resolve()}")
    print(f"  Train steps: {ModelConfig.TRAIN_STEPS}")
    print(f"  Bars: {info['bars']}")
    print(f"  Mode: {'from_scratch' if from_scratch else 'resume'}")
    print(f"  Algorithm: {algorithm_mode.upper()}")
    print(f"  Eval device: {ModelConfig.DEVICE}  batch_eval={ModelConfig.GPU_BATCH_EVAL}")
    print(f"  Search plugins: {os.getenv('ALPHAMASTER_SEARCH_CONFIG', '{}')}")
    print(f"{'=' * 60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        total_bars = mgr.raw_dict["open"].shape[1]
        train_end = max(3, min(total_bars - 2, int(total_bars * DEFAULT_TRAIN_RATIO)))
        mgr._raw_dict = {k: v[:, :train_end].clone() for k, v in mgr.raw_dict.items()}
        mgr._target_ret = MT5DataManager._compute_target_ret(mgr.raw_dict["open"])
        print(f"  Data loaded: {total_bars} bars")
        print(f"  No-leak split: train bars=[0,{train_end}); OOS bars=[{train_end},{total_bars})")
    except Exception as e:
        print(f"  [error] data loading failed: {e}")
        return None

    engine_cls = GeneticAlphaEngine if algorithm_mode == "ga" else AlphaEngine
    engine = engine_cls(data_manager=mgr, target_symbol=symbol)
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = f"{algorithm_mode}_parquet_file"
    engine.algorithm_mode = algorithm_mode
    engine.train_steps = ModelConfig.TRAIN_STEPS
    engine.train_sample = "train_only"
    engine.train_ratio = DEFAULT_TRAIN_RATIO
    engine.total_bars = total_bars
    engine.train_end_bar = train_end
    engine.oos_start_bar = train_end

    ckpt_files = sorted(_glob.glob(_checkpoint_pattern(symbol, timeframe, algorithm_mode)))
    start_step = 0

    if from_scratch:
        removed = 0
        for p in ckpt_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [warn] could not remove checkpoint {p}: {e}")
        hist_path = _history_path(symbol, timeframe, algorithm_mode)
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [retrain] removed {removed} checkpoints; starting at step 0")
        print(f"  [retrain] removed {algorithm_mode.upper()} checkpoints only; saved champion is kept on disk")
        ckpt_files = []
    elif ckpt_files:
        latest = ckpt_files[-1]
        try:
            start_step = engine.load_checkpoint(latest)
            print(f"  [resume] loaded {latest}; start_step={start_step}")
        except Exception as e:
            print(f"  [warn] checkpoint load failed: {e}; starting from step 0")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [done] {symbol} already completed {ModelConfig.TRAIN_STEPS} steps")
        _save_strategy(engine, symbol, timeframe, data_file, total_bars, train_end)
        return engine

    if start_step == 0 and not from_scratch:
        hist_path = _history_path(symbol, timeframe)
        if hist_path.exists():
            hist_path.unlink()
        print("  [new] starting from step 0")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol, timeframe, data_file, total_bars, train_end)
    return engine


def _seed_best_from_strategy(
    engine: AlphaEngine,
    symbol: str,
    timeframe: str,
    data_file: str,
    total_bars: int,
    train_end: int,
) -> None:
    path = _strategy_path(symbol, timeframe, getattr(engine, "algorithm_mode", "rl"))
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] could not read existing strategy: {e}")
        return
    if not _same_scope(data, symbol, timeframe):
        print(f"  [strategy] ignored {path}: symbol/timeframe mismatch")
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    try:
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        merged = _merge_metadata(data, symbol, timeframe, data_file, total_bars, train_end)
        if merged != data:
            path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [strategy] completed legacy metadata for {path}")
        print(f"  [retrain] score floor is {engine.best_score:.4f}; only higher scores overwrite it")
    except (TypeError, ValueError) as e:
        print(f"  [warn] existing strategy cannot be used as score floor: {e}")


def _save_strategy(engine: AlphaEngine, symbol: str, timeframe: str, data_file: str, total_bars: int, train_end: int) -> None:
    algorithm_mode = getattr(engine, "algorithm_mode", "rl")
    path = _strategy_path(symbol, timeframe, algorithm_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine.best_formula is None:
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                merged = _merge_metadata(old, symbol, timeframe, data_file, total_bars, train_end)
                if merged != old:
                    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"  [strategy] completed legacy metadata for {path}")
                print("  [strategy] no new run winner; kept existing saved champion")
                return
            except (json.JSONDecodeError, OSError):
                pass
        print("  [strategy] no valid formula found; nothing saved")
        return
    if path.exists() and engine.best_formula is not None:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_score = old.get("best_score")
            if _same_scope(old, symbol, timeframe) and old_score is not None and float(old_score) > float(engine.best_score):
                merged = _merge_metadata(old, symbol, timeframe, data_file, total_bars, train_end)
                if merged != old:
                    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"  [strategy] completed legacy metadata for {path}")
                print(f"  [strategy] kept stronger disk score {float(old_score):.4f} > this run {float(engine.best_score):.4f}")
                return
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    data = {
        "vocab_version": VOCAB_VERSION,
        **_metadata(symbol, timeframe, data_file, total_bars, train_end),
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula) if engine.best_formula else None,
        "best_score": engine.best_score,
        "algorithm_mode": algorithm_mode,
        "strategy_source": "champion",
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Strategy saved: {path}")


if __name__ == "__main__":
    ModelConfig.REWARD_MODE = "ftmo"

    if "--data-file" not in sys.argv:
        print(r"Usage: python train_file.py --data-file PATH\TO\SYMBOL_TF.parquet [--from-scratch]")
        sys.exit(1)

    idx = sys.argv.index("--data-file")
    if idx + 1 >= len(sys.argv):
        print("Error: --data-file requires a path")
        sys.exit(1)

    data_file = sys.argv[idx + 1]
    from_scratch = "--from-scratch" in sys.argv
    t0 = time.time()
    eng = train_from_file(data_file, from_scratch=from_scratch)
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] training finished: best_score={eng.best_score:.4f}, elapsed={elapsed / 3600:.2f}h")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< training failed")
        sys.exit(1)
