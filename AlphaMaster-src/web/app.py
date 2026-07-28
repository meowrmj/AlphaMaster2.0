"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import inspect_parquet_file
from model_core.config import ModelConfig
from web.file_dialog import pick_data_root_dir, pick_parquet_file, pick_strategy_file
from web.progress import (
    STRATEGIES_DIR,
    _decode_formula,
    _load_checkpoint_meta,
    checkpoint_glob,
    get_symbol_progress,
    get_strategy_for_export,
    invalidate_checkpoint_cache,
    list_strategies,
    build_strategy_export_filename,
)
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import load_settings, save_settings
from web.strategy_file import (
    inspect_strategy_file,
    resolve_strategy_file,
    strategy_path_for_symbol,
    sync_best_strategy_for_symbol,
)
from web.training_manager import training_manager
from web.training_time import get_training_time_summary
from web.training_package import build_training_export_zip, import_training_package
from web.backtest_manager import backtest_manager
from web.realtime_manager import realtime_manager
from web.data_sources.factory import list_sources
from strategy_manager.live_signal import min_exposure

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"

setup_logging()
logger = get_logger()

app = FastAPI(title="AlphaMaster Training", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _project_relative_path(path: str | Path) -> str:
    p = Path(str(path)).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _with_relative_strategy_file(info: dict[str, Any]) -> dict[str, Any]:
    out = dict(info)
    strategy_file = str(out.get("strategy_file") or "").strip()
    if strategy_file:
        out["strategy_file"] = _project_relative_path(strategy_file)
    return out


class StartTrainingRequest(BaseModel):
    data_file: str
    from_scratch: bool = False
    eval_mode: str = "cpu_batch"


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    data_root_dir: str | None = None
    last_strategy_file: str | None = None
    debug_mode: bool | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    bt_commission_pct: float | None = None
    bt_slippage_pct: float | None = None
    realtime_source: str | None = None


class AnalyzeTrainingRequest(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    symbol: str | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str
    data_file: str | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None


class AddWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str


class RemoveWatchRequest(BaseModel):
    id: str


class FeishuSettingsRequest(BaseModel):
    enabled: bool | None = None
    webhook_url: str | None = None
    secret: str | None = None


class FeishuTestRequest(BaseModel):
    webhook_url: str | None = None
    secret: str | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_error(f"{request.method} {request.url.path} unhandled", exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    if is_debug_mode():
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    if response.status_code >= 400:
        log_error(f"{request.method} {request.url.path} -> HTTP {response.status_code}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_error(f"{request.method} {request.url.path} HTTP {exc.status_code}: {exc.detail}")
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_error(f"{request.method} {request.url.path} crashed", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )


def _inspect_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_parquet_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        l_path = str(Path(left).resolve())
        r_path = str(Path(right).resolve())
    except OSError:
        l_path = str(left)
        r_path = str(right)
    return l_path.casefold() == r_path.casefold()


def _browse_data_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening native file picker")
    settings = load_settings()
    initialdir = Path(str(settings.get("data_root_dir") or "")).expanduser()
    if not initialdir.is_dir():
        initialdir = None
    try:
        path = pick_parquet_file(initialdir=initialdir)
    except Exception as exc:
        log_error("File picker failed", exc)
        raise HTTPException(500, f"鏂囦欢閫夋嫨澶辫触: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("File picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected file: %s", path)
    info = _inspect_or_http(path)
    stopped_training = None
    status = training_manager.status()
    job = status.get("job") or {}
    if status.get("active") and not _same_path(job.get("data_file"), info.get("data_file")):
        if is_debug_mode():
            logger.info(
                "Data file changed from %s to %s; stopping active training",
                job.get("data_file"),
                info.get("data_file"),
            )
        stopped = training_manager.stop()
        stopped_training = {
            "ok": stopped,
            "previous_data_file": job.get("data_file"),
            "previous_symbol": job.get("symbol"),
            "previous_timeframe": job.get("timeframe"),
        }
    save_settings({"last_data_file": info["data_file"]})
    return {
        "ok": True,
        "cancelled": False,
        "stopped_training": stopped_training,
        **info,
    }


def _browse_data_root_dir() -> dict[str, Any]:
    settings = load_settings()
    initialdir = Path(str(settings.get("data_root_dir") or "")).expanduser()
    if not initialdir.is_dir():
        initialdir = ROOT.parent
    try:
        path = pick_data_root_dir(initialdir=initialdir)
    except Exception as exc:
        log_error("Data root picker failed", exc)
        raise HTTPException(500, f"数据源文件夹选择失败: {exc}") from exc
    if not path:
        return {"ok": True, "cancelled": True}
    root = Path(path).resolve()
    if not root.is_dir():
        raise HTTPException(400, f"不是有效文件夹: {root}")
    settings = save_settings({"data_root_dir": str(root)})
    return {
        "ok": True,
        "cancelled": False,
        "data_root_dir": settings.get("data_root_dir") or str(root),
    }


def _strategy_context() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    train_symbol = None
    train_timeframe = None
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            train_symbol = info.get("symbol")
            train_timeframe = info.get("timeframe")
        except Exception:
            pass

    resolved = resolve_strategy_file(
        settings.get("last_strategy_file") or "",
        train_symbol,
        train_timeframe,
    )
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(
                resolved,
                data_file_hint=settings.get("last_data_file") or None,
            )
        except Exception as e:
            strategy_info = {
                "strategy_file": resolved,
                "valid": False,
                "message": str(e),
            }
    return {
        "last_strategy_file": _project_relative_path(resolved) if resolved else "",
        "strategy_file": _with_relative_strategy_file(strategy_info) if strategy_info else None,
        "train_symbol": train_symbol,
        "train_timeframe": train_timeframe,
    }


def _browse_strategy_file() -> dict[str, Any]:
    raise HTTPException(
        400,
        "策略文件选择已改为页面内下拉列表，避免 Windows Tk 文件框导致服务崩溃。",
    )


def _inspect_strategy_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_strategy_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _infer_data_file_from_name(path: Path) -> dict[str, Any] | None:
    stem = path.stem
    if "_" not in stem:
        return None
    symbol, raw_tf = stem.rsplit("_", 1)
    aliases = {
        "daily": "D1",
        "day": "D1",
        "d1": "D1",
        "1d": "D1",
        "15min": "M15",
        "15m": "M15",
        "m15": "M15",
        "60min": "H1",
        "1h": "H1",
        "h1": "H1",
        "30min": "M30",
        "30m": "M30",
        "m30": "M30",
    }
    timeframe = aliases.get(raw_tf.lower(), raw_tf.upper())
    return {"symbol": symbol, "timeframe": timeframe}


def _list_local_parquet_files(symbol_filter: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    settings = load_settings()
    configured_root = Path(str(settings.get("data_root_dir") or "")).expanduser()
    roots = [
        configured_root if configured_root.is_dir() else ROOT.parent / "AlphaMaster-data" / "parquet",
        ROOT / "data",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    wanted = str(symbol_filter or "").strip()
    for base in roots:
        if not base.exists():
            continue
        pattern = f"{wanted}_*.parquet" if wanted else "*.parquet"
        for path in sorted(base.rglob(pattern)):
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            info = _infer_data_file_from_name(path)
            if not info:
                continue
            if wanted and str(info.get("symbol") or "") != wanted:
                continue
            rows.append(
                {
                    "data_file": str(path.resolve()),
                    "relative_path": _project_relative_path(path),
                    "filename": path.name,
                    "symbol": info.get("symbol"),
                    "timeframe": info.get("timeframe"),
                    "bars": None,
                }
            )
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    rows.sort(key=lambda r: (str(r.get("symbol") or ""), str(r.get("timeframe") or ""), str(r.get("filename") or "")))
    return rows


def _write_current_run_strategy(symbol: str, timeframe: str | None, data_file: str | None = None) -> dict[str, Any] | None:
    ckpts = checkpoint_glob(symbol, timeframe)
    if not ckpts:
        return None
    latest = ckpts[-1]
    meta = _load_checkpoint_meta(latest)
    formula = meta.get("best_formula")
    score = meta.get("best_score")
    if not formula or score is None:
        return None
    tf = str(timeframe or "").strip().upper() or None
    suffix = symbol.replace(".", "_") + (f"_{tf}" if tf else "")
    out_path = STRATEGIES_DIR / f"current_run_best_{suffix}.json"
    payload = {
        "vocab_version": "checkpoint",
        "symbol": symbol,
        "timeframe": tf,
        "formula": formula,
        "formula_decoded": _decode_formula(formula),
        "best_score": float(score),
        "train_step": int(meta.get("step") or 0),
        "source": "current_run_checkpoint",
        "checkpoint_path": str(latest.relative_to(ROOT)).replace("\\", "/"),
    }
    if data_file:
        payload["data_file"] = data_file
        payload["mode"] = "parquet_file"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    info = inspect_strategy_file(str(out_path.resolve()), data_file_hint=data_file)
    return _with_relative_strategy_file(info)


def _resolve_train_symbol(symbol: str | None = None) -> str | None:
    if symbol:
        return symbol.strip() or None
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return inspect_parquet_file(data_file).get("symbol")
    except Exception:
        return None


def _resolve_train_identity(
    symbol: str | None = None,
    timeframe: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    sym = symbol.strip() if symbol else None
    tf = str(timeframe or "").strip().upper() or None
    data_file: str | None = None

    job = training_manager.status().get("job") or {}
    if (not sym or str(job.get("symbol") or "") == sym) and job.get("symbol"):
        sym = sym or str(job.get("symbol") or "").strip()
        tf = tf or str(job.get("timeframe") or "").strip().upper() or None
        data_file = job.get("data_file") or None

    settings = load_settings()
    if not data_file:
        data_file = settings.get("last_data_file") or None
    if data_file and (not sym or not tf):
        try:
            info = inspect_parquet_file(data_file)
            if not sym:
                sym = info.get("symbol")
            if not tf and (not sym or info.get("symbol") == sym):
                tf = str(info.get("timeframe") or "").strip().upper() or None
        except Exception:
            pass
    return sym, tf, data_file


def _wait_training_idle(timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        training_manager.status()
        if not training_manager.status().get("active"):
            return
        time.sleep(0.2)


def _sync_and_persist_best_strategy(
    symbol: str,
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any] | None:
    invalidate_checkpoint_cache()
    hint = data_file_hint
    if not hint:
        job = training_manager.status().get("job") or {}
        if str(job.get("symbol") or "") == symbol:
            hint = job.get("data_file") or None
    if not hint:
        hint = load_settings().get("last_data_file") or None
    timeframe = None
    job = training_manager.status().get("job") or {}
    if str(job.get("symbol") or "") == symbol and job.get("timeframe"):
        timeframe = str(job.get("timeframe") or "").strip().upper() or None
    if hint:
        try:
            timeframe = timeframe or inspect_parquet_file(hint).get("timeframe")
        except Exception:
            pass
    info = sync_best_strategy_for_symbol(symbol, data_file_hint=hint, timeframe=timeframe)
    if info:
        rel_info = _with_relative_strategy_file(info)
        save_settings({"last_strategy_file": rel_info["strategy_file"]})
        return rel_info
    return info


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.1.0"}


@app.get("/api/routes")
def api_routes() -> dict[str, Any]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": sorted(routes, key=lambda r: r["path"])}


@app.get("/api/debug/logs")
def api_debug_logs(lines: int = 200) -> dict[str, Any]:
    return debug_snapshot(lines)


@app.post("/api/debug/client-log")
def api_client_log(req: ClientLogRequest) -> dict[str, bool]:
    msg = req.message
    if req.context:
        msg = f"{msg} | context={req.context}"
    if req.level == "error":
        log_error(f"[client] {msg}")
    elif is_debug_mode():
        logger.info("[client] %s", msg)
    return {"ok": True}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return load_settings()


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.last_data_file is not None:
        payload["last_data_file"] = req.last_data_file
    if req.data_root_dir is not None:
        root_text = str(req.data_root_dir or "").strip()
        if root_text:
            root = Path(root_text).expanduser()
            if not root.is_dir():
                raise HTTPException(400, f"数据源文件夹不存在: {root}")
            payload["data_root_dir"] = str(root.resolve())
        else:
            payload["data_root_dir"] = ""
    if req.last_strategy_file is not None:
        payload["last_strategy_file"] = req.last_strategy_file
    if req.debug_mode is not None:
        payload["debug_mode"] = req.debug_mode
    if req.ai_provider is not None:
        payload["ai_provider"] = req.ai_provider
    if req.ai_api_key is not None:
        payload["ai_api_key"] = req.ai_api_key
    if req.bt_commission_pct is not None:
        payload["bt_commission_pct"] = req.bt_commission_pct
    if req.bt_slippage_pct is not None:
        payload["bt_slippage_pct"] = req.bt_slippage_pct
    if req.realtime_source is not None:
        payload["realtime_source"] = req.realtime_source
    saved = save_settings(payload)
    if req.debug_mode is not None:
        set_debug_mode(req.debug_mode)
    return {"ok": True, **saved}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
        except Exception as e:
            file_info = {
                "data_file": data_file,
                "valid": False,
                "message": str(e),
            }
    snap = debug_snapshot(1)
    strat_ctx = _strategy_context()
    return {
        "train_steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "reward_mode": ModelConfig.REWARD_MODE,
        "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        "device": str(ModelConfig.DEVICE),
        "data_root_dir": settings.get("data_root_dir", ""),
        "last_data_file": data_file,
        "data_file": file_info,
        "last_strategy_file": strat_ctx["last_strategy_file"],
        "strategy_file": strat_ctx["strategy_file"],
        "debug_mode": load_settings().get("debug_mode", False),
        "ai_provider": load_settings().get("ai_provider", "deepseek"),
        "ai_api_key": load_settings().get("ai_api_key", ""),
        "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
        "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
        "realtime_source": settings.get("realtime_source", "mt5"),
        "server_log": snap["server_log"],
        "error_log": snap["error_log"],
    }


@app.get("/api/ai/providers")
def api_ai_providers() -> dict[str, Any]:
    from web.ai_providers import provider_status

    status = provider_status()
    settings = load_settings()
    status["selected"] = settings.get("ai_provider", "deepseek")
    status["has_api_key"] = bool(settings.get("ai_api_key"))
    return status


@app.post("/api/ai/analyze-training")
def api_ai_analyze_training(req: AnalyzeTrainingRequest):
    from fastapi.responses import StreamingResponse

    from web.ai_analyze import analyze_training_stream

    settings = load_settings()
    raw_key = req.api_key if req.api_key is not None else settings.get("ai_api_key") or ""
    key_lower = str(raw_key).strip().lower()

    # openclaw_wb 蹇呴』鍏堜簬 openclaw 鍒ゆ柇
    if key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        provider = "openclaw_wb"
    elif key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        provider = "openclaw"
    else:
        provider = (req.provider or settings.get("ai_provider") or "deepseek").strip()

    save_settings({
        "ai_provider": provider,
        "ai_api_key": str(raw_key).strip(),
    })

    def event_gen():
        try:
            for event in analyze_training_stream(
                provider=provider,
                api_key=str(raw_key).strip() or None,
                symbol=req.symbol,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file() -> dict[str, Any]:
    return _browse_data_file()


@app.post("/api/data-root/browse")
@app.get("/api/data-root/browse")
def api_browse_data_root() -> dict[str, Any]:
    return _browse_data_root_dir()


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file() -> dict[str, Any]:
    return _browse_strategy_file()


@app.get("/api/backtest/options")
def api_backtest_options(symbol: str | None = None) -> dict[str, Any]:
    settings = load_settings()
    wanted_symbol = (symbol or "").strip() or None
    data_files = _list_local_parquet_files(wanted_symbol)
    strategies = []
    for s in list_strategies():
        path = Path(str(s.get("file") or ""))
        if not path.is_absolute():
            path = (STRATEGIES_DIR / path).resolve()
        if not path.exists():
            continue
        strategies.append(
            {
                "kind": "saved",
                "label": f"{s.get('symbol')} {s.get('display_timeframe') or s.get('timeframe') or '未标注'} · 保存冠军 · 分数 {float(s.get('best_score') or 0):.3f}",
                "strategy_file": _project_relative_path(path),
                "symbol": s.get("symbol"),
                "timeframe": s.get("display_timeframe") or s.get("timeframe"),
                "best_score": s.get("best_score"),
                "formula_decoded": s.get("formula_decoded"),
            }
        )

    for df in data_files:
        sym = str(df.get("symbol") or "")
        tf = str(df.get("timeframe") or "").strip().upper() or None
        cur = _write_current_run_strategy(sym, tf, df.get("data_file"))
        if not cur:
            continue
        strategies.append(
            {
                "kind": "current_run",
                "label": f"{sym} {tf or ''} · 本轮最优 checkpoint · 分数 {float(cur.get('best_score') or 0):.3f}",
                "strategy_file": cur.get("strategy_file"),
                "symbol": sym,
                "timeframe": tf,
                "best_score": cur.get("best_score"),
                "formula_decoded": cur.get("formula_decoded"),
            }
        )

    strategies.sort(key=lambda r: (str(r.get("symbol") or ""), str(r.get("timeframe") or ""), str(r.get("kind") or "")))
    return {
        "strategies": strategies,
        "data_files": data_files,
        "last_strategy_file": settings.get("last_strategy_file") or "",
        "last_data_file": settings.get("last_data_file") or "",
    }


@app.post("/api/strategy-file/sync-best")
@app.get("/api/strategy-file/sync-best")
def api_sync_best_strategy(symbol: str | None = None) -> dict[str, Any]:
    sym = _resolve_train_symbol(symbol)
    if not sym:
        raise HTTPException(400, "请先选择训练数据文件或指定品种")
    info = _sync_and_persist_best_strategy(sym)
    if not info:
        raise HTTPException(404, f"未找到 {sym} 的可用策略")
    return {"ok": True, **info}


def _progress_with_live_step(
    symbol: str,
    active: bool,
    timeframe: str | None = None,
) -> dict[str, Any]:
    p = get_symbol_progress(symbol, timeframe)
    current_step = p.current_step
    if active:
        live = training_manager.parse_step_from_log()
        if live is not None:
            current_step = max(current_step, live)
    train_steps = p.train_steps
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    val_score = None
    hist = p.history or {}
    vals = hist.get("val_score") or []
    if vals:
        try:
            val_score = float(vals[-1])
        except (TypeError, ValueError):
            val_score = None
    stagnation_steps = None
    bests = hist.get("best_score") or []
    steps = hist.get("step") or []
    if bests and steps:
        try:
            latest_best = float(bests[-1])
            last_breakthrough_idx = 0
            for idx in range(len(bests) - 1, -1, -1):
                if abs(float(bests[idx]) - latest_best) > 1e-12:
                    last_breakthrough_idx = min(idx + 1, len(steps) - 1)
                    break
            else:
                last_breakthrough_idx = 0
            last_breakthrough_step = int(steps[last_breakthrough_idx])
            stagnation_steps = max(0, current_step - last_breakthrough_step)
        except (TypeError, ValueError, IndexError):
            stagnation_steps = None
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "val_score": val_score,
        "champion_score": p.best_score,
        "candidate_val_score": val_score,
        "stagnation_steps": stagnation_steps,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
    }


def _attach_training_time(
    row: dict[str, Any] | None,
    *,
    symbol: str | None,
    timeframe: str | None = None,
    job: dict[str, Any] | None,
    active: bool,
) -> dict[str, Any] | None:
    if not row or not symbol:
        return row
    summary = get_training_time_summary(
        symbol,
        timeframe=timeframe,
        job=job,
        active=active,
    )
    row = dict(row)
    row["session_seconds"] = summary.session_seconds
    row["history_total_seconds"] = summary.history_total_seconds
    return row


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    training = training_manager.status()
    job = training.get("job")
    active = bool(training.get("active"))

    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
            sym = file_info.get("symbol")
            row = _progress_with_live_step(sym, active=False, timeframe=file_info.get("timeframe"))
            progress = {
                "symbol": row["symbol"],
                "timeframe": file_info.get("timeframe"),
                "status": row["status"],
                "current_step": row["current_step"],
                "train_steps": row["train_steps"],
                "progress_pct": row["progress_pct"],
                "best_score": row["best_score"],
                "val_score": row.get("val_score"),
                "champion_score": row.get("champion_score"),
                "candidate_val_score": row.get("candidate_val_score"),
                "stagnation_steps": row.get("stagnation_steps"),
                "formula_decoded": row["formula_decoded"],
                "has_checkpoint": row.get("has_checkpoint", False),
                "has_strategy": row.get("has_strategy", False),
            }
            progress = _attach_training_time(
                progress,
                symbol=sym,
                timeframe=file_info.get("timeframe"),
                job=job,
                active=active and job and job.get("symbol") == sym,
            )
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    if job and job.get("symbol") and active:
        sym = job["symbol"]
        row = _progress_with_live_step(sym, active=True, timeframe=job.get("timeframe"))
        progress = {
            "symbol": row["symbol"],
            "timeframe": job.get("timeframe"),
            "status": "running_job",
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "val_score": row.get("val_score"),
            "champion_score": row.get("champion_score"),
            "candidate_val_score": row.get("candidate_val_score"),
            "stagnation_steps": row.get("stagnation_steps"),
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
        }
        progress = _attach_training_time(
            progress,
            symbol=sym,
            timeframe=job.get("timeframe"),
            job=job,
            active=True,
        )

    return {
        "data_file": file_info,
        "progress": progress,
        "training": training,
    }


@app.get("/api/symbols/{symbol}")
def api_symbol(symbol: str, timeframe: str | None = None) -> dict[str, Any]:
    tf = str(timeframe or "").strip().upper() or None
    if not tf:
        job = training_manager.status().get("job") or {}
        if str(job.get("symbol") or "") == symbol and job.get("timeframe"):
            tf = str(job.get("timeframe") or "").strip().upper() or None
    if not tf:
        data_file = load_settings().get("last_data_file") or ""
        if data_file:
            try:
                info = inspect_parquet_file(data_file)
                if info.get("symbol") == symbol:
                    tf = str(info.get("timeframe") or "").strip().upper() or None
            except Exception:
                pass
    p = get_symbol_progress(symbol, tf)
    return {
        "symbol": p.symbol,
        "timeframe": tf,
        "status": p.status,
        "current_step": p.current_step,
        "train_steps": p.train_steps,
        "progress_pct": round(p.progress_pct, 1),
        "best_score": p.best_score,
        "best_formula": p.best_formula,
        "formula_decoded": p.formula_decoded,
        "has_strategy": p.has_strategy,
        "strategy_score": p.strategy_score,
        "checkpoint_path": p.checkpoint_path,
        "history": p.history,
    }


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list_strategies()}


@app.get("/api/strategies/{symbol}/export")
def api_export_strategy(symbol: str, timeframe: str | None = None):
    import json

    from fastapi.responses import Response

    sym, tf, _ = _resolve_train_identity(symbol, timeframe)
    sym = sym or symbol
    try:
        payload = get_strategy_for_export(sym, tf)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    progress = get_symbol_progress(sym, tf)
    step = progress.current_step
    score = payload.get("best_score")
    if score is None:
        score = progress.strategy_score if progress.strategy_score is not None else progress.best_score
    filename = build_strategy_export_filename(sym, step, score, tf)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.get("/api/training/{symbol}/export")
def api_export_training(symbol: str, timeframe: str | None = None):
    from fastapi.responses import Response

    sym, tf, _ = _resolve_train_identity(symbol, timeframe)
    sym = sym or symbol
    try:
        body, zip_name = build_training_export_zip(sym, tf)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.post("/api/training/import")
async def api_import_training(
    file: UploadFile = File(...),
    symbol: str | None = Query(None, description="当前选择的品种，用于校验导入包是否一致"),
) -> dict[str, Any]:
    if training_manager.status().get("active"):
        raise HTTPException(409, "训练进行中，请先停止再导入")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "涓婁紶鏂囦欢涓虹┖")

    try:
        return import_training_package(
            raw,
            file.filename or "upload.zip",
            expected_symbol=symbol or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/training/status")
def api_training_status() -> dict[str, Any]:
    status = training_manager.status()
    status["log_tail"] = training_manager.tail_log(150)
    return status


@app.post("/api/training/start")
def api_training_start(req: StartTrainingRequest) -> dict[str, Any]:
    info = _inspect_or_http(req.data_file)
    save_settings({"last_data_file": info["data_file"]})
    try:
        job = training_manager.start(
            data_file=info["data_file"],
            symbol=info["symbol"],
            timeframe=info["timeframe"],
            mode="ftmo",
            from_scratch=bool(req.from_scratch),
            eval_mode=req.eval_mode,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if req.from_scratch:
        invalidate_checkpoint_cache()
    return {
        "ok": True,
        "job": job.to_dict(),
        "data_file": info,
        "from_scratch": bool(req.from_scratch),
        "eval_mode": job.eval_mode,
    }


@app.post("/api/training/stop")
def api_training_stop() -> dict[str, Any]:
    job = training_manager.status().get("job") or {}
    symbol = job.get("symbol")
    data_file_hint = job.get("data_file")
    stopped = training_manager.stop()
    strategy_file = None
    if symbol:
        _wait_training_idle()
        strategy_file = _sync_and_persist_best_strategy(
            symbol,
            data_file_hint=data_file_hint,
        )
    return {
        "ok": stopped,
        "training": training_manager.status(),
        "strategy_file": strategy_file,
    }


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鍥炴祴 API
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "profit_loss_ratio",
    "n_trades", "win_rate", "avg_hold_bars",
)


def _load_backtest_report() -> dict[str, Any] | None:
    import json

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if not report_path.exists():
        return None
    return _load_json_lenient(report_path)


def _load_json_lenient(path: Path) -> dict[str, Any] | None:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            data = json.loads(path.read_text(encoding=encoding))
            return data if isinstance(data, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            continue
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _backtest_focus_symbol(symbol: str | None = None) -> str | None:
    """Resolve the symbol used to filter backtest charts/report for the web UI."""
    if symbol:
        return symbol.strip() or None

    job = backtest_manager.status().get("job") or {}
    if job.get("symbol"):
        return str(job["symbol"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("symbol"):
        return str(strat["symbol"])

    report = _load_backtest_report()
    if report:
        keys = list((report.get("symbols") or {}).keys())
        if len(keys) == 1:
            return keys[0]
    return None


def _backtest_focus_timeframe() -> str | None:
    job = backtest_manager.status().get("job") or {}
    if job.get("timeframe"):
        return str(job["timeframe"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("timeframe"):
        return str(strat["timeframe"])
    return None


def _filter_report_for_symbol(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = report.get("symbols") or {}
    if symbol not in symbols:
        return report

    sym_data = symbols[symbol]
    return {
        **report,
        "focus_symbol": symbol,
        "symbols": {symbol: sym_data},
        "portfolio": {
            "total_return": sym_data.get("total_return"),
            "sharpe": sym_data.get("sharpe"),
            "sortino": sym_data.get("sortino"),
            "profit_loss_ratio": sym_data.get("profit_loss_ratio"),
            "n_trades": sym_data.get("n_trades"),
            "win_rate": sym_data.get("win_rate"),
        },
    }


def _list_backtest_charts(symbol: str | None = None) -> list[dict[str, str]]:
    """List backtest charts; single-symbol mode only returns matching charts."""
    if not BACKTEST_OUTPUT_DIR.exists():
        return []

    if symbol:
        charts: list[dict[str, str]] = []
        equity = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
        if equity.exists():
            charts.append(
                {"name": equity.name, "label": f"{symbol} 璧勯噾鏇茬嚎", "kind": "equity"}
            )
        return charts

    charts = []
    portfolio = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
    if portfolio.exists():
        charts.append({"name": "portfolio_equity.png", "label": "缁勫悎璧勯噾鏇茬嚎", "kind": "portfolio"})
    for path in sorted(BACKTEST_OUTPUT_DIR.glob("equity_*.png")):
        sym = path.stem.replace("equity_", "", 1)
        charts.append({"name": path.name, "label": f"{sym} 璧勯噾鏇茬嚎", "kind": "symbol"})
    return charts


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    status = backtest_manager.status()
    status["log_tail"] = backtest_manager.tail_log(200)
    return status


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    info = _inspect_strategy_or_http(req.strategy_file)
    settings = load_settings()
    commission = (
        float(req.commission_pct)
        if req.commission_pct is not None
        else float(settings.get("bt_commission_pct", 0.02))
    )
    slippage = (
        float(req.slippage_pct)
        if req.slippage_pct is not None
        else float(settings.get("bt_slippage_pct", 0.01))
    )
    if commission < 0 or slippage < 0:
        raise HTTPException(400, "手续费和滑点不能为负数")

    rel_info = _with_relative_strategy_file(info)
    save_settings({
        "last_strategy_file": rel_info["strategy_file"],
        "bt_commission_pct": commission,
        "bt_slippage_pct": slippage,
    })

    data_file: str | None = None
    explicit_data = (req.data_file or "").strip()
    if explicit_data:
        try:
            pf = inspect_parquet_file(explicit_data)
            if pf.get("valid") is False:
                raise HTTPException(400, f"选择的数据文件无效: {pf.get('message') or explicit_data}")
            if info.get("symbol") and pf.get("symbol") != info.get("symbol"):
                raise HTTPException(400, f"数据文件品种 {pf.get('symbol')} 与策略品种 {info.get('symbol')} 不一致")
            data_file = pf["data_file"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"选择的数据文件无法加载: {explicit_data}\n{e}") from e
    # 1) 浼樺厛鐢ㄧ瓥鐣?JSON 閲岃?褰曠殑璁?粌鏁版嵁璺?緞
    strat_data = (info.get("data_file") or "").strip()
    if not data_file and strat_data:
        try:
            pf = inspect_parquet_file(strat_data)
            if pf.get("valid") is False:
                raise HTTPException(
                    400,
                    f"绛栫暐璁板綍鐨勬暟鎹?枃浠舵棤鏁? {pf.get('message') or strat_data}",
                )
            data_file = pf["data_file"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                400,
                f"绛栫暐璁板綍鐨勬暟鎹?枃浠舵棤娉曞姞杞? {strat_data}\n{e}",
            ) from e
    else:
        # 2) 鍥為€€锛氳?缁冮〉鏈€杩戦€夋嫨鐨勩€佸悓鍝佺? Parquet
        last_data = settings.get("last_data_file") or ""
        if last_data:
            try:
                pf = inspect_parquet_file(last_data)
                if pf.get("symbol") == info.get("symbol") and pf.get("valid") is not False:
                    data_file = pf["data_file"]
            except Exception:
                pass

    if not data_file:
        raise HTTPException(
            400,
            "该策略没有可用数据文件。请在回测页选择同品种 Parquet 数据源后再开始回测。",
        )

    save_settings({"last_data_file": data_file})

    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
            commission_pct=commission,
            slippage_pct=slippage,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "strategy_file": rel_info, "data_file": data_file}


@app.post("/api/backtest/stop")
def api_backtest_stop() -> dict[str, Any]:
    stopped = backtest_manager.stop()
    return {"ok": stopped, "backtest": backtest_manager.status()}


@app.get("/api/backtest/report")
def api_backtest_report(symbol: str | None = None) -> dict[str, Any]:
    report = _load_backtest_report()
    focus = _backtest_focus_symbol(symbol)
    focus_timeframe = _backtest_focus_timeframe()
    if report and focus:
        report = _filter_report_for_symbol(report, focus)
    return {
        "available": report is not None,
        "report": report,
        "charts": _list_backtest_charts(focus),
        "focus_symbol": focus,
        "focus_timeframe": focus_timeframe,
    }


@app.get("/api/backtest/equity")
def api_backtest_equity(symbol: str | None = None) -> dict[str, Any]:
    """Return raw equity curve data for the interactive backtest chart."""
    path = BACKTEST_OUTPUT_DIR / "equity_curve.json"
    focus = _backtest_focus_symbol(symbol)
    focus_timeframe = _backtest_focus_timeframe()
    if not path.exists():
        return {"available": False, "focus_symbol": focus, "focus_timeframe": focus_timeframe, "data": None}
    data = _load_json_lenient(path)
    if not data:
        return {"available": False, "focus_symbol": focus, "focus_timeframe": focus_timeframe, "data": None}

    # 鍗曞搧绉嶆ā寮忥細鍙?繚鐣欒仛鐒﹀搧绉嶏紝鍘绘帀鏃犲叧搴忓垪
    if focus and isinstance(data.get("symbols"), dict) and focus in data["symbols"]:
        data = {
            **data,
            "symbols": {focus: data["symbols"][focus]},
        }
        data.pop("portfolio", None)

    return {"available": True, "focus_symbol": focus, "focus_timeframe": focus_timeframe, "data": data}


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    # 闃叉?璺?緞绌胯秺锛氫粎鍏佽?杈撳嚭鐩?綍鍐呯殑 png 鏂囦欢
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(400, "非法文件名")
    path = (BACKTEST_OUTPUT_DIR / name).resolve()
    try:
        path.relative_to(BACKTEST_OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "非法路径") from None
    if not path.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 瀹炴椂琛屾儏鍒嗘瀽 API
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


@app.on_event("startup")
def _startup_realtime() -> None:
    try:
        realtime_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("realtime load_persisted failed", exc)


@app.get("/api/realtime/sources")
def api_realtime_sources() -> dict[str, Any]:
    return {
        "sources": list_sources(),
        "min_exposure": min_exposure(),
        "selected_source": load_settings().get("realtime_source", "mt5"),
    }


@app.post("/api/realtime/tradingview/probe")
def api_realtime_tradingview_probe() -> dict[str, Any]:
    """Probe TradingView reachability (same behavior as PA_Agent before fetch)."""
    from web.data_sources.tradingview_connectivity import (
        TV_CLOUD_SERVER_WIKI_URL,
        TV_CONNECTIVITY_MESSAGE,
        check_tradingview_connectivity,
    )

    ok, detail = check_tradingview_connectivity(
        timeout_s=15.0, max_attempts=2, retry_delay_s=2.0
    )
    return {
        "ok": ok,
        "detail": detail,
        "blocked": not ok,
        "title": "鏃犳硶浣跨敤 TradingView",
        "message": None if ok else TV_CONNECTIVITY_MESSAGE,
        "wiki_url": TV_CLOUD_SERVER_WIKI_URL,
    }


@app.get("/api/realtime/strategies")
def api_realtime_strategies() -> dict[str, Any]:
    """Return saved best_*.json strategies for realtime dropdowns."""
    rows = []
    for s in list_strategies():
        sym = s.get("symbol")
        if not sym:
            continue
        path = Path(str(s.get("file") or ""))
        if not path.is_absolute():
            path = (STRATEGIES_DIR / path).resolve()
        if not path.exists():
            continue
        rows.append(
            {
                "symbol": sym,
                "timeframe": s.get("timeframe"),
                "display_timeframe": s.get("display_timeframe"),
                "timeframe_source": s.get("timeframe_source"),
                "best_score": s.get("best_score"),
                "formula_decoded": s.get("formula_decoded"),
                "filename": s.get("file"),
                "is_legacy": s.get("is_legacy"),
                "is_canonical": s.get("is_canonical"),
                "strategy_file": _project_relative_path(path),
            }
        )
    rows.sort(
        key=lambda r: (
            str(r.get("symbol") or ""),
            str(r.get("display_timeframe") or ""),
            bool(r.get("is_legacy")),
            -(float(r.get("best_score") or 0)),
        )
    )
    return {"strategies": rows}


@app.get("/api/realtime/status")
def api_realtime_status() -> dict[str, Any]:
    return realtime_manager.status()


@app.post("/api/realtime/watch")
def api_realtime_watch(req: AddWatchRequest) -> dict[str, Any]:
    try:
        watch = realtime_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "watch": watch}


@app.post("/api/realtime/unwatch")
def api_realtime_unwatch(req: RemoveWatchRequest) -> dict[str, Any]:
    removed = realtime_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/realtime/start")
def api_realtime_start() -> dict[str, Any]:
    realtime_manager.start()
    return {"ok": True, **realtime_manager.status()}


@app.post("/api/realtime/stop")
def api_realtime_stop() -> dict[str, Any]:
    realtime_manager.stop()
    return {"ok": True, "running": False}


@app.get("/api/realtime/feishu")
def api_realtime_feishu_get() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("feishu_enabled")),
        "webhook_url": s.get("feishu_webhook_url") or "",
        "secret": s.get("feishu_secret") or "",
    }


@app.put("/api/realtime/feishu")
def api_realtime_feishu_put(req: FeishuSettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.enabled is not None:
        payload["feishu_enabled"] = bool(req.enabled)
    if req.webhook_url is not None:
        payload["feishu_webhook_url"] = req.webhook_url
    if req.secret is not None:
        payload["feishu_secret"] = req.secret
    saved = save_settings(payload)
    return {
        "ok": True,
        "enabled": bool(saved.get("feishu_enabled")),
        "webhook_url": saved.get("feishu_webhook_url") or "",
        "secret": saved.get("feishu_secret") or "",
    }


@app.post("/api/realtime/feishu/test")
def api_realtime_feishu_test(req: FeishuTestRequest) -> dict[str, Any]:
    from web.feishu_notify import send_text

    url = (req.webhook_url or "").strip()
    if not url:
        url = (load_settings().get("feishu_webhook_url") or "").strip()
    if not url:
        raise HTTPException(400, "请先填写 Webhook URL")
    secret = req.secret
    if secret is None:
        secret = load_settings().get("feishu_secret") or ""
    ok, msg = send_text(
        "AlphaMaster 飞书通知测试：配置正常，信号方向转折时会推送提醒。",
        webhook_url=url,
        secret=secret or "",
    )
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

