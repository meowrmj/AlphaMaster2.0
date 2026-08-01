from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = PROJECT_ROOT / "logs" / "training_control"


def _safe_tag(value: str | None) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def control_key(symbol: str | None, timeframe: str | None, algorithm_mode: str | None) -> str:
    return "_".join(
        [
            _safe_tag(algorithm_mode or "rl"),
            _safe_tag(symbol),
            _safe_tag(timeframe),
        ]
    )


def request_path(symbol: str | None, timeframe: str | None, algorithm_mode: str | None) -> Path:
    return CONTROL_DIR / f"{control_key(symbol, timeframe, algorithm_mode)}.stop.json"


def ack_path(symbol: str | None, timeframe: str | None, algorithm_mode: str | None) -> Path:
    return CONTROL_DIR / f"{control_key(symbol, timeframe, algorithm_mode)}.ack.json"


def request_checkpoint_stop(
    *,
    symbol: str,
    timeframe: str | None,
    algorithm_mode: str,
    pid: int | None,
    reason: str,
) -> Path:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "algorithm_mode": algorithm_mode,
        "pid": int(pid) if pid is not None else None,
        "reason": reason,
        "requested_at": time.time(),
    }
    path = request_path(symbol, timeframe, algorithm_mode)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    old_ack = ack_path(symbol, timeframe, algorithm_mode)
    try:
        old_ack.unlink(missing_ok=True)
    except OSError:
        pass
    return path


def read_checkpoint_stop_request(
    *,
    symbol: str | None,
    timeframe: str | None,
    algorithm_mode: str | None,
) -> dict[str, Any] | None:
    path = request_path(symbol, timeframe, algorithm_mode)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    requested_pid = payload.get("pid")
    if requested_pid is not None:
        requested_pid = int(requested_pid)
        if requested_pid not in {os.getpid(), os.getppid()}:
            return None
    return payload if isinstance(payload, dict) else None


def acknowledge_checkpoint_stop(
    *,
    symbol: str | None,
    timeframe: str | None,
    algorithm_mode: str | None,
    step: int,
    checkpoint_path: str,
) -> Path:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "algorithm_mode": algorithm_mode,
        "pid": os.getpid(),
        "step": int(step),
        "checkpoint_path": checkpoint_path,
        "ack_at": time.time(),
    }
    path = ack_path(symbol, timeframe, algorithm_mode)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    try:
        request_path(symbol, timeframe, algorithm_mode).unlink(missing_ok=True)
    except OSError:
        pass
    return path
