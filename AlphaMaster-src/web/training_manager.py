"""Subprocess manager for train_file.py jobs."""
from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import threading
import time
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi
from model_core.training_control import request_checkpoint_stop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
JOB_STATE_PATH = LOG_DIR / "training_job_state.json"

EVAL_MODES = {"cpu_batch", "cuda_batch", "legacy_cpu"}
ALGORITHM_MODES = {"rl", "ga", "hybrid"}
REPLAY_MODULES = {"qd", "incubation"}
REPLAY_POLICIES = {"qd_incubation", "qd", "incubation", "none"}
SEARCH_MODULES = {"annealing", "genetic"}
VCVARS64_BAT = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
CUDA_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
_NATIVE_TOOLCHAIN_ENV_CACHE: dict[str, str] | None = None
_NATIVE_TOOLCHAIN_ENV_LOCK = threading.Lock()


def _direct_child_pids(parent_pid: int | None) -> list[int]:
    if os.name != "nt" or not parent_pid:
        return []
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return []

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    children: list[int] = []
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32ParentProcessID) == int(parent_pid):
                children.append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return children


def _control_pid_for_process(proc: subprocess.Popen | None) -> int | None:
    parent_pid = getattr(proc, "pid", None)
    children = _direct_child_pids(parent_pid)
    return children[0] if len(children) == 1 else parent_pid


def _normalize_eval_mode(value: str | None) -> str:
    mode = str(value or "cpu_batch").strip().lower()
    return mode if mode in EVAL_MODES else "cpu_batch"


def _normalize_algorithm_mode(value: str | None) -> str:
    mode = str(value or "rl").strip().lower()
    return mode if mode in ALGORITHM_MODES else "rl"


def _legacy_replay_modules(value: str | None) -> dict[str, bool]:
    policy = str(value or "qd_incubation").strip().lower()
    return {
        "qd": policy in {"qd_incubation", "qd", "hybrid"},
        "incubation": policy in {"qd_incubation", "incubation", "hybrid"},
    }


def _replay_policy_name(modules: dict[str, bool]) -> str:
    qd = bool(modules.get("qd"))
    incubation = bool(modules.get("incubation"))
    if qd and incubation:
        return "qd_incubation"
    if qd:
        return "qd"
    if incubation:
        return "incubation"
    return "none"


def _normalize_replay_config(value: Any | None) -> tuple[str, dict[str, Any]]:
    if isinstance(value, dict):
        raw_modules = value.get("modules")
        if isinstance(raw_modules, dict):
            modules = {
                key: bool(raw_modules.get(key, True))
                for key in REPLAY_MODULES
            }
        else:
            modules = _legacy_replay_modules(str(value.get("name") or value.get("policy") or "qd_incubation"))
    else:
        modules = _legacy_replay_modules(str(value or "qd_incubation"))
    config = {
        "version": 1,
        "modules": modules,
    }
    return _replay_policy_name(modules), config


def _normalize_search_config(value: Any | None) -> dict[str, Any]:
    if isinstance(value, dict):
        raw_modules = value.get("modules")
        if isinstance(raw_modules, dict):
            modules = {key: bool(raw_modules.get(key, False)) for key in SEARCH_MODULES}
        else:
            modules = {key: False for key in SEARCH_MODULES}
    elif isinstance(value, str):
        parts = {p.strip().lower() for p in value.split(",") if p.strip()}
        modules = {key: key in parts for key in SEARCH_MODULES}
    else:
        modules = {key: False for key in SEARCH_MODULES}
    return {"version": 1, "modules": modules}


def _apply_eval_mode_env(env: dict[str, str], eval_mode: str) -> None:
    if eval_mode == "cuda_batch":
        env["ALPHAMASTER_DEVICE"] = "cuda"
        env["ALPHAMASTER_GPU_BATCH_EVAL"] = "1"
        env["ALPHAMASTER_GPU_BATCH_EVAL_STRICT"] = "1"
        env["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "1"
        env["ALPHAMASTER_NATIVE_OP_POLICY"] = "aggressive"
    elif eval_mode == "legacy_cpu":
        env["ALPHAMASTER_DEVICE"] = "cpu"
        env["ALPHAMASTER_GPU_BATCH_EVAL"] = "0"
        env["ALPHAMASTER_GPU_BATCH_EVAL_STRICT"] = "1"
        env["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "0"
        env.pop("ALPHAMASTER_NATIVE_OP_POLICY", None)
    else:
        env["ALPHAMASTER_DEVICE"] = "cpu"
        env["ALPHAMASTER_GPU_BATCH_EVAL"] = "1"
        env["ALPHAMASTER_GPU_BATCH_EVAL_STRICT"] = "1"
        env["ALPHAMASTER_NATIVE_FORMULA_OPS"] = "0"
        env.pop("ALPHAMASTER_NATIVE_OP_POLICY", None)


def _apply_native_toolchain_env(env: dict[str, str]) -> None:
    """Inject the MSVC/CUDA build environment needed by torch C++ extensions."""
    if os.name != "nt":
        return
    global _NATIVE_TOOLCHAIN_ENV_CACHE
    with _NATIVE_TOOLCHAIN_ENV_LOCK:
        cached = _NATIVE_TOOLCHAIN_ENV_CACHE
        if cached is None:
            if not VCVARS64_BAT.exists():
                raise RuntimeError(f"native CUDA toolchain missing: {VCVARS64_BAT}")
            nvcc = CUDA_HOME / "bin" / "nvcc.exe"
            if not nvcc.exists():
                raise RuntimeError(f"native CUDA nvcc missing: {nvcc}")

            cmd = f'call "{VCVARS64_BAT}" >nul && set'
            output = subprocess.check_output(
                cmd,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            cached = {}
            for line in output.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key:
                    cached[key] = value
            cuda_home = str(CUDA_HOME)
            cached["CUDA_HOME"] = cuda_home
            cached["CUDA_PATH"] = cuda_home
            vc_tools = cached.get("VCToolsInstallDir", "")
            cl_dir = Path(vc_tools) / "bin" / "Hostx64" / "x64" if vc_tools else None
            prefix = [str(CUDA_HOME / "bin"), str(CUDA_HOME / "libnvvp")]
            if cl_dir and (cl_dir / "cl.exe").exists():
                prefix.insert(0, str(cl_dir))
            cached["_ALPHAMASTER_NATIVE_PATH_PREFIX"] = ";".join(prefix)
            _NATIVE_TOOLCHAIN_ENV_CACHE = cached

    path_prefix = cached.get("_ALPHAMASTER_NATIVE_PATH_PREFIX", "")
    for key, value in cached.items():
        if key != "_ALPHAMASTER_NATIVE_PATH_PREFIX":
            env[key] = value
    if path_prefix:
        env["PATH"] = ";".join([path_prefix, env.get("PATH", "")])


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingJob:
    data_file: str
    symbol: str
    timeframe: str
    mode: str
    algorithm_mode: str = "rl"
    eval_mode: str = "cpu_batch"
    replay_policy: str = "qd_incubation"
    replay_config: dict[str, Any] | None = None
    search_config: dict[str, Any] | None = None
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_file": self.data_file,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "algorithm_mode": self.algorithm_mode,
            "eval_mode": self.eval_mode,
            "replay_policy": self.replay_policy,
            "replay_config": self.replay_config,
            "search_config": self.search_config,
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJob":
        state_value = str(data.get("state") or JobState.RUNNING.value)
        try:
            state = JobState(state_value)
        except ValueError:
            state = JobState.RUNNING
        return cls(
            data_file=str(data.get("data_file") or ""),
            symbol=str(data.get("symbol") or ""),
            timeframe=str(data.get("timeframe") or ""),
            mode=str(data.get("mode") or "ftmo"),
            algorithm_mode=_normalize_algorithm_mode(data.get("algorithm_mode")),
            eval_mode=_normalize_eval_mode(data.get("eval_mode")),
            replay_policy=str(data.get("replay_policy") or "qd_incubation"),
            replay_config=data.get("replay_config") if isinstance(data.get("replay_config"), dict) else None,
            search_config=data.get("search_config") if isinstance(data.get("search_config"), dict) else None,
            state=state,
            pid=int(data["pid"]) if data.get("pid") is not None else None,
            log_path=str(data.get("log_path") or ""),
            started_at=str(data.get("started_at") or ""),
            finished_at=data.get("finished_at"),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
        )


class ExternalProcessHandle:
    """Small handle for a train_file.py process recovered after web restart."""

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)

    def poll(self) -> int | None:
        return None if _pid_exists(self.pid) else 0

    def terminate(self) -> None:
        _terminate_pid_tree(self.pid, force=True)

    def kill(self) -> None:
        _terminate_pid_tree(self.pid, force=True)

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.1)
        return 0


def _pid_exists(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _terminate_pid_tree(pid: int, *, force: bool) -> None:
    if os.name == "nt":
        cmd = ["taskkill", "/PID", str(int(pid)), "/T"]
        if force:
            cmd.append("/F")
        subprocess.run(cmd, capture_output=True, text=True)
        return
    os.kill(int(pid), signal.SIGKILL if force else signal.SIGTERM)


def _load_job_state() -> dict[str, Any] | None:
    try:
        data = json.loads(JOB_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_job_state(job: TrainingJob) -> None:
    tmp = JOB_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(JOB_STATE_PATH)


def _read_windows_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*train_file.py*' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate | "
        "ConvertTo-Json -Depth 3"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        return []
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    return data if isinstance(data, list) else []


def _find_running_train_process() -> dict[str, Any] | None:
    root = str(PROJECT_ROOT).lower()
    rows = []
    for row in _read_windows_processes():
        cmd = str(row.get("CommandLine") or "")
        cmd_lower = cmd.lower()
        if "train_file.py" not in cmd_lower or "--data-file" not in cmd_lower:
            continue
        if root not in cmd_lower:
            continue
        rows.append(row)
    if not rows:
        return None
    pids = {int(r.get("ProcessId") or 0) for r in rows}
    parents = [
        r for r in rows
        if int(r.get("ParentProcessId") or 0) not in pids
    ]
    return parents[0] if parents else rows[0]


def _cmd_arg(cmd: str, name: str) -> str | None:
    marker = f"{name} "
    idx = cmd.find(marker)
    if idx < 0:
        return None
    rest = cmd[idx + len(marker):].strip()
    if not rest:
        return None
    if rest[0] == '"':
        end = rest.find('"', 1)
        return rest[1:end] if end > 1 else rest[1:]
    return rest.split()[0]


def _infer_identity_from_data_file(data_file: str) -> tuple[str, str]:
    stem = Path(data_file).stem.lower()
    symbol = stem.split("_", 1)[0].upper()
    timeframe = "H1"
    if "daily" in stem or stem.endswith("_d1"):
        timeframe = "D1"
    elif "15min" in stem or "m15" in stem:
        timeframe = "M15"
    elif "5min" in stem or "m5" in stem:
        timeframe = "M5"
    elif "60min" in stem or "h1" in stem:
        timeframe = "H1"
    return symbol, timeframe


def _latest_log_for_symbol(symbol: str) -> str:
    logs = sorted(
        LOG_DIR.glob(f"train_{symbol.replace('.', '_')}_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        return ""
    return str(logs[0].relative_to(PROJECT_ROOT)).replace("\\", "/")


class TrainingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | ExternalProcessHandle | None = None
        self._job: TrainingJob | None = None
        self._log_fp = None
        self._stopped_by_user = False
        self._recorded_log_paths: set[str] = set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._recover_state_if_needed(discover=self._job is None)
            self._refresh_state()
            return {
                "active": self._job is not None and self._job.state == JobState.RUNNING,
                "job": self._job.to_dict() if self._job else None,
            }

    def start(
        self,
        data_file: str,
        symbol: str,
        timeframe: str,
        mode: str = "ftmo",
        *,
        from_scratch: bool = False,
        algorithm_mode: str = "rl",
        eval_mode: str = "cpu_batch",
        replay_policy: Any = "qd_incubation",
        search_plugins: Any = None,
    ) -> TrainingJob:
        with self._lock:
            self._recover_state_if_needed(discover=True)
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                sym = self._job.symbol if self._job else "unknown"
                raise RuntimeError(f"已有训练任务在运行: {sym}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_sym = symbol.replace(".", "_")
            log_path = LOG_DIR / f"train_{safe_sym}_{ts}.log"

            hist_path = PROJECT_ROOT / f"training_history_{symbol}.json"
            try:
                hist_path.unlink(missing_ok=True)
            except OSError:
                pass

            cmd = [
                sys.executable,
                "-u",
                "train_file.py",
                "--data-file",
                data_file,
            ]
            if from_scratch:
                cmd.append("--from-scratch")

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"
            algorithm_mode = _normalize_algorithm_mode(algorithm_mode)
            eval_mode = _normalize_eval_mode(eval_mode)
            replay_policy, replay_config = _normalize_replay_config(replay_policy)
            search_config = _normalize_search_config(search_plugins)
            _apply_eval_mode_env(env, eval_mode)
            if eval_mode == "cuda_batch":
                _apply_native_toolchain_env(env)
            env["ALPHAMASTER_ALGORITHM_MODE"] = algorithm_mode
            env["ALPHAMASTER_REPLAY_POLICY"] = replay_policy
            env["ALPHAMASTER_REPLAY_CONFIG"] = json.dumps(replay_config, ensure_ascii=False, separators=(",", ":"))
            env["ALPHAMASTER_SEARCH_CONFIG"] = json.dumps(search_config, ensure_ascii=False, separators=(",", ":"))

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._job = TrainingJob(
                data_file=data_file,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                algorithm_mode=algorithm_mode,
                eval_mode=eval_mode,
                replay_policy=replay_policy,
                replay_config=replay_config,
                search_config=search_config,
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            _write_job_state(self._job)
            return self._job

    def stop(self, reason: str = "unspecified") -> bool:
        with self._lock:
            self._recover_state_if_needed(discover=self._job is None)
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            if self._log_fp:
                try:
                    self._log_fp.write(f"\n[WebStop] reason={reason}\n")
                    self._log_fp.flush()
                except Exception:
                    pass
            try:
                if os.name == "nt" and getattr(self._proc, "pid", None):
                    _terminate_pid_tree(int(self._proc.pid), force=True)
                else:
                    self._proc.terminate()
            except Exception:
                self._proc.kill()
            return True

    def stop_after_checkpoint(self, reason: str = "mode_switch", timeout_s: float = 120.0) -> bool:
        with self._lock:
            self._recover_state_if_needed(discover=self._job is None)
            self._refresh_state()
            if self._proc is None or self._proc.poll() is not None or self._job is None:
                return False
            self._stopped_by_user = True
            request_checkpoint_stop(
                symbol=self._job.symbol,
                timeframe=self._job.timeframe,
                algorithm_mode=self._job.algorithm_mode,
                pid=_control_pid_for_process(self._proc),
                reason=reason,
            )
            if self._log_fp:
                try:
                    self._log_fp.write(f"\n[WebCheckpointStop] reason={reason}\n")
                    self._log_fp.flush()
                except Exception:
                    pass
            proc = self._proc

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.status()
                return True
            time.sleep(0.2)
        raise TimeoutError(f"training process did not checkpoint-stop within {timeout_s:.0f}s")

    def parse_step_from_log(self) -> int | None:
        """从日志尾部解析当前步数，用于 checkpoint 写入前的进度展示。"""
        import re

        for line in reversed(self.tail_log(80)):
            m = re.search(r"\[(\d+)/\d+\]", line)
            if m:
                return int(m.group(1))
        return None

    def tail_log(self, lines: int = 200) -> list[str]:
        with self._lock:
            self._recover_state_if_needed(discover=self._job is None)
            if not self._job or not self._job.log_path:
                return []
            path = PROJECT_ROOT / self._job.log_path
            if not path.exists():
                return []
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return []
            return [strip_ansi(line) for line in content.splitlines()[-lines:]]

    def _refresh_state(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._job.state == JobState.RUNNING:
            if self._stopped_by_user:
                self._job.state = JobState.STOPPED
            elif code == 0:
                self._job.state = JobState.COMPLETED
            elif code in (-signal.SIGTERM, 1) and sys.platform != "win32":
                self._job.state = JobState.STOPPED
            elif code < 0:
                self._job.state = JobState.STOPPED
            else:
                self._job.state = JobState.FAILED
        if self._job.state == JobState.FAILED and self._job.error is None:
            self._job.error = f"训练进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 训练进程已结束，退出码: {code}\n")
            except OSError:
                pass
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._record_session_time()
        _write_job_state(self._job)
        self._proc = None

    def _recover_state_if_needed(self, *, discover: bool) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._job is not None and self._job.state == JobState.RUNNING:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                return

        saved = _load_job_state()
        if saved:
            job = TrainingJob.from_dict(saved)
            if job.state != JobState.RUNNING:
                self._job = job
                self._proc = None
                return
            if job.pid and _pid_exists(int(job.pid)):
                job.state = JobState.RUNNING
                self._job = job
                self._proc = ExternalProcessHandle(int(job.pid))
                return
            job.state = JobState.STOPPED
            job.finished_at = job.finished_at or datetime.now(timezone.utc).isoformat()
            job.exit_code = job.exit_code if job.exit_code is not None else 0
            self._job = job
            self._proc = None
            _write_job_state(job)
            return

        if not discover:
            return
        proc = _find_running_train_process()
        if not proc:
            return

        pid = int(proc.get("ProcessId") or 0)
        cmd = str(proc.get("CommandLine") or "")
        data_file = _cmd_arg(cmd, "--data-file") or ""
        symbol, timeframe = _infer_identity_from_data_file(data_file)
        replay_policy, replay_config = _normalize_replay_config(None)
        job = TrainingJob(
            data_file=data_file,
            symbol=symbol,
            timeframe=timeframe,
            mode="ftmo",
            algorithm_mode="rl",
            eval_mode="cpu_batch",
            replay_policy=replay_policy,
            replay_config=replay_config,
            search_config=_normalize_search_config(None),
            state=JobState.RUNNING,
            pid=pid,
            log_path=_latest_log_for_symbol(symbol),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._job = job
        self._proc = ExternalProcessHandle(pid)
        _write_job_state(job)

    def _record_session_time(self) -> None:
        job = self._job
        if job is None or not job.log_path or not job.started_at:
            return
        rel = job.log_path.replace("\\", "/")
        if rel in self._recorded_log_paths:
            return
        if job.state == JobState.RUNNING:
            return
        from web.training_time import record_training_session

        record_training_session(
            symbol=job.symbol,
            timeframe=job.timeframe,
            started_at=job.started_at,
            finished_at=job.finished_at,
            log_path=rel,
        )
        self._recorded_log_paths.add(rel)


training_manager = TrainingManager()
