"""Native file pickers for local training UI."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES_DIR = PROJECT_ROOT / "strategies"


def _make_root():
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    return root


def _open_file_dialog(
    title: str,
    filetypes: list[tuple[str, str]],
    initialdir: Path | None = None,
) -> str | None:
    try:
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）") from exc

    root = _make_root()
    try:
        path = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir) if initialdir else None,
            filetypes=filetypes,
        )
    finally:
        root.destroy()
    return path or None


def _open_directory_dialog(title: str, initialdir: Path | None = None) -> str | None:
    try:
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）") from exc

    root = _make_root()
    try:
        path = filedialog.askdirectory(
            title=title,
            initialdir=str(initialdir) if initialdir else None,
            mustexist=True,
        )
    finally:
        root.destroy()
    return path or None


def pick_parquet_file(initialdir: Path | None = None) -> str | None:
    return _open_file_dialog(
        "选择 K 线 Parquet 文件",
        [
            ("Parquet K线", "*.parquet"),
            ("所有文件", "*.*"),
        ],
        initialdir=initialdir,
    )


def pick_data_root_dir(initialdir: Path | None = None) -> str | None:
    return _open_directory_dialog("选择数据源文件夹", initialdir=initialdir or PROJECT_ROOT)


def pick_strategy_file() -> str | None:
    return _open_file_dialog(
        "选择策略 JSON 文件",
        [
            ("策略 JSON", "*.json"),
            ("所有文件", "*.*"),
        ],
        initialdir=STRATEGIES_DIR if STRATEGIES_DIR.is_dir() else PROJECT_ROOT,
    )
