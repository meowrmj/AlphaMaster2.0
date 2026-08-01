"""Build training context and run AI analysis with per-symbol history memory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from data_pipeline.parquet_manager import inspect_parquet_file
from web.ai_providers import resolve_provider
from web.progress import PROJECT_ROOT, _decode_formula, _load_checkpoint_meta, checkpoint_glob, get_symbol_progress
from web.settings import load_settings
from web.training_manager import training_manager

HISTORY_PATH = PROJECT_ROOT / "ai_analysis_history.json"
_MAX_HISTORY_PER_KEY = 5

_SYSTEM_PROMPT = """你是量化因子挖掘与强化学习训练顾问。用户正在用 AlphaMaster / AlphaGPT 训练可解释因子公式。

回答要求：
- 用中文，尽量说人话，少用术语。
- 不要编造快照里没有的数据。
- 结论必须先分清“训练流程是否正常”和“策略质量是否值得信任”，不要混成一句话。
- 评估是否值得继续训练时，必须同时看 val_score、best_score、batch_best_val_score、new_candidate_best_val_score、entropy、elite_replay_used、timing_*。不要只看单个指标。
- val_score 是当前批次/训练步的平均验证表现，不等于冠军策略分数；它可以为负，但同一批里仍可能出现高分候选。
- best_score 是本轮运行最大值/冠军线，天然会长时间横盘；横盘不等于程序卡死。判断是否停滞，要结合 new_candidate_best_val_score 是否还在接近或冲击 best_score。
- batch_best_val_score 是本批最高分，可能包含精英回放；new_candidate_best_val_score 是本步不含精英回放的新候选最高分，更适合判断模型是否真的在发现新东西。
- entropy 上升通常表示重新探索/分布变松，不要简单说成“瞎探索”；entropy 很低且多样性下降才更像坍缩。
- elite_replay_used 表示本步注入了多少条优秀旧公式。当前系统使用 QD 优秀池 + 冷却恢复机制：重启后先降低精英回放，再逐步恢复，目的是减少旧冠军把模型拉回同一方向。
- incubation_pool / incubation_replay_used 是“新方向孵化池”：重启后的新候选如果结构有差异，即使暂时弱于历史冠军，也会被短期保护和少量回放，用来验证新方向是否能成长。
- 不要因为保存冠军分高、本轮候选分低，就直接说训练坏了；重新训练时本轮分数低于历史保存冠军是正常的。
- timing_total_ms / timing_eval_ms / timing_sample_elite_ms / timing_grad_ms 用来判断性能瓶颈；如果存在这些字段，说明训练正在记录拆分耗时。
- “saved_champion” 是当前保存下来的冠军策略，可能来自更早训练。
- “current_run_best” 是本轮训练最新 checkpoint 里的本轮最优策略。
- 第二部分必须分别解释 saved_champion 和 current_run_best 的公式含义；如果两者相同，要明确说它们一致。
- 公式是栈式 VM 公式，不一定能按“第一步输入第二步”机械解释成线性因果链；解释时可以说大体看哪些量价/波动/方向信息，但不要武断断言一定做多或做空，最终方向要以信号输出和回测为准。

必须使用这两个小标题：

## 1. 当前训练情况怎么样？是否值得继续
说明进度、训练是否活着、当前模式、平均验证分、本批最高分、新候选最高分、本轮最优分、熵、精英回放和耗时瓶颈。先判断流程是否正常，再判断策略质量是否可靠，并给出继续训练、观察到某个步数、或先回测的建议。

## 2. 因子的含义与原理
分别解释：
- 保存冠军公式 saved_champion
- 本轮最优公式 current_run_best
解释每条公式大概用了哪些信息，以及可能反映的市场状态。必须提醒：这是栈式公式的近似解释，交易方向和有效性要以回测结果为准。
"""


def _current_run_best(symbol: str, timeframe: str) -> dict[str, Any] | None:
    ckpts = checkpoint_glob(symbol, timeframe)
    if not ckpts:
        return None
    try:
        meta = _load_checkpoint_meta(ckpts[-1])
    except Exception:
        return None
    formula = meta.get("best_formula")
    score = meta.get("best_score")
    if not formula or score is None:
        return None
    return {
        "score": float(score),
        "formula": formula,
        "formula_decoded": _decode_formula(formula),
        "checkpoint_path": str(ckpts[-1].relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "step": int(meta.get("step") or 0),
    }


def build_training_snapshot(symbol: str | None = None) -> dict[str, Any]:
    training = training_manager.status()
    job = training.get("job") or {}
    settings = load_settings()

    sym = (symbol or job.get("symbol") or "").strip()
    timeframe = str(job.get("timeframe") or "").strip().upper()

    data_file = settings.get("last_data_file") or ""
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            if not sym:
                sym = str(info.get("symbol") or "").strip()
            if not timeframe:
                timeframe = str(info.get("timeframe") or "").strip().upper()
        except Exception:
            pass

    if not sym:
        raise ValueError("请先选择训练数据文件或指定品种")
    if not timeframe:
        timeframe = "H1"

    progress = get_symbol_progress(sym, timeframe)
    if training.get("active"):
        live_step = training_manager.parse_step_from_log()
        if live_step is not None and live_step > progress.current_step:
            progress = progress.__class__(
                symbol=progress.symbol,
                train_steps=progress.train_steps,
                current_step=live_step,
                best_score=progress.best_score,
                best_formula=progress.best_formula,
                formula_decoded=progress.formula_decoded,
                has_strategy=progress.has_strategy,
                strategy_score=progress.strategy_score,
                checkpoint_path=progress.checkpoint_path,
                checkpoint_mtime=progress.checkpoint_mtime,
                history=progress.history,
            )

    history = progress.history or {}
    current_run_best = _current_run_best(sym, timeframe)
    curve = _training_curve(history, max_points=500)
    history_bests = history.get("best_score") or []
    current_run_best_score = max(history_bests) if history_bests else None

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "data_file": data_file or None,
        "training_active": bool(training.get("active")),
        "job_state": job.get("state"),
        "current_step": progress.current_step,
        "train_steps": progress.train_steps,
        "progress_pct": round(progress.progress_pct, 2),
        "status": progress.status,
        "best_score": progress.best_score,
        "strategy_score": progress.strategy_score,
        "has_strategy": progress.has_strategy,
        "formula": progress.best_formula,
        "formula_decoded": progress.formula_decoded,
        "saved_champion": {
            "score": progress.strategy_score,
            "formula": progress.best_formula if progress.has_strategy else None,
            "formula_decoded": progress.formula_decoded if progress.has_strategy else None,
        },
        "current_run_best": current_run_best,
        "current_run_best_score_from_history": current_run_best_score,
        "current_run_best_formula_note": (
            "history has a newer/higher best score, but the matching formula is not available until the next checkpoint"
            if current_run_best_score is not None
            and current_run_best is not None
            and float(current_run_best_score) > float(current_run_best["score"])
            else None
        ),
        "checkpoint_path": progress.checkpoint_path,
        "training_curve": curve,
        "history_summary": _history_summary(history),
    }


def analyze_training(
    *,
    provider: str,
    api_key: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    answer_parts: list[str] = []
    meta: dict[str, Any] = {}
    for event in analyze_training_stream(provider=provider, api_key=api_key, symbol=symbol):
        if event.get("type") == "meta":
            meta = event
        elif event.get("type") == "delta":
            answer_parts.append(event.get("text") or "")
        elif event.get("type") == "error":
            raise RuntimeError(event.get("message") or "分析失败")
        elif event.get("type") == "done":
            return {
                "ok": True,
                "provider": event.get("provider") or meta.get("provider"),
                "model": event.get("model") or meta.get("model"),
                "label": event.get("label") or meta.get("label"),
                "symbol": event.get("symbol") or meta.get("symbol"),
                "timeframe": event.get("timeframe") or meta.get("timeframe"),
                "snapshot": event.get("snapshot") or meta.get("snapshot"),
                "prior_count": event.get("prior_count", meta.get("prior_count", 0)),
                "answer": event.get("answer") or "".join(answer_parts),
            }
    raise RuntimeError("AI 流式分析未正常结束")


def analyze_training_stream(
    *,
    provider: str,
    api_key: str | None = None,
    symbol: str | None = None,
):
    """Yield SSE-ready event dicts: meta / delta / done / error."""
    from web.ai_providers import stream_chat_completions

    try:
        snapshot = build_training_snapshot(symbol)
        prior = load_prior_analyses(snapshot["symbol"], snapshot["timeframe"])
        resolved = resolve_provider(provider, api_key)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    parts = [
        "请根据以下训练快照回答：",
        "1. 当前训练情况怎么样？是否值得继续？",
        "2. 请分别解释 saved_champion（保存冠军公式）和 current_run_best（本轮最优公式）的含义与原理；如果两者相同，请说明它们目前一致。",
        "",
        "请特别注意：val_score 是当前步/当前批的平均验证分，不是冠军分；best_score 是本轮运行最大值，横盘不等于卡死；new_candidate_best_val_score 才更能说明模型是否在发现新的高分候选；batch_best_val_score 可能含精英回放；elite_replay_used 要结合 QD 优秀池的冷却/恢复机制理解。",
        "",
        "【当前训练快照】",
        f"```json\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n```",
    ]
    if prior:
        parts.extend(
            [
                "",
                f"【同品种同周期历史分析记录，共 {len(prior)} 次，按时间从旧到新】",
                "请对比这些历史记录，判断相对上次是否有改善。",
                f"```json\n{json.dumps(prior, ensure_ascii=False, indent=2)}\n```",
            ]
        )
    else:
        parts.append("\n（尚无同品种同周期的历史分析记录，这是首次分析。）")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

    yield {
        "type": "meta",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
    }

    answer_parts: list[str] = []
    try:
        for text in stream_chat_completions(resolved, messages):
            answer_parts.append(text)
            yield {"type": "delta", "text": text}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    answer = "".join(answer_parts).strip()
    if not answer:
        yield {"type": "error", "message": "AI 返回内容为空"}
        return

    record = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "provider": resolved.provider,
        "model": resolved.model,
        "snapshot": {
            "symbol": snapshot["symbol"],
            "timeframe": snapshot["timeframe"],
            "current_step": snapshot["current_step"],
            "train_steps": snapshot["train_steps"],
            "progress_pct": snapshot["progress_pct"],
            "best_score": snapshot["best_score"],
            "strategy_score": snapshot["strategy_score"],
            "formula_decoded": snapshot["formula_decoded"],
            "saved_champion": snapshot.get("saved_champion"),
            "current_run_best": snapshot.get("current_run_best"),
            "history_summary": snapshot.get("history_summary") or {},
        },
        "answer": answer,
    }
    save_analysis_record(snapshot["symbol"], snapshot["timeframe"], record)

    yield {
        "type": "done",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
        "answer": answer,
    }


def history_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.strip().upper()}|{str(timeframe).strip().upper()}"


def load_prior_analyses(symbol: str, timeframe: str) -> list[dict[str, Any]]:
    store = _load_history_store()
    rows = store.get(history_key(symbol, timeframe)) or []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows[-_MAX_HISTORY_PER_KEY:]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "analyzed_at": row.get("analyzed_at"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "snapshot": row.get("snapshot") or {},
                "answer": row.get("answer") or "",
            }
        )
    return out


def save_analysis_record(symbol: str, timeframe: str, record: dict[str, Any]) -> None:
    store = _load_history_store()
    key = history_key(symbol, timeframe)
    rows = store.get(key) or []
    if not isinstance(rows, list):
        rows = []
    rows.append(record)
    store[key] = rows[-_MAX_HISTORY_PER_KEY:]
    HISTORY_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_history_store() -> dict[str, Any]:
    if not HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _training_curve(history: dict[str, Any], max_points: int = 500) -> dict[str, Any]:
    if not history:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}
    steps = history.get("step") or []
    if not isinstance(steps, list) or not steps:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    total = len(steps)
    keys = (
        "step",
        "best_score",
        "val_score",
        "batch_best_val_score",
        "new_candidate_best_val_score",
        "entropy",
        "avg_reward",
        "stable_rank",
        "elite_replay_used",
        "incubation_replay_used",
        "incubation_pool_size",
        "incubation_archive_cells",
        "timing_total_ms",
        "timing_sample_elite_ms",
        "timing_eval_ms",
        "timing_grad_ms",
        "timing_rest_ms",
    )
    available = [key for key in keys if isinstance(history.get(key), list) and history.get(key)]

    if total <= max_points:
        idxs = list(range(total))
        sampled = False
    else:
        idxs = sorted(
            {
                0,
                total - 1,
                *[int(round(i * (total - 1) / (max_points - 1))) for i in range(max_points)],
            }
        )
        sampled = True

    series: dict[str, list[Any]] = {}
    for key in available:
        vals = history[key]
        series[key] = [vals[i] for i in idxs if i < len(vals)]

    return {
        "total_points": total,
        "sampled": sampled,
        "points": len(idxs),
        "note": (
            f"已从全部 {total} 个记录点均匀抽样为 {len(idxs)} 点，覆盖训练全程"
            if sampled
            else f"已发送全部 {total} 个记录点"
        ),
        "series": series,
    }


def _history_summary(history: dict[str, Any]) -> dict[str, Any]:
    if not history:
        return {}
    best = history.get("best_score") or []
    val = history.get("val_score") or []
    batch_best = history.get("batch_best_val_score") or []
    new_candidate_best = history.get("new_candidate_best_val_score") or []
    entropy = history.get("entropy") or []
    elite_replay = history.get("elite_replay_used") or []
    incubation_replay = history.get("incubation_replay_used") or []
    incubation_pool = history.get("incubation_pool_size") or []
    incubation_cells = history.get("incubation_archive_cells") or []
    timing_total = history.get("timing_total_ms") or []
    timing_eval = history.get("timing_eval_ms") or []
    timing_sample_elite = history.get("timing_sample_elite_ms") or []
    timing_grad = history.get("timing_grad_ms") or []
    steps = history.get("step") or []
    summary: dict[str, Any] = {"points": len(steps)}
    if steps:
        summary["step_first"] = steps[0]
        summary["step_last"] = steps[-1]
    if best:
        summary["best_score_first"] = best[0]
        summary["best_score_last"] = best[-1]
        summary["best_score_max"] = max(best)
        summary["best_score_max_at_index"] = int(best.index(max(best)))
        peak = max(best)
        trail = 0
        for value in reversed(best):
            if abs(float(value) - float(peak)) < 1e-9:
                trail += 1
            else:
                break
        summary["best_score_stagnation_points"] = trail
        n = len(best)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["best_score_phase_means"] = {
                "early": sum(best[:a]) / max(1, a),
                "mid": sum(best[a:b]) / max(1, b - a),
                "late": sum(best[b:]) / max(1, n - b),
            }
    if val:
        summary["val_score_first"] = val[0]
        summary["val_score_last"] = val[-1]
        summary["val_score_max"] = max(val)
        n = len(val)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["val_score_phase_means"] = {
                "early": sum(val[:a]) / max(1, a),
                "mid": sum(val[a:b]) / max(1, b - a),
                "late": sum(val[b:]) / max(1, n - b),
            }
    if entropy:
        summary["entropy_first"] = entropy[0]
        summary["entropy_last"] = entropy[-1]
        if len(entropy) >= 20:
            tail = entropy[-20:]
            summary["entropy_tail20_mean"] = sum(tail) / len(tail)
            summary["entropy_tail20_trend"] = tail[-1] - tail[0]
    if batch_best:
        summary["batch_best_val_score_last"] = batch_best[-1]
        summary["batch_best_val_score_max"] = max(batch_best)
        tail = batch_best[-50:]
        if tail:
            summary["batch_best_val_score_tail50_max"] = max(tail)
            summary["batch_best_val_score_tail50_mean"] = sum(tail) / len(tail)
    if new_candidate_best:
        summary["new_candidate_best_val_score_last"] = new_candidate_best[-1]
        summary["new_candidate_best_val_score_max"] = max(new_candidate_best)
        tail = new_candidate_best[-50:]
        if tail:
            summary["new_candidate_best_val_score_tail50_max"] = max(tail)
            summary["new_candidate_best_val_score_tail50_mean"] = sum(tail) / len(tail)
            if best:
                summary["new_candidate_tail50_gap_to_best"] = max(best) - max(tail)
    if elite_replay:
        summary["elite_replay_used_last"] = elite_replay[-1]
        tail = elite_replay[-50:]
        if tail:
            summary["elite_replay_used_tail50_min"] = min(tail)
            summary["elite_replay_used_tail50_max"] = max(tail)
            summary["elite_replay_used_tail50_mean"] = sum(tail) / len(tail)
    if incubation_replay:
        summary["incubation_replay_used_last"] = incubation_replay[-1]
        tail = incubation_replay[-50:]
        if tail:
            summary["incubation_replay_used_tail50_max"] = max(tail)
            summary["incubation_replay_used_tail50_mean"] = sum(tail) / len(tail)
    if incubation_pool:
        summary["incubation_pool_size_last"] = incubation_pool[-1]
        summary["incubation_pool_size_max"] = max(incubation_pool)
    if incubation_cells:
        summary["incubation_archive_cells_last"] = incubation_cells[-1]
        summary["incubation_archive_cells_max"] = max(incubation_cells)
    if timing_total:
        tail = timing_total[-50:]
        summary["timing_total_ms_last"] = timing_total[-1]
        summary["timing_total_ms_tail50_mean"] = sum(tail) / len(tail)
    if timing_eval:
        tail = timing_eval[-50:]
        summary["timing_eval_ms_last"] = timing_eval[-1]
        summary["timing_eval_ms_tail50_mean"] = sum(tail) / len(tail)
    if timing_sample_elite:
        tail = timing_sample_elite[-50:]
        summary["timing_sample_elite_ms_last"] = timing_sample_elite[-1]
        summary["timing_sample_elite_ms_tail50_mean"] = sum(tail) / len(tail)
    if timing_grad:
        tail = timing_grad[-50:]
        summary["timing_grad_ms_last"] = timing_grad[-1]
        summary["timing_grad_ms_tail50_mean"] = sum(tail) / len(tail)
    return summary
