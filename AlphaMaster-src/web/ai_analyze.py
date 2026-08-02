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
- 必须全程使用简体中文，只允许公式 token、字段名、文件名保留英文原样；不要写英文段落、英文小标题或英文解释。
- 只输出最终分析，不要输出“我需要先分析”“用户想问的是”“首先我思考”等推理过程或草稿。
- 第一行必须直接写“## 1. 当前训练情况怎么样？是否值得继续”，前面不要有任何寒暄、摘要、解释或过渡句。
- 说人话，少用术语；必须让用户能看懂当前系统到底在做什么。
- 不要编造快照里没有的数据。
- 结论必须先分清“训练流程是否正常”和“策略质量是否值得信任”，不要混成一句话。
- 评估是否值得继续训练时，必须同时看 val_score、best_score、batch_best_val_score、new_candidate_best_val_score、entropy、elite_replay_used、timing_*。不要只看单个指标。
- val_score 是当前批次/训练步的平均验证表现，不等于冠军策略分数；它可以为负，但同一批里仍可能出现高分候选。
- best_score 是本轮运行最大值/冠军线，天然会长时间横盘；横盘不等于程序卡死。判断是否停滞，要结合 new_candidate_best_val_score 是否还在接近或冲击 best_score。
- batch_best_val_score 是本批最高分，可能包含精英回放；new_candidate_best_val_score 是本步不含精英回放的新候选最高分，更适合判断模型是否真的在发现新东西。
- entropy 上升通常表示重新探索/分布变松，不要简单说成“瞎探索”；entropy 很低且多样性下降才更像坍缩。
- elite_replay_used 表示本步注入了多少条优秀旧公式。当前系统使用 QD 优秀池 + 冷却恢复机制：重启后先降低精英回放，再逐步恢复，目的是减少旧冠军把模型拉回同一方向。
- incubation_pool / incubation_replay_used 是“新方向孵化池”：重启后的新候选如果结构有差异，即使暂时弱于历史冠军，也会被短期保护和少量回放，用来验证新方向是否能成长。
- search_config.modules.genetic=true 表示遗传搜索增强已开启；search_plugin_used 或日志里的“搜索=N”表示本步实际造了多少条搜索候选。开启但本步为 0 也可能发生，要按快照判断。
- search_config.modules.annealing=true 表示退火搜索增强已开启；它基于当前公式做小扰动，接受一部分弱但有潜力的邻域候选，不直接改模型权重。
- 遗传搜索增强不是旧版“精英回放策略”。新版遗传只在搜索增强层造候选：从 QD 优秀池/搜索档案里选父代，做树结构交叉、子树变异、合法性修复，再统一进入评估。只有评估后真正优秀的公式，才会自然进入 QD、孵化池或冠军。
- 当前已经实现的是“结构型 QD 优秀池”和“搜索增强插件”；更完整的语义行为 QD 分桶仍是蓝图，不要说成已经完全实现。
- 回放层和搜索增强层要分开解释：回放层把已有公式放进本批训练；搜索增强层额外造候选公式；Transformer 采样层按模型概率生成新公式；三者最后合并成同一批公式统一评估。
- 不要因为保存冠军分高、本轮候选分低，就直接说训练坏了；重新训练时本轮分数低于历史保存冠军是正常的。
- timing_total_ms / timing_eval_ms / timing_sample_elite_ms / timing_grad_ms 用来判断性能瓶颈；如果存在这些字段，说明训练正在记录拆分耗时。
- “saved_champion” 是当前保存下来的冠军策略，可能来自更早训练。
- “current_run_best” 是本轮训练最新 checkpoint 里的本轮最优策略。
- 第二部分必须分别解释 saved_champion 和 current_run_best 的公式含义；如果两者相同，要明确说它们一致。
- 公式是栈式 VM 公式，不一定能按“第一步输入第二步”机械解释成线性因果链；解释时可以说大体看哪些量价/波动/方向信息，但不要武断断言一定做多或做空，最终方向要以信号输出和回测为准。

必须使用这些小标题：

## 1. 当前训练情况怎么样？是否值得继续
说明进度、训练是否活着、当前模式、平均验证分、本批最高分、新候选最高分、本轮最优分、熵、精英回放和耗时瓶颈。先判断流程是否正常，再判断策略质量是否可靠，并给出继续训练、观察到某个步数、或先回测的建议。

## 2. 插件有没有生效？现在到底是谁在影响训练
分别说明回放策略、搜索增强、遗传、退火、评估加速模式的状态。必须解释“开启”和“本步实际产生候选数”不是同一个概念。

## 3. 因子的含义与原理
分别解释：
- 保存冠军公式 saved_champion
- 本轮最优公式 current_run_best
解释每条公式大概用了哪些信息，以及可能反映的市场状态。必须提醒：这是栈式公式的近似解释，交易方向和有效性要以回测结果为准。

## 4. 下一步建议
给出具体、克制的下一步建议。不要保证收益，不要把训练分数直接等同于实盘能力。
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
    curve = _training_curve(history, max_points=240)
    history_bests = history.get("best_score") or []
    current_run_best_score = max(history_bests) if history_bests else None
    system_context = _system_context(job)

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
        "system_context": system_context,
        "training_curve": curve,
        "history_summary": _history_summary(history),
    }


def _system_context(job: dict[str, Any]) -> dict[str, Any]:
    replay_config = job.get("replay_config") or {}
    search_config = job.get("search_config") or {}
    replay_modules = replay_config.get("modules") or {}
    search_modules = search_config.get("modules") or {}
    return {
        "architecture_version": "rl_replay_search_split",
        "algorithm_mode": job.get("algorithm_mode") or "rl",
        "eval_mode": job.get("eval_mode"),
        "replay_policy": job.get("replay_policy"),
        "replay_config": replay_config,
        "search_config": search_config,
        "implemented_replay_modules": {
            "qd": bool(replay_modules.get("qd")),
            "incubation": bool(replay_modules.get("incubation")),
        },
        "implemented_search_modules": {
            "annealing": bool(search_modules.get("annealing")),
            "genetic": bool(search_modules.get("genetic")),
        },
        "module_contract": {
            "replay_layer": "只回放已有公式进入本批训练，不直接产生新公式。",
            "search_layer": "额外制造候选公式，统一评估后才可能进入优秀池、孵化池或冠军。",
            "transformer_layer": "按当前策略分布逐 token 采样新公式，并用策略梯度更新模型。",
            "evaluation_layer": "所有来源的公式使用同一套 VM/回测/评分流程，不能跨评分体系比较。",
        },
        "genetic_contract": {
            "role": "搜索增强，不属于精英回放策略。",
            "parents": "从 QD 优秀池或搜索档案选择父代。",
            "operators": "树结构交叉、子树变异、约束修复。",
            "effect_on_gradient": "先造候选并统一评估；高分候选进入本批奖励后，才间接影响策略梯度。",
        },
        "qd_status": "当前是结构/行为混合 QD 优秀池：已加入起始特征家族、算子家族、复杂度和核心 token 多样性约束；完整的 IC 稳定性/暴露强弱语义分桶仍是蓝图，不应描述为已完全实现。",
        "removed_modules": ["旧版 elite_genetic 回放插件"],
        "blueprint_file": "docs/training_algorithm_blueprint.md",
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
        "必须用简体中文回答，只输出最终分析，不要输出推理过程、草稿或英文解释。",
        "第一行必须直接写：## 1. 当前训练情况怎么样？是否值得继续",
        "请根据以下训练快照回答：",
        "1. 当前训练情况怎么样？是否值得继续？",
        "2. 当前插件是否真正生效？请区分回放策略、搜索增强、遗传、退火、评估加速模式。",
        "3. 请分别解释 saved_champion（保存冠军公式）和 current_run_best（本轮最优公式）的含义与原理；如果两者相同，请说明它们目前一致。",
        "",
        "请特别注意：val_score 是当前步/当前批的平均验证分，不是冠军分；best_score 是本轮运行最大值，横盘不等于卡死；new_candidate_best_val_score 才更能说明模型是否在发现新的高分候选；batch_best_val_score 可能含精英回放；search_plugin_used 才能说明本步搜索增强实际用了多少候选；elite_replay_used 要结合 QD 优秀池的冷却/恢复机制理解。",
        "",
        "【系统结构说明】",
        "当前训练架构已经拆成三层：回放层负责把已有优秀公式放入本批；搜索增强层负责额外制造候选公式；Transformer 采样层负责按模型概率生成新公式。三路公式合并后统一评估，只有评估后的优秀公式才会进入 QD、孵化池或冠军。遗传属于搜索增强，不是旧版精英回放策略。",
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
        for text in stream_chat_completions(resolved, messages, max_tokens=8192):
            answer_parts.append(text)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    answer = _clean_final_answer("".join(answer_parts))
    if not answer:
        yield {"type": "error", "message": "AI 返回内容为空"}
        return
    yield {"type": "delta", "text": answer}

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
                "answer": _prior_answer_excerpt(row.get("answer") or ""),
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


def _clean_final_answer(text: str) -> str:
    answer = (text or "").strip()
    if not answer:
        return ""
    markers = (
        "## 1.",
        "## 一",
        "## 当前训练情况",
        "# 1.",
    )
    starts = [answer.find(marker) for marker in markers if answer.find(marker) >= 0]
    if starts:
        answer = answer[min(starts):].strip()
    return _drop_obvious_reasoning_prefix(answer).strip()


def _prior_answer_excerpt(text: str, limit: int = 1400) -> str:
    raw = text or ""
    if not any(marker in raw for marker in ("## 1.", "## 一", "## 当前训练情况", "# 1.")):
        return "（旧分析正文格式不可靠，已忽略；请以上方历史快照字段为准。）"
    answer = _clean_final_answer(text)
    if len(answer) <= limit:
        return answer
    return answer[:limit].rstrip() + "\n（历史分析过长，已截断。）"


def _drop_obvious_reasoning_prefix(text: str) -> str:
    lines = (text or "").splitlines()
    if not lines:
        return ""
    drop_prefixes = (
        "用户",
        "我们被要求",
        "我需要",
        "先整理",
        "首先",
        "Let me",
        "I need",
        "We need",
        "The user",
        "好的，用户",
    )
    cut = 0
    for idx, line in enumerate(lines[:30]):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("##", "#")):
            cut = idx
            break
        if any(stripped.startswith(prefix) for prefix in drop_prefixes):
            cut = idx + 1
            continue
        if cut:
            cut = idx
            break
    if cut:
        return "\n".join(lines[cut:]).strip()
    return text


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
        "search_plugin_used",
        "search_plugin_modules",
        "search_archive_size",
        "search_archive_cells",
        "anneal_accept_rate",
        "genetic_planned",
        "genetic_produced",
        "genetic_parent_count",
        "genetic_parent_source",
        "timing_total_ms",
        "timing_sample_elite_ms",
        "timing_eval_ms",
        "timing_grad_ms",
        "timing_rest_ms",
        "timing_replay_plan_ms",
        "timing_search_plugin_ms",
        "timing_ab_forward_ms",
        "timing_policy_sample_ms",
        "timing_memory_logprob_ms",
        "timing_loss_build_ms",
        "timing_backward_ms",
        "timing_optimizer_ms",
        "timing_dist_stats_ms",
        "eval_engine",
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
    search_used = history.get("search_plugin_used") or []
    search_modules = history.get("search_plugin_modules") or []
    search_archive = history.get("search_archive_size") or []
    search_cells = history.get("search_archive_cells") or []
    anneal_accept = history.get("anneal_accept_rate") or []
    genetic_planned = history.get("genetic_planned") or []
    genetic_produced = history.get("genetic_produced") or []
    genetic_parent_count = history.get("genetic_parent_count") or []
    genetic_parent_source = history.get("genetic_parent_source") or []
    timing_total = history.get("timing_total_ms") or []
    timing_eval = history.get("timing_eval_ms") or []
    timing_sample_elite = history.get("timing_sample_elite_ms") or []
    timing_grad = history.get("timing_grad_ms") or []
    timing_replay_plan = history.get("timing_replay_plan_ms") or []
    timing_search_plugin = history.get("timing_search_plugin_ms") or []
    timing_ab_forward = history.get("timing_ab_forward_ms") or []
    timing_policy_sample = history.get("timing_policy_sample_ms") or []
    timing_memory_logprob = history.get("timing_memory_logprob_ms") or []
    timing_loss_build = history.get("timing_loss_build_ms") or []
    timing_backward = history.get("timing_backward_ms") or []
    timing_optimizer = history.get("timing_optimizer_ms") or []
    timing_dist_stats = history.get("timing_dist_stats_ms") or []
    eval_engine = history.get("eval_engine") or []
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
    if search_used:
        summary["search_plugin_used_last"] = search_used[-1]
        tail = search_used[-50:]
        if tail:
            summary["search_plugin_used_tail50_min"] = min(tail)
            summary["search_plugin_used_tail50_max"] = max(tail)
            summary["search_plugin_used_tail50_mean"] = sum(tail) / len(tail)
    if search_modules:
        summary["search_plugin_modules_last"] = search_modules[-1]
    if search_archive:
        summary["search_archive_size_last"] = search_archive[-1]
        summary["search_archive_size_max"] = max(search_archive)
    if search_cells:
        summary["search_archive_cells_last"] = search_cells[-1]
        summary["search_archive_cells_max"] = max(search_cells)
    if anneal_accept:
        summary["anneal_accept_rate_last"] = anneal_accept[-1]
    if genetic_planned:
        summary["genetic_planned_last"] = genetic_planned[-1]
        tail = genetic_planned[-50:]
        if tail:
            summary["genetic_planned_tail50_mean"] = sum(tail) / len(tail)
    if genetic_produced:
        summary["genetic_produced_last"] = genetic_produced[-1]
        tail = genetic_produced[-50:]
        if tail:
            summary["genetic_produced_tail50_sum"] = sum(tail)
            summary["genetic_produced_tail50_mean"] = sum(tail) / len(tail)
            summary["genetic_produced_tail50_max"] = max(tail)
    if genetic_parent_count:
        summary["genetic_parent_count_last"] = genetic_parent_count[-1]
        tail = genetic_parent_count[-50:]
        if tail:
            summary["genetic_parent_count_tail50_mean"] = sum(tail) / len(tail)
    if genetic_parent_source:
        summary["genetic_parent_source_last"] = genetic_parent_source[-1]
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
    for key, values in (
        ("timing_replay_plan_ms", timing_replay_plan),
        ("timing_search_plugin_ms", timing_search_plugin),
        ("timing_ab_forward_ms", timing_ab_forward),
        ("timing_policy_sample_ms", timing_policy_sample),
        ("timing_memory_logprob_ms", timing_memory_logprob),
        ("timing_loss_build_ms", timing_loss_build),
        ("timing_backward_ms", timing_backward),
        ("timing_optimizer_ms", timing_optimizer),
        ("timing_dist_stats_ms", timing_dist_stats),
    ):
        if values:
            tail = values[-50:]
            summary[f"{key}_last"] = values[-1]
            summary[f"{key}_tail50_mean"] = sum(tail) / len(tail)
    if eval_engine:
        summary["eval_engine_last"] = eval_engine[-1]
    return summary
