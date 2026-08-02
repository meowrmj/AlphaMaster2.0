"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import math
import os

import torch
from .vocab import FORMULA_VOCAB


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _select_device() -> torch.device:
    mode = os.getenv("ALPHAMASTER_DEVICE", "cpu").strip().lower()
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class ModelConfig:
    ALGORITHM_MODE: str = os.getenv("ALPHAMASTER_ALGORITHM_MODE", "rl").strip().lower()

    # ── 训练设备 ─────────────────────────────────────────────────────────
    # 注意：本任务 CPU 训练速度反而比 GPU 快（实测约 2.3 倍），故强制用 CPU。
    # 原因：
    #   1. 张量太小——forex 组仅 (2 品种 × 3508 × 20 特征)，单个算子的
    #      计算量小于 CUDA kernel 启动开销（数十微秒），GPU 算得快但启动慢。
    #   2. 训练循环是 Python 串行调度：每 step 逐条跑 128 条公式 × 8 个
    #      VM 步 × 4 个 walk-forward 折，GPU 被切成上万个碎片段，吃不满。
    #   3. host↔device 拷贝 + kernel 启动延迟主导总耗时，而非张量计算本身。
    #   4. 实测 GPU 利用率 ~51%，正是 GPU 一半时间在干等 Python 喂下一个
    #      kernel 的证据（不是“还能压榨”，而是“调度瓶颈”）。
    # 基准测试（forex 组, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若后续改为批量并行公式评估（一次喂大批张量进 GPU），再切回 cuda。
    DEVICE_MODE = os.getenv("ALPHAMASTER_DEVICE", "cpu").strip().lower()
    DEVICE = _select_device()
    GPU_BATCH_EVAL: bool = _env_bool("ALPHAMASTER_GPU_BATCH_EVAL", False)
    GPU_BATCH_EVAL_STRICT: bool = _env_bool("ALPHAMASTER_GPU_BATCH_EVAL_STRICT", True)
    EVALUATOR_ENGINE: str = os.getenv("ALPHAMASTER_EVALUATOR_ENGINE", "auto").strip().lower()
    EVALUATOR_GUARD: bool = _env_bool("ALPHAMASTER_EVALUATOR_GUARD", True)
    EVALUATOR_GUARD_EVERY: int = int(os.getenv("ALPHAMASTER_EVALUATOR_GUARD_EVERY", "1"))
    EVALUATOR_GUARD_SAMPLE: int = int(os.getenv("ALPHAMASTER_EVALUATOR_GUARD_SAMPLE", "8"))
    EVALUATOR_SCORE_TOL: float = float(os.getenv("ALPHAMASTER_EVALUATOR_SCORE_TOL", "3e-3"))
    EVALUATOR_FACTOR_TOL: float = float(os.getenv("ALPHAMASTER_EVALUATOR_FACTOR_TOL", "2.5e-1"))

    # ── 训练参数（大搜索空间适配版，2026-07-04 重构）─────────────────────
    # 背景：特征库扩展到 65、算子库扩展到 66（vocab=131），8-token 搜索空间
    #   从旧版 ~7亿 暴增到 ~8.67×10^16（1.2 亿倍）。旧的采样预算（128×3000）
    #   覆盖率趋近于零，导致熵坍塌 Early Stop、公式退化。
    # 对策（训练时间不敏感场景）：
    #   1. 特征剪枝（active_features.json）把 vocab 降到 ~90，空间缩小约 20 倍
    #   2. 放大采样预算：BATCH_SIZE 128→256，TRAIN_STEPS 3000→8000
    #   3. 更大精英池（60）保留更多历史最优
    BATCH_SIZE      = 192   # 每步采样公式数（原 128，1.5x 提升覆盖率）
    TRAIN_STEPS     = 9000  # 每组训练步数（55次重启需要更多步数）
    MAX_FORMULA_LEN = 8     # 公式长度上限：保持 8（10 会导致 CPU 训练慢 3 倍）

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控 ──────────────────────────────────
    # IC_NEG_MULT 0.30→0.50：0.30 对反向因子惩罚过重，可能误杀非线性高收益因子。
    # 收益优先模式下，只要年化收益是正的，适当负 IC 可以接受。
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.01
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.75   # 收益优先：不过度误杀反向/非线性高收益因子

    # ── FTMO 专属奖励模式 ─────────────────────────────────────────────
    # "standard": 收益+风险平衡（默认，原权重）
    # "ftmo":     FTMO 考试盘专属——年化收益权重 0.60→0.75，Calmar 0.05→0.10
    #             （控制 MDD 贴近 10% Max Loss 上限），其余指标权重下调。
    #             目标：在 10% Max Loss 约束下最大化年化收益，快速达标。
    # "forex":    外汇均值回归专属（2026-07-08）——
    #             降年化收益权重(0.80→0.25)、提IC权重(0.03→0.25)、
    #             新增反转奖励(0.20，奖励低/负因子自相关)和多空对称检查(0.15)。
    #             原因：外汇H1以震荡为主，趋势算子效果差，需引导模型偏好
    #             均值回归信号而非追涨杀跌。
    REWARD_MODE:       str = "ftmo"

    # ── 熵保护（大空间加强版）──────────────────────────────────────────
    # ENTROPY_COEFF_MAX 0.5→1.0：加倍探索压力，对抗大 vocab 的过早收敛。
    # ENTROPY_COLLAPSE_THRESH 改为相对阈值 0.15×ln(vocab)：大 vocab 最大熵更高
    #   （ln(131)≈4.87 vs ln(54)≈3.99），绝对阈值 0.5 不再合理。
    # ENTROPY_COLLAPSE_STEPS 15→40：给模型更长的自我恢复窗口，不急于重启。
    ENTROPY_COEFF_MAX:   float = 1.0
    ENTROPY_COEFF_POWER: float = 1.0  # 降低幂次，让低熵时系数更激进（原1.3）
    ENTROPY_COLLAPSE_THRESH: float = 0.15 * math.log(FORMULA_VOCAB.size)
    ENTROPY_COLLAPSE_STEPS:  int   = 20  # 更快检测坍塌并重启

    # ── 熵下限惩罚（Fix 1: H→0 时熵项归零问题）──────────────────────────
    # 当 H < ENTROPY_FLOOR_THRESH 时，加入固定惩罚 λ×(thresh-H)。
    # 这确保即使 mean_ent→0，loss 中仍有非零探索压力。
    ENTROPY_FLOOR:        bool  = True
    ENTROPY_FLOOR_THRESH: float = 1.0   # 熵低于此值时触发固定惩罚（提高介入时机）
    ENTROPY_FLOOR_LAMBDA: float = 5.0   # 惩罚强度系数（加大力度对抗坍塌）

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25
    REPLAY_POLICY:      str   = os.getenv("ALPHAMASTER_REPLAY_POLICY", "qd_incubation")
    ELITE_POOL_SIZE:    int   = 60    # 30→60：大空间需要更大的精英记忆
    ELITE_REWARD_SCALE: float = 0.4
    ELITE_BUCKET_CAP:   int   = 3
    ELITE_CORE_CAP:     int   = int(os.getenv("ALPHAMASTER_ELITE_CORE_CAP", "8"))
    ELITE_START_TOKEN_CAP: int = int(os.getenv("ALPHAMASTER_ELITE_START_TOKEN_CAP", "12"))
    ELITE_REPLAY_COOLDOWN_STEPS: int = 80
    ELITE_REPLAY_RECOVERY_STEPS: int = 120

    # 重启后的“新方向孵化池”：保护冷却期产生的新公式，避免还没成熟就被历史高分精英挤掉。
    INCUBATION_POOL_SIZE: int = 36
    INCUBATION_BUCKET_CAP: int = 2
    INCUBATION_CORE_CAP: int = int(os.getenv("ALPHAMASTER_INCUBATION_CORE_CAP", "6"))
    INCUBATION_START_TOKEN_CAP: int = int(os.getenv("ALPHAMASTER_INCUBATION_START_TOKEN_CAP", "8"))
    INCUBATION_CAPTURE_STEPS: int = 180
    INCUBATION_REPLAY_STEPS: int = 260
    INCUBATION_REPLAY_FRAC: float = 0.08
    INCUBATION_MIN_SCORE: float = -0.5

    # Optional search plugins. They propose extra formulas for evaluation, but
    # V1 keeps them out of REINFORCE gradients so they do not pull the generator.
    SEARCH_PLUGIN_FRAC: float = float(os.getenv("ALPHAMASTER_SEARCH_PLUGIN_FRAC", "0.20"))
    SEARCH_ARCHIVE_SIZE: int = int(os.getenv("ALPHAMASTER_SEARCH_ARCHIVE_SIZE", "96"))
    SEARCH_BUCKET_CAP: int = int(os.getenv("ALPHAMASTER_SEARCH_BUCKET_CAP", "4"))
    SEARCH_CORE_CAP: int = int(os.getenv("ALPHAMASTER_SEARCH_CORE_CAP", "12"))
    SEARCH_START_TOKEN_CAP: int = int(os.getenv("ALPHAMASTER_SEARCH_START_TOKEN_CAP", "16"))
    ANNEAL_TEMP: float = float(os.getenv("ALPHAMASTER_ANNEAL_TEMP", "0.35"))
    ANNEAL_TEMP_MIN: float = float(os.getenv("ALPHAMASTER_ANNEAL_TEMP_MIN", "0.03"))
    ANNEAL_DECAY: float = float(os.getenv("ALPHAMASTER_ANNEAL_DECAY", "0.997"))
    GA_MUTATION_RATE: float = float(os.getenv("ALPHAMASTER_GA_MUTATION_RATE", "0.45"))
    GA_TOURNAMENT_K: int = int(os.getenv("ALPHAMASTER_GA_TOURNAMENT_K", "4"))
    GA_PARENT_CORE_CAP: int = int(os.getenv("ALPHAMASTER_GA_PARENT_CORE_CAP", "4"))
    GA_PARENT_START_TOKEN_CAP: int = int(os.getenv("ALPHAMASTER_GA_PARENT_START_TOKEN_CAP", "8"))
    GA_CROSS_NICHE_RATE: float = float(os.getenv("ALPHAMASTER_GA_CROSS_NICHE_RATE", "0.85"))
    GA_CHILD_SIMILARITY_MAX: float = float(os.getenv("ALPHAMASTER_GA_CHILD_SIMILARITY_MAX", "0.82"))
    GA_RANDOM_IMMIGRANT_FRAC: float = float(os.getenv("ALPHAMASTER_GA_RANDOM_IMMIGRANT_FRAC", "0.25"))
    GA_POPULATION_SIZE: int = int(os.getenv("ALPHAMASTER_GA_POPULATION_SIZE", "384"))
    GA_ELITE_FRAC: float = float(os.getenv("ALPHAMASTER_GA_ELITE_FRAC", "0.06"))
    GA_RANDOM_INJECT_FRAC: float = float(os.getenv("ALPHAMASTER_GA_RANDOM_INJECT_FRAC", "0.08"))
    GA_CROSSOVER_RATE: float = float(os.getenv("ALPHAMASTER_GA_CROSSOVER_RATE", "0.70"))

    # ── 坍塌重启（大空间加强版）─────────────────────────────────────────
    # MAX_RESTARTS 8→25→55、RESTART_NOISE 0.05→0.1→0.25：时间不敏感，多给机会+更强扰动。
    # 配合 engine.py：超过 MAX_RESTARTS 后不再 Early Stop，改为强扰动继续训练。
    # 2026-07-09: US100 训练 24/25 重启仍有突破，扩到 55 次。
    MAX_RESTARTS:   int   = 55
    RESTART_NOISE:  float = 0.25

    # ── 自适应噪声：Best 停滞时自动增大扰动 ─────────────────────────────
    # stagnation_window: 判断停滞的步数窗口
    # noise_min / noise_max: 噪声下界和上界
    # noise_boost: 停滞时噪声提升倍率
    ADAPTIVE_NOISE:      bool  = True
    STAGNATION_WINDOW:   int   = 500
    NOISE_MIN:           float = 0.15
    NOISE_MAX:           float = 0.60
    NOISE_BOOST_FACTOR:  float = 2.0   # noise += 0.2 * (stagnation / window)

    # ── 重启多样性（Fix 2: best_snapshot 吸引子效应）─────────────────────
    # 每 FULL_RESET_EVERY 次重启中，做 1 次完全随机初始化而非从 best_snapshot 恢复。
    FULL_RESET_EVERY:    int   = 3     # 每 3 次重启中第 3 次做 full reset

    # ── Reward baseline（Fix 3: 全负 batch 相对优选问题）──────────────────
    # 用 EMA baseline 替代 batch mean 计算 advantage，避免全负 batch 的问题。
    REWARD_EMA_BASELINE:     bool  = True
    REWARD_EMA_DECAY:        float = 0.95   # EMA 衰减系数
    REWARD_EMA_WARMUP:       int   = 10     # 前 N 步用 batch mean（EMA 未稳定）

    # ── 重启时部分重置参数：保留底层，扰动顶层 ───────────────────────────
    PARTIAL_RESET:       bool  = True
    PARTIAL_RESET_LAYERS: tuple = ("ln_f", "mtp_head", "head_critic", "blocks", "token_emb")

    # ── Elite Replay 衰减：旧 elite 采样权重随时间衰减 ──────────────────
    ELITE_DECAY:         bool  = True
    ELITE_DECAY_HALF_LIFE: int = 300   # 每 300 步旧 elite 权重减半

    # ── 多起点并行（Island）──────────────────────────────────────────────
    # 注意：Island 模式在 CPU 训练下会让总时间变成 N 倍（islands 串行），
    # 对于 index 这类大数组（T=32076）会变得极慢。当前默认关闭，保留配置开关。
    N_ISLANDS:              int   = 1
    MIGRATION_INTERVAL:     int   = 500
    MIGRATION_TOP_K:        int   = 5
    # island 默认关闭，避免用户误开导致速度爆炸

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 25
    CORR_THRESHOLD:   float = 0.85
    CORR_PENALTY:     float = 0.8

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    # P2-6 修复说明：gap 必须按 target_horizon 标定，且与 CPCV purge gap 统一。
    # 当前 target_horizon=2（data_manager 用 log(open[t+2]/open[t+1]) 作 target_ret），
    # gap=20 相当于 10 个「真实预测步」——对 H1 数据足够，但若切换到 D1/15min
    # 需要重新评估。CPCV 模式（未实现）的 purge gap 应等于 WF_GAP，避免两套阈值。
    # 调整时同时检查：
    #   1. engine.py _build_walk_forward_folds 的 val_start = train_end + gap
    #   2. fold_size 应 >> gap（建议 fold_size >= 5*gap）以保证 val 有效性
    WF_GAP: int = 20

    # ── 公式结构约束（2026-07-05 新增）──────────────────────────────────
    # 背景：index 组因子因 TS_RANK 连续使用退化为 beta 因子（91.8% 做多），
    # 前半段市场跌亏钱、后半段市场涨赚钱，不是 alpha 而是 beta。
    # 对策：在采样阶段禁止恒正算子链，在评分阶段添加 beta 中性 + 前后一致性奖惩。
    ENABLE_FORMULA_STRUCTURE_CONSTRAINT: bool = True   # 总开关
    BETA_NEUTRAL_PENALTY:     bool  = True             # 多空比例失衡惩罚
    HALF_CONSISTENCY_BONUS:   bool  = True             # 前后一致性奖惩
    BETA_NEUTRAL_THRESH:      float = 0.85             # 超过此比例同方向触发重罚
    BETA_NEUTRAL_LIGHT_THRESH: float = 0.70            # 轻度失衡阈值

    # ── 并行评估（CPU 利用率优化）────────────────────────────────────────
    # 将 Part C 公式评估（VM 执行 + WF 折叠评估 + IC + 惩罚）并行化到
    # ThreadPoolExecutor（PyTorch CPU 算子释放 GIL，多线程真正并行）。
    # workers × intra_threads ≈ 物理核数，避免超线程过度订阅。
    #   EVAL_WORKERS=0: 自动 = 物理核数（cap 在 8）
    #   EVAL_INTRA_THREADS=0: 自动 = 物理核数（不分割，让每个 worker 的 torch 跑满）
    PARALLEL_EVAL:        bool = True
    EVAL_WORKERS:         int  = 0    # 0=auto (physical cores, cap 8)
    EVAL_INTRA_THREADS:   int  = 0    # 0=auto (physical cores)
