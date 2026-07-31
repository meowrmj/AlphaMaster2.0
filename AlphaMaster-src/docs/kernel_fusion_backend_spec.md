# Formula VM Kernel Fusion 后端规格

目标：实现朋友所说的“按算子分类并合并 kernel 启动”，但不改变公式语义和评分结果。

## 不变的语义

以下内容不能被 kernel 优化改变：

- 输入公式 token 序列不变。
- StackVM 的栈语义不变。
- 每个算子的数学定义不变。
- NaN/Inf 处理规则不变。
- 最终因子归一化规则不变。
- reward、val_score、回测评分不变。

优化只允许改变“怎么批量执行”，不允许改变“算什么”。

## 第一版后端边界

第一版 fused 后端只接管公式 VM 的因子计算：

```text
formulas + features -> factors + valid_mask
```

不接管：

- Transformer 采样
- reward 计算
- backtest 评分
- 梯度更新
- 策略保存

这些继续走原逻辑。

## 输入输出

输入：

```text
formulas: [B, L] int64
features: [N, F, T] float32
plan: KernelExecutionPlan
```

输出：

```text
factors: [B, N, T] float32
valid: [B] bool
```

其中：

- B = 候选公式数，当前常见 192。
- L = 公式 token 长度，当前通常 8。
- N = 品种数量，单品种为 1，多品种大于 1。
- F = 特征数量。
- T = 历史 bar 数量。

## 第一批优先 kernel

根据当前真实 checkpoint，第一批最值得做的是：

```text
BRANCH:
  IF_GT
  GATE

ELEMENTWISE:
  ADD
  SUB
  MUL
  DIV
  TANH_SQUASH
  SIGNED_LOG
  SIGNED_POWER_2
  SQRT
  CLIP
  WINSORIZE

ROLLING:
  TS_ZSCORE_10
  TS_ZSCORE_20
  TS_MEAN_5/10/20
  TS_SUM_10/20
  TS_RANK_5/10/20
  EMA_5/20
  WMA
  TS_DECAY_EXP_5

SHIFT:
  DELTA
  MOMENTUM_5
  MOMENTUM_10
```

理由：

- `IF_GT/GATE` 在当前 D1 强化学习 checkpoint 中最高频。
- GA checkpoint 中 `TS_DECAY_EXP_5/DELTA/SIGNED_POWER_2/SIGNED_LOG` 高频。
- elementwise 算子容易先做数值对齐。
- rolling 算子是性能大头，但实现风险更高，应排在第二批。

## 责任链

执行顺序：

```text
fused kernel backend
  -> PyTorch batch VM fallback
  -> standard StackVM fallback
```

任何不支持的 bucket 必须 fallback。

任何 ScoreGuard 不通过的 batch 必须 fallback。

## 校验规则

每个 fused 后端必须和标准解释器对齐：

```text
factor 最大绝对误差 <= 1e-4
reward 最大绝对误差 <= 1e-4
val_score 最大绝对误差 <= 1e-4
status 必须一致
```

第一版上线前必须跑：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_kernel_planner.py --limit 4 --repeat 200 --batch-size 192
.\.venv\Scripts\python.exe scripts\analyze_kernel_coverage.py --limit 8
```

接入训练后必须保留 ScoreGuard。

## 为什么不能一次性全写

一次性实现所有算子会有三个风险：

- rolling / rank / zscore 类算子最容易产生细微数值差异。
- 单品种和多品种的 cross-sectional 行为不同。
- Windows 下 CUDA 扩展环境复杂，错误定位成本高。

所以正确顺序是：

1. 先做 elementwise + branch。
2. 用 ScoreGuard 对齐。
3. 再做 shift。
4. 再做 rolling。
5. 最后处理 cross-sectional。

每一批都必须能独立回退。
