# Formula VM Kernel Fusion 后端规格

目标：实现“按算子分桶、用 native CUDA kernel 批量执行”的公式 VM 加速路线，同时保持公式语义、评分结果、训练流程不变。

## 不变的语义边界

native kernel 只允许改变“怎么算得更快”，不能改变“算什么”：

- 输入公式 token 序列不变。
- StackVM 栈语义不变。
- 每个算子的数学定义不变。
- NaN/Inf 处理规则不变。
- 因子归一化、reward、val_score、回测评分不变。
- Transformer 采样、梯度更新、策略保存逻辑不由 native kernel 接管。

## 后端职责

第一阶段只接管公式因子执行：

```text
formulas + features -> factors + valid_mask
```

不接管：

```text
Transformer 采样
reward / val_score 聚合
backtest 评分
梯度更新
checkpoint / champion 保存
```

## 执行结构

```text
公式 token
  -> IR Plan
  -> 按 step + op + arity 分桶
  -> native CUDA kernel 执行可支持桶
  -> PyTorch batch VM fallback 执行未支持桶
  -> ScoreGuard 对齐标准解释器
```

单品种和多品种使用同一张量结构：

```text
formulas: [B, L]
features: [N, F, T]
factors:  [B, N, T]
```

B 是候选公式数量，N 是品种数，T 是历史 bar 数。单品种时 N=1，多品种时 N>1。

## 当前 native 覆盖

已经实现并通过数值对齐的算子：

```text
一元:
NEG / ABS / SIGN / POWER / SIGNED_POWER_2 / SIGNED_LOG / SQRT / CLIP / SIGMOID / TANH_SQUASH

二元:
ADD / SUB / MUL / DIV / MAX / MIN
TS_CORR_10 / COVARIANCE_10

三元:
IF_GT / GATE

位移:
DELAY1 / DELAY4 / DELTA / DELTA_5

滚动:
TS_MEAN_5/10/20
TS_SUM_5/10/20
TS_ZSCORE_10/20
TS_STD_5/10/20
TS_RANK_5/10/20
TS_MIN_10/20
TS_MAX_10/20
TS_QUANTILE_10
TS_ARG_MAX_5 / TS_ARG_MIN_5
DECAY / WMA / DECAY_LINEAR_5 / TS_DECAY_EXP_5
EMA_5 / EMA_20
MOMENTUM_5 / MOMENTUM_10
MAX3
```

## 当前验证证据

工具链：

```text
CUDA Toolkit 12.8
Visual Studio Build Tools 2022
PyTorch 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

数值测试命令：

```powershell
cmd /c scripts\with_native_toolchain.bat .venv\Scripts\python.exe scripts\test_native_elementwise.py
```

结果：全部通过。新增 rolling/EMA/decay 类算子的最大误差约在 `1e-6` 以内，未超过 `1e-5` 测试阈值。

代表性速度：

```text
TS_RANK_20:     native 约 10.15x
EMA_20:         native 约 533.59x
TS_DECAY_EXP_5: native 约 3.56x
DECAY:          native 约 4.80x
MAX3:           native 约 2.50x
TS_CORR_10:     native 约 49.76x
COVARIANCE_10:  native 约 23.63x
```

注意：这是单算子速度，不等于完整训练 step 速度。完整 step 还包括采样、精英/孵化策略、打分聚合、梯度更新等环节。

已验证但暂不启用：

```text
SCALE: 数值正确，但当前 native 串行扫描实现比 PyTorch 慢。
JUMP: 数值正确，但当前 native 串行扫描实现比 PyTorch 慢。
PRODUCT_5: 当前 native 实现与 PyTorch 路径最大误差超过 1e-5，继续 fallback。
```

## 当前真实公式覆盖

计划器验证命令：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_kernel_planner.py --limit 4 --repeat 30 --batch-size 192
```

当前结果：

```text
formulas = 192
token_steps = 8
bucketed launches = 185
native executable launches = 163
fallback launches = 22
```

剩余 fallback 主要来自：

```text
WINSORIZE
CS_RANK
JUMP
CS_SCALE
SCALE
CS_NEUTRALIZE
TS_SKEW_10
PRODUCT_5
```

## 下一步优先级

要继续接近“整轮 500-800ms”，只靠单算子还不够。后续优先级应该是：

1. 扩展高频 fallback 算子，优先 `WINSORIZE / CS_RANK / CS_SCALE / CS_NEUTRALIZE / TS_SKEW_10`。
2. 做更粗粒度的 kernel fusion，减少公式 step 内多次 launch。
3. 优化 AB 阶段，也就是 Transformer 采样、旧方向/新方向候选生成、精英或孵化策略注入。
4. 保留 ScoreGuard，任何 native 快路径与标准路径不一致都必须 fallback。

## 上线原则

native 后端必须作为可选后端接入，不直接替换标准解释器：

```text
native fast path
  -> 数值一致：使用 native 结果
  -> 数值不一致或不支持：fallback 到 PyTorch batch VM
  -> 仍异常：fallback 到标准 StackVM
```

这样可以继续榨 GPU 性能，但不牺牲评分正确性。
