## Purpose

定义使用 LeRobot v3 数据的 SO-101 双臂折布 π0.5 训练配方，包含学习率、quantile 统计、EMA 和原生验证划分。

## Requirements

### Requirement: 折布训练入口必须显式设置目标配方
双臂 SO-101 π0.5 折布训练入口 MUST 使用 LeRobot v3 版 `so101_bimanual_folding-cloth-720p` 数据，明确设置 seed 42、40,000 优化步、AdamW `(0.9, 0.95)`、`eps=1e-8`、weight decay `0.01`、梯度范数裁剪 1.0、1,000 步 warmup、峰值学习率 `4e-5`、40,000 步衰减到 `4e-6`，以及 EMA 常数衰减 `0.999`。默认每卡 batch MUST 为 6、使用两卡（全局 batch 12），checkpoint 保存间隔 MUST 为 20,000 步。本变更不得引入数据版本等价性检查。

#### Scenario: 使用默认折布配方
- **WHEN** 用户运行折布训练入口且未覆盖相关参数
- **THEN** 训练使用上述学习率、优化器、seed、EMA 和 v3 数据设置，默认每卡 batch 为 6、两卡全局 batch 为 12，每 20,000 步保存 checkpoint

### Requirement: 状态和动作使用 quantile 归一化
折布配方 MUST 按同一个原生 `eval_split` 从 LeRobot v3 的训练 episode 生成或加载状态/动作的 q01/q99 统计，并用这些统计启用 quantile 归一化；统计准备 MUST 作为首次训练前的独立步骤，直接写入本地 v3 数据集的 `meta/stats.json`；训练入口 MUST 读取已有统计，不得在每次启动时重算；统计过程不得上传数据集；补充 q01/q99 时 MUST 保留且不得覆盖既有 mean/std、min/max、count 等统计字段，使 MEAN_STD 与 QUANTILES 两种归一化仍可用。视觉图像保留既有范围处理。图像增强的开关与行为见 `pi05-openpi-image-augmentation`。

#### Scenario: v3 元数据缺少 quantile 统计
- **WHEN** v3 `meta/stats.json` 只有 min/max/mean/std
- **THEN** 系统基于训练 episode 生成或加载 q01/q99 后再启动 quantile 归一化，且不会向 Hub 写入；追加分位数时保留原有 mean/std 等字段，已有 mean/std 值不变

### Requirement: 验证划分和权重身份必须可复现
折布配方 MUST 使用 LeRobot 原生 `dataset.eval_split`，默认比例 `3/96`；训练集 MUST 排除原生划分选出的验证 episode；离线验证间隔为 500 步且不得增强。离线验证 loss MUST 标记为实时参数结果；EMA checkpoint/策略评估 MUST 单独标识。

#### Scenario: 两种归一化模式共存
- **WHEN** 用户在保留原统计字段的数据上分别选择 `MEAN_STD` 和 `QUANTILES`
- **THEN** 两种模式都能读取各自所需统计量并完成归一化

#### Scenario: 验证划分
- **WHEN** 用 LeRobot v3 折布数据启动训练
- **THEN** 原生划分从当前单任务 96 个 episode 中留出末尾 `[93, 94, 95]`，三者不进入训练采样，验证每 500 步运行

#### Scenario: EMA 评估
- **WHEN** 用户评估保存的 EMA 权重
- **THEN** 输出明确标记为 EMA，不与实时参数的离线验证 loss 混报
