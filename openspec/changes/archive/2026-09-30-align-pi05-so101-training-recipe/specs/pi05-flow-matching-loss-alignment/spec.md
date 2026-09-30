## ADDED Requirements

### Requirement: 折布配方可按 openpi 语义训练 padded action 维度
π0.5 MUST 提供 `train_on_padded_action_dims` 配置，通用默认值为 `false`。开启时，flow matching 训练 loss 和离线验证 loss MUST 在模型的全部 `max_action_dim` 维上计算逐元素 MSE 并取均值，不能先截断到数据集真实动作维数；双臂 SO-101 折布对齐脚本 MUST 开启此配置。关闭时 MUST 保持现有行为：只在真实动作维度上计算 loss。该选项不得改变动作推理输出的真实维度截断。

#### Scenario: 对齐 openpi 的 12 维 SO-101 动作
- **WHEN** 数据动作有 12 维、模型 `max_action_dim=32` 且 `train_on_padded_action_dims=true`
- **THEN** 训练和离线验证在补齐后的 32 维动作上计算 MSE 均值，20 个 padding 维的目标及梯度均参与优化

#### Scenario: 保持其他 π0.5 任务默认行为
- **WHEN** 用户未启用 `train_on_padded_action_dims` 或显式设为 `false`
- **THEN** loss 只覆盖真实动作维度，训练语义与现有版本一致

#### Scenario: 采样动作仍返回硬件维度
- **WHEN** 已启用 padded action loss 的策略执行动作推理
- **THEN** 策略仍仅返回数据/机器人定义的 12 个动作维度
