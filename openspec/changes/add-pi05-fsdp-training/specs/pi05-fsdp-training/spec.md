## ADDED Requirements

### Requirement: PI05 全参训练具有正确的 FSDP 分片边界

系统 MUST 在启用 FSDP2 时将同层 VLM 与 action expert 参数组成独立联合单元，保持单一注册归属和参数对象身份；普通训练结构 SHALL 不受影响。

#### Scenario: 联合前向与梯度重算
- **WHEN** 使用相同输入、噪声和 timestep 分别执行原结构及 FSDP 准备后的结构
- **THEN** loss、梯度与更新结果在对应精度容差内一致，重算使用转换后的输入精度及已重新收集的参数

#### Scenario: Prefix 与 suffix 推理
- **WHEN** FSDP 准备后的模型分别处理 prefix 与使用 KV cache 的 suffix
- **THEN** 两条路径均经过联合层，结果与原结构一致

### Requirement: 权重导出与恢复兼容

系统 MUST 导出使用现有 PI05 参数名称的完整权重，并支持内部 DCP 状态保存恢复。

#### Scenario: 导出到普通模型
- **WHEN** 导出已准备分片结构的完整权重
- **THEN** 普通 PI05 可严格加载全部权重，参数键集合与原结构一致且无重复副本

#### Scenario: 分片恢复
- **WHEN** 双卡训练保存后重新构建模型与优化器并恢复
- **THEN** 模型、Adam 状态和后续一步更新与连续训练一致

### Requirement: 配置边界与可复现启动

系统 SHALL 提供显式 FSDP 启动选项，并 MUST 拒绝尚未支持的组合。

#### Scenario: 默认训练
- **WHEN** 未显式启用 FSDP
- **THEN** 现有 DDP 和 LoRA 行为保持不变

#### Scenario: 不支持的组合
- **WHEN** PI05 FSDP 配合 PEFT、视觉 MEM、编译、非全参模式、非 FP32 参数或自定义 wrap policy
- **THEN** 启动时给出明确配置错误

#### Scenario: GPU 验证不可用
- **WHEN** 执行环境没有两张可用 CUDA GPU
- **THEN** 双卡测试明确跳过，文档和交付记录不得声称显存、吞吐或 CUDA 分布式验证已经通过

### Requirement: 训练中断协调退出并清理

系统 MUST 在 SIGINT/SIGTERM 请求后让所有训练 rank 在共同边界停止，并在正常、异常及中断退出时清理已初始化的分布式进程组。

#### Scenario: Ctrl+C 或仅一个 rank 收到信号
- **WHEN** 所有 rank 仍可响应且某个 rank 收到 SIGINT/SIGTERM
- **THEN** 请求在固定训练边界传播到所有 rank，退出前执行清理，不因中断额外保存或发布，退出状态表示中断

#### Scenario: 清理阶段再次收到信号
- **WHEN** worker 正在清理时收到重复 SIGINT/SIGTERM
- **THEN** worker 的清理继续执行，原信号处理器在退出清理范围后恢复

#### Scenario: GPU 功耗验证
- **WHEN** 双卡中断测试结束
- **THEN** 记录 CUDA 进程和功耗状态，区分退出流程验证与高功耗根因，不自动重置设备或启动恢复计算来改变测量结果
