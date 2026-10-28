## Why

当前训练框架具备 FSDP2，但 PI05 的联合 attention 直接访问 VLM 与 action expert 的层内部模块，缺少可靠的分片边界。需要让双卡全参训练能够分片参数、梯度和优化器状态，同时继续使用现有预训练权重和单卡 rollout。

## What Changes

- 在 FSDP 准备阶段将 PI05 对应的 VLM/expert 层组成联合分片单元，普通训练和推理保持现有结构。
- 对联合前向、梯度检查点及 prefix/suffix 推理建立完整的模块调用边界。
- 支持完整权重导出和分片恢复，保留现有权重格式兼容性。
- 为训练脚本增加显式 FSDP 选项、限制校验和使用文档。
- 增加数值、模块归属、checkpoint 兼容性与真实双卡测试；如无 GPU，明确记录验证缺口。
- 补齐 Ctrl+C/SIGTERM 的协调退出与异常清理，避免中断直接绕过分布式进程组销毁。此修复不等同于确认或根治 NVIDIA 退出后高功耗问题。

## Capabilities

### New Capabilities

- `pi05-fsdp-training`: PI05 全参 FSDP2 训练、checkpoint 兼容及可验证的配置边界。

### Modified Capabilities

无。

## Impact

涉及 PI05 模型、分布式准备与权重导出扩展点、训练脚本、文档及测试。不新增依赖，不修改数据集或启动长时间训练。

## Non-goals

本次不支持 LoRA + FSDP、视觉历史 MEM + FSDP、FSDP 与 torch.compile 组合，也不承诺未经测量的显存或吞吐提升。
