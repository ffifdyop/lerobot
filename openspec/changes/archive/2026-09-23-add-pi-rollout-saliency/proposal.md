## Why

变更前本地 π0/π0.5 没有 rollout saliency，SmolVLA 固定解释 a₀，而 RTC 可能已丢弃该动作。本变更统一解释下一入队策略动作，并支持观察指定层与去噪阶段；实现与验收记录见 tasks.md 和 validation.md。

## What Changes

- 为 π0/π0.5 增加 grad_x_input、attention，复用异步重放与显示。
- **BREAKING**：正常 Policy saliency 从固定 a₀ 改为 RTC 实际保留的首个动作 a_d；推理前保存输入，接受入队后提交。
- 支持真实零基层号和去噪步骤筛选，默认全选，单组等权平均。
- π 系列预算保留 action 份额；π0.5 文本及离散状态统一标为 prompt。
- 泛化 patch/letterbox 映射；MEM 模式明确提前拒绝。
- 训练和默认推理语义保持不变；梯度调整、attention 捕获仅作用于解释器专用路径，不删除共享 RTC 的梯度截断。
- 重放一致性纳入测试/离线验收，不引入默认逐帧门禁；预算舍入容差按实际捕获精度验证。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `rollout-saliency`：支持本地 π 模型、下一入队动作归因、单组 attention 筛选及架构相关预算。

## Impact

影响 saliency 包、RTC engine/queue、策略副本采样与 attention 观察、显示和测试。同步修正 SmolVLA；无需新增服务。与进行中的 `add-rollout-human-takeover` 存在 saliency 规格重叠，合并时必须协调 Policy a_d 与 Shadow 反事实语义。

不更改公共 Policy 推理签名、训练 loss/参数梯度语义、checkpoint 格式或队列接受算法。队列只暴露原有裁切元数据，新增观察入口默认关闭。开启 saliency 的资源与时序影响单独测量，不承诺同 seed 的整体 rollout 轨迹不变。

## Non-goals

首版不支持 MEM saliency、远程策略、非 RTC 推理、实际执行前缀归因及多组输出。同次重放的多组对比记录为后续功能。
