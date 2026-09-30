## Why

当前 `lerobot-rollout` 的 DAgger 可以执行人工修正，但无法在 policy 与双主臂之间直接、可逆地切换控制权：暂停阶段相机 telemetry 会冻结，人工阶段 RTC/saliency 会停止，切回时也缺少可供分析的同帧 policy proposal。需要在复用既有 rollout、DAgger、RTC 与双臂 teleop 管线的前提下，提供持续可观察的在线接管能力。

## What Changes

- 为 DAgger 增加明确的 `POLICY`、`HUMAN`、`RESUMING`、`PAUSED` 与 `FAULT` 控制阶段，以及按住接管、锁定接管、暂停和停止事件。
- 在 policy 控制期间持续让双主臂跟随双从臂实际姿态；接管时停止 policy 下发并释放主臂扭矩，切回时 reset RTC 并等待 fresh policy chunk。
- 为 RTC 增加 latest-only shadow inference，使人工、暂停和诊断阶段继续生成不会下发的 policy proposal 与反事实 saliency。
- 将相机 telemetry 和控制状态显示移出阶段分支，保证所有阶段持续显示相机、saliency 模式、控制来源、跟随误差与故障状态。
- 扩展 rollout 数据契约，区分选中动作、同帧 policy proposal、控制来源和机器人实际接受的动作；非动作阶段写入事件 sidecar，而非训练帧。
- 增加双臂共享的 recoverable/fatal fault 模型，任何单臂控制完整性失败都停止双臂新动作且保留诊断显示。

## Capabilities

### New Capabilities

- `rollout-human-takeover`: 覆盖双主臂同步、控制仲裁、可逆接管、RTC 恢复、数据记录和故障状态。

### Modified Capabilities

- `rollout-saliency`: 增加非 policy 控制阶段的 shadow saliency 语义、标识、latest-only 刷新和显示连续性要求。

## Impact

影响 `lerobot.rollout.strategies.dagger`、RTC inference engine、通用 rollout 控制/telemetry、双臂 teleoperator feedback、数据集 feature 构建、Rerun/Foxglove 状态显示、`rollout.sh` 配置及相关测试。默认未启用接管时必须保持现有 rollout 行为兼容。

## Non-goals

- 不直接合并 HIL-SERL leader-intervention 分支，也不引入其 EE-delta/RL replay 管线。
- 不修改 leader PID、电流、力矩、速度或加速度限制，不改变 follower 既有配置，不提供人体接触安全认证。
- 不实现 policy 与 human action 的加权混合，也不把暂停/故障 hold 帧作为训练示范。
