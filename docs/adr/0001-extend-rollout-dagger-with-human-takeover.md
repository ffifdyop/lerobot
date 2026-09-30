---
status: accepted
date: 2026-09-17
---

# Extend rollout DAgger with reversible human takeover

LeRobot 真机 rollout 需要在不中断相机和 saliency 显示的情况下，由双主臂随时接管双从臂，并能安全地把控制权交还给策略。我们决定扩展现有 `lerobot-rollout` DAgger/RTC 管线，而不是新建控制循环或直接采用 HIL-SERL 分支，因为 DAgger 已拥有 teleop、数据集和交接生命周期，而当前 rollout 分支拥有必须保留的 RTC saliency 与双臂 EE6D 支持。

## Decision

### Integration boundary

- 新功能从 commit `84a89644` 建立的 `codex/rollout-human-takeover` 分支开发。
- `lerobot-rollout`、DAgger、RTC、现有 teleop/robot processors 和数据集生命周期保持为主干。
- `origin/feat/rl-leader-arm-intervention` 仅作为 leader-follow 行为和测试思路的参考；不整体 cherry-pick。该分支面向单臂 HIL-SERL/EE delta，且与当前 rollout 分支存在大量无关的 RL 重构。
- 抽取一个单/双臂共用的 Leader–Follower Synchronization 能力，基于标准 teleoperator feedback/torque 接口工作，不移植 HIL-SERL 的单臂 EE intervention processor。

### Control model

控制阶段为 `POLICY`、`HUMAN`、`RESUMING`、`PAUSED` 和 `FAULT`，事件优先级为：

```text
STOP > FAULT > PAUSED > HUMAN > POLICY
```

- `POLICY`：策略动作是唯一 Control Source；双主臂每个控制节拍跟随双从臂的实际姿态。
- `HUMAN`：接管事件先阻止新的策略动作下发，再释放双主臂扭矩；人工动作经现有 teleop 和 robot processor 后成为唯一 Control Source。
- `RESUMING`：清空 RTC queue/epoch 和插值状态，保持最后一条人工命令，等待基于最新观测生成的 fresh policy chunk，再平滑切回 `POLICY`。人工阶段产生的影子候选动作不得直接用于恢复。
- `PAUSED`：保持最后一条实际发送动作，不接受策略或人工新动作；解除暂停后返回暂停前的控制来源。
- `FAULT`：停止产生新机器人动作。Recoverable Fault 允许人工确认和重新同步后恢复；Fatal Fault 保存并结束当前 episode，且绝不自动恢复。

首个输入后端使用可配置键盘事件：Shift 按住接管、Tab 锁定/解除人工控制、Space 暂停/恢复、Esc 停止。`human_active = hold_pressed OR latch_enabled`；事件接口应允许以后接入脚踏板而不改变控制仲裁。

### Shadow inference and saliency

RTC 增加显式 shadow 模式。`HUMAN`、`PAUSED` 和需要诊断的 `FAULT` 阶段继续接收最新观测、生成 Policy Proposal 与 saliency，但不把候选动作合并到可执行队列。Shadow 请求采用 latest-only 语义，推理来不及时丢弃旧观测，机器人控制和相机显示优先。

非 Policy 阶段的热力图必须标记为 Shadow Saliency；它只说明策略在当前观测下的反事实关注区域，不能解释人工动作。退出 shadow 后必须 reset RTC，并等待 fresh chunk，防止未执行的候选动作泄漏到机器人。

### Display contract

Observation processing、相机 telemetry 和控制状态记录位于阶段分支之外，每个控制节拍都执行。Policy、Human、Resuming、Paused 和 Fault 均不得重启或冻结相机/Rerun 生命周期；saliency 失败或显示后端失败只能告警，不能改变机器人控制状态。

Rerun/Foxglove 从第一帧开始记录数值型控制状态，包括 phase、source、intervention、左右臂 tracking error、fault code 和 saliency mode。若显示文字状态需要扩展后端，应使用专用 status logger，不得把 telemetry-only 字段送入 robot action processor。

### Data contract

LeRobotDataset 只保存产生有效动作的 `POLICY` 和 `HUMAN` 帧。`PAUSED`、`RESUMING` 和 `FAULT` 的完整时间线写入 Rerun/事件 sidecar，避免 hold 帧被误当成训练示范。

每个训练帧区分：

- `action`：仲裁后选中的策略动作空间动作，是默认训练标签；
- `policy_action` 与 `policy_action_valid`：同观测的 Policy Proposal 及其有效位；
- `intervention` / `control_source`：动作由 Policy 还是 Human 选择；
- `executed_action`：robot processor 和安全裁剪后，机器人接口实际接受的动作，使用独立 robot-action schema。

事件 sidecar 至少记录 timestamp、phase、输入事件、tracking error、fault code 和 RTC epoch。

### Fault model

任一侧 leader/follower 通信或 torque 操作失败都作用于共享双臂故障域，禁止单臂继续运行。

Recoverable Fault 包括持续但未达到硬故障级别的同步误差、同步超时和可恢复的控制周期超时；恢复需要人工确认和重新同步。Fatal Fault 包括串口中断、左右臂部分成功、关键状态缺失、非有限或结构错误的动作、严重 tracking error，以及无法确认的执行器控制失败。

故障期间停止新动作，但相机、Shadow Saliency 和诊断记录继续运行；Fatal Fault 保存失败 episode 后要求人工排障并重新启动。

### Explicit safety scope

本变更不修改 leader PID、电流、力矩、速度或加速度限制，不改变 follower 的既有配置行为，也不增加实体急停。Leader 沿用各舵机 EEPROM 中现有 PID。该功能定位为研究原型，不提供人体接触安全保证；Fault 机制用于控制完整性和诊断，不等同于安全认证回路。

## Considered options

- **仅通过 CLI 启用现有 DAgger**：拒绝。当前 DAgger 在 Paused 阶段冻结相机 telemetry，在 Human 阶段暂停 RTC/saliency，且交接是阻塞式两步操作。
- **直接使用 HIL-SERL intervention processor**：拒绝。当前实现面向同步 RL transition 和 EE delta，普通双 SO leader 不满足其事件/action 契约，也不处理 RTC stale queue。
- **从 `origin/feat/rl-leader-arm-intervention` 起分支或整体 cherry-pick**：拒绝。该分支只有单臂实现、缺少 rollout 显示集成，并携带大范围无关 RL 重构。
- **给 BaseStrategy 新建 takeover 控制循环**：拒绝。它会重复 DAgger 已有的 teleop、episode、processor 和 handover 生命周期。
- **Policy/Human 动作加权混合**：拒绝。任一控制节拍只能有一个明确 Control Source。

## Consequences

- RTC 需要正式的 shadow 生命周期，而不是在人工阶段简单 pop/discard 正常队列。
- DAgger 的 observation/telemetry 路径必须从 phase 分支中解耦，现有阻塞式 handover 不能位于显示主循环中。
- Robot dispatch 需要保留 `robot.send_action()` 返回值，不能再把 Selected Action 当作严格的 Executed Action。
- 数据集 schema 增加 policy proposal、来源和实际发送动作；非动作阶段进入事件 sidecar 而非训练数据。
- 必须增加跨阶段相机连续性、shadow action 不下发、RTC reset/fresh resume、双臂故障联动、输入监听清理和数据字段语义的自动化测试。
