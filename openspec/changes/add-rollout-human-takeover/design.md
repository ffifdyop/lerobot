## Context

当前 `lerobot-rollout` 已将机器人、teleoperator、policy processors、RTC inference、DAgger 数据录制、Rerun/Foxglove 与 saliency sidecar 组装在同一生命周期中。现有 DAgger 提供 `AUTONOMOUS → PAUSED → CORRECTING` 三态和双主臂平滑交接，但进入暂停会阻塞式对齐主臂并暂停 RTC；暂停阶段不记录 telemetry，人工阶段也不产生新的 saliency 或同帧 policy proposal。

当前 HIL-SERL 采用“每 tick 计算 policy，再由 intervention processor 覆盖”的同步仲裁模型；远端 leader-intervention 分支还实现过单臂 follower→leader 跟随。然而这些实现面向 EE-delta/RL transition，不能直接满足双臂 DAgger、异步 RTC stale queue、相机/saliency 连续显示和 LeRobotDataset 语义。

本设计必须复用现有 rollout 生命周期和 DAgger 资产，禁止引入第二套机器人控制循环。未启用 takeover 时，现有 rollout 行为必须兼容。

## Goals / Non-Goals

**Goals:**

- 必须支持 policy 与双主臂之间的硬切换，同一控制节拍只能有一个 Control Source。
- 必须支持按住接管、锁定接管、暂停和停止，并允许将键盘事件后端替换为脚踏板。
- Policy 控制期间双主臂必须跟随双从臂实际姿态，使接管不依赖现有阻塞式两秒对齐。
- Human/Paused/Fault 阶段必须持续刷新相机；shadow saliency 必须明确标记为反事实解释。
- 切回 policy 时必须拒绝 RTC 旧队列，等待基于当前观测产生的 fresh chunk。
- 数据必须区分 Policy Proposal、Selected Action 与 Executed Action，并能重建控制来源和故障时间线。
- 任一单臂控制完整性失败必须作用于共享双臂故障域。

**Non-Goals:**

- 禁止把 HIL-SERL EE-delta processor 或 RL replay buffer 引入 rollout。
- 不修改 leader PID、电流、力矩、速度或加速度限制，不改变 follower 既有配置。
- 不实现 human/policy 加权混合，不把非动作阶段的 hold 帧作为训练示范。
- 不把软件 Fault 状态描述为实体急停或人体接触安全保证。

## Decisions

### 1. 扩展 DAgger，而不是 BaseStrategy 或 HIL-SERL

DAgger 已拥有 teleop 连接、键盘/踏板事件、teleop/robot processors、episode 生命周期、RTC reset 与数据集写入；在 BaseStrategy 重写会复制这些职责。HIL-SERL 的同步 actor 没有 RTC 队列，也不满足当前双主臂 joint-position action 契约。因此 takeover 必须作为 DAgger 的 opt-in 能力实现，并抽取可复用的小组件，而不是新建策略或整体 cherry-pick 远端分支。

配置校验必须在启动前确认 teleoperator 已配置，并支持 feedback 与 torque capability；不满足时必须 fail-fast。未启用 takeover 时禁止启动额外 listener、shadow 路径或 leader synchronization。

### 2. 显式控制阶段和单一仲裁器

内部阶段统一为 `POLICY`、`HUMAN`、`RESUMING`、`PAUSED` 与 `FAULT`，外部停止事件优先级为：

```text
STOP > FAULT > PAUSED > HUMAN > POLICY
```

`human_active = hold_pressed OR latch_enabled`。Shift 默认作为 hold 输入，Tab 切换 latch，Space 切换 pause，Esc 停止；键位必须可配置，事件模型禁止依赖具体键盘库。

每个控制节拍先收集事件和观测，再由唯一 arbiter 选择 Policy Proposal 或 teleop action。禁止两个来源同时向 robot 发送动作，也禁止在 processor 之后通过数值混合隐式改变 Control Source。

### 3. 共享 LeaderFollowerSync

新增的 synchronization 对象只依赖标准 teleoperator feedback/torque capability，并通过 feature key 交集兼容单臂与带 `left_`/`right_` 前缀的双臂。

- 进入 `POLICY` 时启用 leader torque；每个控制节拍以 follower 实际观测位置调用 `send_feedback`。
- 进入 `HUMAN` 时必须先阻止新的 policy dispatch，再关闭双 leader torque，然后才允许读取并下发人工动作。
- 返回 `POLICY` 时，leader 已与 human 驱动后的 follower 同姿；恢复 fresh policy 后重新启用跟随。
- 任一侧 feedback/torque/read 失败必须让整个双臂 synchronization 失败，禁止另一侧继续。

目标必须来自 follower 实际姿态，而不是尚未执行的 policy proposal。该组件禁止写 PID；leader 使用 EEPROM 中既有参数。

### 4. DispatchResult 明确动作语义

现有发送路径把 policy-space action 与 `robot.send_action()` 的返回值混为一谈。控制层必须产生结构化 dispatch 结果，至少包含：

- `policy_action`：当前观测的 Policy Proposal；
- `selected_action`：仲裁后、robot processor 前的训练动作；
- `robot_command`：robot processor 后的命令；
- `executed_action`：`robot.send_action()` 返回的实际接受值；
- `control_source` 与 `phase`。

数据集的既有 `action` 继续表示 `selected_action`，避免改变训练标签的动作空间；`executed_action` 必须使用 robot action feature schema，禁止假设其形状与 policy action 相同。

### 5. RTC 使用独立 shadow 生命周期

Human/Paused/需要诊断的 Fault 阶段需要 Policy Proposal 和 saliency，但正常 RTC merge 假设此前 policy action 已实际执行，不能继续复用。因此 RTC 必须提供显式 shadow 模式：

- 进入 shadow 时清空可执行 queue、递增 epoch，使在途结果失效；
- shadow inference 只基于最新观测生成独立 proposal，禁止合并进可执行 queue；
- shadow 请求采用容量为一或等价 latest-only 语义，过时请求必须丢弃；
- shadow proposal 可用于记录和 saliency，禁止用于机器人 dispatch；
- saliency 使用显式 queued-policy/shadow mode：Policy 接受入队后解释 a_d；Shadow proposal 有效后独立提交，显式 target_action_index=0，不伪造 first_queued_action_index，不等待 queue merge。两方法共享目标、layer/step 筛选和原始预算；阶段切换使旧 epoch 快照失效。实现时 recorder 须增加模式分支，当前仅接受入队快照的入口不能直接接收 Shadow；
- 退出 shadow 后必须 reset RTC、processors 和 interpolator，并等待 fresh executable chunk。

`RESUMING` 期间 follower 保持最后一条 Executed Action。fresh chunk 到达后允许使用既有插值器做短暂平滑恢复，禁止使用人工阶段缓存的 shadow proposal 快速恢复。

### 6. Observation/telemetry 与控制阶段解耦

每个控制节拍必须先读取并处理 observation，再执行相机 telemetry 和状态 telemetry；阶段分支只能决定动作来源和 dispatch。现有阻塞式 `teleop_smooth_move_to` 禁止在显示主循环中使用。

从第一帧开始记录数值型 `control.phase`、`control.source`、`control.intervention`、左右 tracking error、fault code 与 `saliency.mode`，保证 Rerun 初始 blueprint 能包含这些路径。若增加文字状态，必须使用独立 status logger，禁止把 telemetry 字段污染到 robot action processor。

显示和 saliency 是旁路：显示后端失败或 saliency 丢帧只能告警，禁止触发 robot Fault；robot 控制错误也不得提前关闭显示生命周期。

### 7. 训练数据与事件时间线分离

LeRobotDataset 只写 `POLICY` 和 `HUMAN` 的有效动作帧。每帧包含 `action`、`policy_action`、`policy_action_valid`、`intervention`、`control_source` 与 `executed_action`；Human 阶段的 `policy_action` 来自同观测 shadow inference。

Paused、Resuming 和 Fault 不写训练帧，必须写入事件 sidecar/Rerun，至少包含 timestamp、phase、输入事件、tracking error、fault code 与 RTC epoch。这样既能完整复盘，又禁止 hold 命令成为训练示范。

### 8. 双臂共享 Fault 域

Fault 分为 recoverable 与 fatal，二者都必须停止产生新 robot action，并尽力关闭双 leader torque；禁止自动恢复。

- Recoverable：持续软 tracking-error 超限、同步超时或可恢复的控制周期超时。人工确认并重新同步后可继续同一 episode。
- Fatal：串口中断、左右臂部分成功、关键状态缺失、非有限/结构错误动作、硬 tracking-error 超限或无法确认的执行器控制失败。必须保存失败 episode 并结束控制。

Fault 期间相机、shadow saliency 和诊断记录必须继续。阈值和连续周期数必须配置化，禁止把未经验证的设备 raw 值硬编码为跨设备安全常量。

## Risks / Trade-offs

- **[Shadow 推理与控制争用 GPU]** → 使用 latest-only 队列，必须优先保障控制和相机；允许 shadow saliency 降帧。
- **[连续 leader follow 被操作者阻挡]** → tracking error 进入共享 Fault；明确该机制不是认证安全回路，且本变更不改 PID/电流限制。
- **[RTC 在途结果泄漏]** → shadow/execute 模式切换必须递增 epoch、清 queue，并用测试证明旧 chunk 永不 dispatch。
- **[双臂部分成功导致不对称状态]** → synchronization 操作必须聚合左右结果；任一异常让双臂共同 Fault。
- **[数据 schema 扩大且动作空间可能不同]** → `executed_action` 独立使用 robot schema；旧数据加载器不要求新字段，takeover 关闭时不创建扩展字段。
- **[阶段重构影响现有 DAgger cadence]** → takeover 必须 opt-in，并保留现有 corrections-only/continuous 回归测试。
- **[相机、saliency 与控制线程竞态]** → 沿用 saliency sidecar 的非阻塞/丢旧留新原则；teardown 必须按 listener、RTC/saliency、visualization 的所有权顺序执行。

## Migration Plan

1. 先增加无行为改变的配置、状态类型、dispatch result 和单元测试；默认关闭 takeover。
2. 加入 LeaderFollowerSync 与事件仲裁，在 mock 单/双臂上验证 policy/human 切换和共享 Fault。
3. 实现 RTC shadow/epoch 隔离和 fresh resume，再接入 shadow saliency。
4. 解耦 DAgger telemetry，增加数据字段与事件 sidecar。
5. 在 `rollout.sh` 以显式参数启用；先进行无负载低速验证，再运行完整双臂 rollout。

回滚必须只需关闭 takeover 配置并恢复原 DAgger 状态机；现有 base rollout 和未启用 takeover 的 DAgger 禁止依赖新字段或 shadow API。

## Open Questions

没有阻塞实现的产品决策。Recoverable/Fatal tracking-error 的默认阈值和连续周期数必须在实现阶段以配置项给出保守默认，并通过 mock 测试验证状态转换；具体硬件调参不属于本 change。
