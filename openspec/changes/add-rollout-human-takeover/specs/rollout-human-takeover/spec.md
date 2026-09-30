## ADDED Requirements

### Requirement: 接管能力为显式 opt-in

系统 SHALL 仅在 takeover 配置启用时创建接管事件后端、Leader–Follower Synchronization 和 RTC shadow 生命周期。启用时系统 MUST 在机器人连接前校验 teleoperator 已配置且支持 feedback、enable torque 与 disable torque；不满足时 MUST fail-fast。系统禁止因 takeover 修改 leader PID 或 follower 既有配置行为。

#### Scenario: Takeover 未启用时保持兼容

- **WHEN** 用户未启用 takeover
- **THEN** rollout 不创建接管 listener 或 synchronization，且现有 Base/DAgger 行为和数据 schema 保持不变

#### Scenario: Teleoperator 不支持 feedback 时拒绝启动

- **WHEN** 用户启用 takeover，但 teleoperator 不提供 feedback 或 torque capability
- **THEN** 配置或上下文构建阶段抛出明确错误，机器人控制循环不启动

#### Scenario: 启用接管不写 PID

- **WHEN** takeover 初始化 Leader–Follower Synchronization
- **THEN** 系统不读写任何 leader PID 配置，也不增加 follower 配置写入

### Requirement: 控制阶段和事件优先级唯一确定

系统 SHALL 使用 `POLICY`、`HUMAN`、`RESUMING`、`PAUSED` 与 `FAULT` 控制阶段，并按 `STOP > FAULT > PAUSED > HUMAN > POLICY` 处理同一控制节拍内的竞争事件。`human_active` MUST 等于 hold 输入与 latch 输入的逻辑或；同一控制节拍 MUST 只有一个 Control Source，禁止混合 policy 与 human action。

#### Scenario: 按住接管并松开

- **WHEN** 当前阶段为 `POLICY`，hold 输入按下后又松开，且 latch 未启用
- **THEN** 系统进入 `HUMAN`，松开后进入 `RESUMING`，收到 fresh policy chunk 后回到 `POLICY`

#### Scenario: Latch 保持人工控制

- **WHEN** latch 已启用且 hold 输入松开
- **THEN** Control Source 仍为 Human，直到 latch 被明确解除

#### Scenario: 暂停覆盖人工接管

- **WHEN** 同一节拍同时存在暂停和人工接管事件
- **THEN** 系统进入 `PAUSED`，不下发新的 policy 或 human action，并保存暂停前的 Control Source 供恢复

#### Scenario: Fault 覆盖暂停

- **WHEN** `PAUSED` 期间发生控制完整性故障
- **THEN** 系统进入 `FAULT`，禁止解除暂停事件绕过 Fault 确认

### Requirement: Policy 阶段持续同步主从臂

在 `POLICY` 阶段，系统 SHALL 每个控制节拍使用 follower 实际观测中的共同位置 feature 调用 teleoperator feedback，使单臂或带左右前缀的双主臂跟随从臂。系统 MUST 在首次 feedback 前启用 leader torque；目标禁止来自未经执行的 Policy Proposal。

#### Scenario: 双臂位置按前缀路由

- **WHEN** follower observation 同时包含 `left_*.pos` 与 `right_*.pos`
- **THEN** synchronization 将左右位置分别路由到对应 leader，且每个共同位置 feature 恰好出现一次

#### Scenario: Policy proposal 未执行时不驱动 leader

- **WHEN** RTC 产生一条随后被丢弃或覆盖的 Policy Proposal
- **THEN** leader feedback 仍取自 follower 实际观测，不使用该 proposal 作为目标

#### Scenario: 单臂 feedback 失败导致双臂 Fault

- **WHEN** 任一侧 leader feedback 或 torque 操作失败
- **THEN** 系统停止两侧后续 synchronization，尽力关闭双 leader torque，并进入共享 `FAULT`

### Requirement: 人工接管采用硬仲裁

从 `POLICY` 进入 `HUMAN` 时，系统 MUST 先阻止新的 policy action dispatch，再关闭双 leader torque，然后才允许 teleop action 经过 teleop action processor、robot action processor 和 robot dispatch。接管节拍禁止向 robot 同时发送 policy 与 human action。

#### Scenario: 接管边界无 policy 泄漏

- **WHEN** 接管事件与一个 RTC action 在同一控制节拍到达
- **THEN** arbiter 选择 Human，RTC action 不发送到 robot，且 robot 最多收到一条 human-derived command

#### Scenario: 人工动作沿用处理链

- **WHEN** `HUMAN` 阶段读取到主臂动作
- **THEN** 该动作依次经过既有 teleop 与 robot processors，robot 接收处理后的命令

### Requirement: RTC shadow 与 fresh resume 隔离

`HUMAN`、`PAUSED` 和可诊断 `FAULT` 阶段的 RTC SHALL 运行在 shadow 模式：只基于最新观测产生 Policy Proposal，禁止把 proposal 合并进可执行 action queue。进入或退出 shadow MUST 使旧 epoch 的在途结果失效。进入 `RESUMING` 时系统 MUST 清空 RTC queue、processors 与 interpolator，保持最后一条 Executed Action，并等待 fresh executable chunk 后平滑回到 `POLICY`。

#### Scenario: Shadow proposal 永不执行

- **WHEN** `HUMAN` 阶段 RTC 连续产生多个 shadow proposal
- **THEN** proposal 可被记录和解释，但 robot 收到的动作全部来自 Human

#### Scenario: Shadow 只保留最新观测

- **WHEN** shadow inference 速度低于 observation 产生速度
- **THEN** 系统丢弃过时的待处理观测，不积压按旧画面生成的 proposal

#### Scenario: 切回时拒绝旧 chunk

- **WHEN** Human 结束且旧 epoch 的 RTC chunk 随后完成
- **THEN** 该 chunk 被丢弃，系统继续保持最后人工命令，直到新 epoch 的 fresh chunk 可用

### Requirement: 所有控制阶段持续观察与显示

系统 MUST 在动作阶段分支之外执行 observation processing、相机 telemetry 和控制状态 telemetry。`POLICY`、`HUMAN`、`RESUMING`、`PAUSED` 与 `FAULT` 均 MUST 持续刷新相机；系统禁止在控制线程中执行会冻结 telemetry 的阻塞式主臂对齐。

#### Scenario: Paused 相机不中断

- **WHEN** rollout 在 `PAUSED` 保持多个控制节拍
- **THEN** 每个节拍仍处理最新 observation 并向显示后端记录相机帧和 phase

#### Scenario: Fault 保留诊断显示

- **WHEN** 系统进入 `FAULT`
- **THEN** 新 robot action 停止，但相机、控制状态和可用的 shadow saliency 继续更新

#### Scenario: 状态路径首帧存在

- **WHEN** Rerun 或 Foxglove 创建本次 rollout 的初始视图
- **THEN** phase、source、intervention、左右 tracking error、fault code 与 saliency mode 的数值路径已存在

### Requirement: 记录候选、选中和实际动作

Takeover 数据集 SHALL 只为 `POLICY` 与 `HUMAN` 的有效动作节拍写训练帧。既有 `action` MUST 表示 Selected Action；每帧 MUST 额外记录 `policy_action`、`policy_action_valid`、`intervention`、`control_source` 与 `executed_action`。`executed_action` MUST 来自 robot dispatch 返回值并使用 robot action feature schema，禁止假设其形状与 policy action 相同。

#### Scenario: Human 帧保留同帧 policy proposal

- **WHEN** `HUMAN` 节拍的 shadow inference 成功
- **THEN** 数据帧的 `action` 为 human selected action，`policy_action` 为同观测 proposal，`policy_action_valid=true`，且 `intervention=true`

#### Scenario: Proposal 缺失时显式无效

- **WHEN** 某个有效动作节拍没有可对齐的 Policy Proposal
- **THEN** `policy_action_valid=false`，系统禁止用零向量冒充有效 proposal

#### Scenario: Robot 裁剪后动作单独记录

- **WHEN** robot processor 或 robot dispatch 对 Selected Action 做安全裁剪
- **THEN** `action` 保留训练动作空间值，`executed_action` 记录 robot 实际接受值

#### Scenario: 非动作阶段不污染训练轨迹

- **WHEN** phase 为 `PAUSED`、`RESUMING` 或 `FAULT`
- **THEN** 系统不向 LeRobotDataset 写训练帧，而向事件 sidecar 记录 timestamp、phase、事件、tracking error、fault code 与 RTC epoch

### Requirement: Fault 分级并作用于双臂共享域

系统 SHALL 将控制完整性故障分为 recoverable 与 fatal。任一 Fault 都 MUST 停止产生新的 robot action、尽力关闭双 leader torque并锁存，禁止自动恢复；任一侧通信、feedback 或 torque 的部分失败 MUST 影响双臂。Recoverable Fault 只允许人工确认并重新同步后恢复；Fatal Fault MUST 保存失败 episode 并结束控制。

#### Scenario: Recoverable tracking error

- **WHEN** tracking error 连续达到配置的软阈值和周期数，但未达到硬阈值
- **THEN** 系统进入 recoverable Fault，双臂停止新动作，等待人工确认和重新同步

#### Scenario: Fatal 双臂部分成功

- **WHEN** 一侧 torque/feedback 成功而另一侧失败
- **THEN** 系统进入 fatal Fault，禁止成功侧继续控制，并保存失败 episode

#### Scenario: Fault 不会自动恢复

- **WHEN** 导致 Fault 的瞬时条件自行消失
- **THEN** 系统仍保持 Fault，直到 recoverable Fault 被人工确认或 fatal Fault 完成 teardown

### Requirement: 输入与生命周期可替换且可清理

系统 SHALL 通过事件接口实现默认 Shift hold、Tab latch、Space pause 和 Esc stop，并允许未来脚踏板后端产生等价事件。Teardown MUST 停止 listener，禁止 teardown 后的迟到事件改变 control phase。

#### Scenario: Listener 正常清理

- **WHEN** rollout 正常结束或因异常进入 teardown
- **THEN** 事件 listener 被停止并回收，随后到达的键盘事件不触发状态转换

#### Scenario: 更换输入后端不改变仲裁

- **WHEN** 脚踏板后端产生与 hold press/release 等价的事件
- **THEN** 状态转换和 Control Source 选择与键盘后端一致
