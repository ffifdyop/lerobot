## ADDED Requirements

### Requirement: 视角消融必须显式启用并在启动期解析

`lerobot-rollout` SHALL 提供默认关闭的 View Ablation 配置，至少包含 enabled、initial views、每视角 hotkeys、transition timeout 与 event log path。启用时，系统 MUST 同时接受 robot raw camera key 与 policy canonical view key，内部状态、显示和持久化记录 MUST 统一使用 canonical key；未知、非视觉或有歧义的名称 MUST 在机器人开始运动前整次拒绝，禁止模糊匹配或部分生效。

#### Scenario: 默认 rollout 行为不变

- **WHEN** 用户未启用 View Ablation
- **THEN** 系统不注册 mask 热键、不注入 mask processor、不增加 dataset 字段，策略输入与变更前一致

#### Scenario: raw key 解析为 canonical key

- **WHEN** 用户启用 View Ablation，并用可唯一映射的 robot raw camera key 指定初始视角
- **THEN** 系统将其解析为 rename 后的 canonical view key，启动日志、运行状态和记录只使用该 canonical key

#### Scenario: 禁用时设置子配置被拒绝

- **WHEN** `view_mask.enabled=false` 且 initial views、hotkeys、timeout 或 event log path 任一项被显式改为非默认值
- **THEN** 配置校验 fail-fast，并提示必须显式启用 View Ablation

#### Scenario: 非法视角整次拒绝

- **WHEN** 一次配置同时包含合法视角和未知、非视觉或有歧义的视角
- **THEN** 整次配置被拒绝，错误列出非法项与可用 canonical views，禁止只应用合法子集

#### Scenario: 允许全部视角被屏蔽

- **WHEN** initial views 或运行时切换后的集合包含全部可用相机视角
- **THEN** 系统接受该集合并按普通 View Mask 处理，不要求额外确认或特殊告警

### Requirement: 黑图替换必须保留 observation schema 与原始数据

系统 SHALL 在 policy observation rename 完成后、normalization 与 policy-specific image transform 之前，将每个 masked canonical view 替换为同 shape、dtype、device 的全零 tensor。系统 MUST 保留相机 key 与模型槽位顺序，禁止删除 key、紧凑重排剩余视角、停止物理相机或修改供 dataset/telemetry 使用的原始 observation。

该替换 MUST 通过 snapshot 携带的空间 mask operator 接口完成；当前全图清零 MUST 是该接口的默认实现，而不是写死在 processor 中。operator MUST 声明受影响的 canonical views、返回不污染输入的 observation mapping，并提供绑定到 revision sidecar 的 JSON-serializable metadata。未来不同形状或尺寸的局部 mask SHALL 能通过新增 operator/factory 接入，而无需修改 processor、Sync/RTC inference、VQA 或 saliency 的调用边界。本 change 不要求提供局部 geometry 的 CLI 配置或具体实现。

#### Scenario: 单视角只在策略输入中变黑

- **WHEN** 三相机 rollout 的第二个 canonical view 被 mask
- **THEN** policy batch 中第二个 view 为全零、第一和第三个 view 保持原值，dataset 与人工显示中的三路图像均为相机原图

#### Scenario: 固定 schema 远程策略保持全部键

- **WHEN** DM05 等要求全部 camera keys 的远程策略启用一个 masked view
- **THEN** 请求仍恰含所有配置 camera keys，masked 槽位编码为黑图，服务端不会因 key 缺失而失败

#### Scenario: mask 操作不污染调用者 observation

- **WHEN** 同一原始 observation 被用于 policy preprocessing、dataset recording 与 telemetry
- **THEN** 仅 policy preprocessing 的派生 mapping/tensor 被修改，其他消费者观察到的对象内容不变

#### Scenario: 全图 mask 是空间 mask 的默认实现

- **WHEN** 当前 CLI/热键选择一个 canonical view，且未配置其他空间 mask operator
- **THEN** snapshot 使用 `FullFrameViewMask` 将该 view 整幅清零，processor 只调用统一 operator 接口，不包含全图 mask 的专用分支

#### Scenario: 自定义空间 mask 保持推理边界不变

- **WHEN** 后续实现通过 controller operator factory 为一个 view 提供不同形状或尺寸的 immutable mask operator
- **THEN** 同一 snapshot 可直接经过 Sync、RTC、VQA 与 saliency 路径并由现有 processor 应用，无需修改这些消费者的接口

### Requirement: 动作、文本与解释必须共享同一 View Mask

动作推理、VQA、自动任务规划与 saliency SHALL 使用同一个 immutable `{revision, masked_views}` snapshot。禁止让文本或解释路径绕过 mask，也禁止把 masked view 的 saliency 人为清零或隐藏。

#### Scenario: VQA 使用黑图输入

- **WHEN** View Mask 为 ACTIVE 且用户发起 VQA
- **THEN** VQA preprocessor 收到与动作推理相同 revision 的黑图视角，而不是原始相机图像

#### Scenario: 被屏蔽视角保留真实 saliency

- **WHEN** 模型对 masked 黑图槽位产生非零 gradient 或 attention
- **THEN** saliency 输出保留该非零结果，并明确标记该视角为 MASKED

### Requirement: 运行时热键必须使用锁存、edge-triggered 语义

启用 View Ablation 后，系统 SHALL 按 canonical view 声明顺序默认绑定 F1、F2、F3……，并允许显式覆盖。一次独立 key-down MUST 切换对应视角的 membership，再次独立按下 MUST 解除；同一次长按的自动重复禁止触发多次。热键与 strategy、teleop 或已有 rollout 控制冲突时 MUST 在启动期报错，禁止静默抢占。

#### Scenario: 按一次加入再按一次移除

- **WHEN** 初始集合为空，操作者按下并松开 F1，再次按下并松开 F1
- **THEN** 第一次使对应视角进入 mask，第二次使其退出 mask，长按期间集合不反复切换

#### Scenario: 同 tick 多键原子合并

- **WHEN** 控制循环消费按键前依次收到 F1 与 F2 的有效 edge
- **THEN** 系统一次性提交包含两个视角的最终集合，只产生一个新 revision 和一次 Inference Flush

#### Scenario: 热键冲突启动失败

- **WHEN** View Mask 热键与 DAgger、Highlight、teleop 或其他 rollout 控制使用同一按键
- **THEN** rollout 启动失败，错误明确列出冲突按键与双方功能

### Requirement: Mask Transition 必须 revision 化且 latest-wins

启动 View Mask SHALL 为 revision 0。每次有效集合变化 MUST 产生严格递增且不复用的新 revision；无实际变化禁止增加 revision。transition MUST 原子清空旧 policy/pre/post 状态、视觉/history、未执行 action queue 与插值序列，同时保留当前 task 和最后一条已发送动作作为 hold。transition 过程中收到的新变化 MUST latest-wins，所有旧 revision 的推理、动作、queue merge 与解释结果 MUST 失效。

#### Scenario: 切换后旧 chunk 不再执行

- **WHEN** RTC queue 或 policy 内部 queue 仍有旧 revision 动作且操作者切换 mask
- **THEN** 旧 queue 被清空，机器人保持最后一条已发送动作，直到最新 revision 的 fresh action 到达

#### Scenario: Sync 中途结果被丢弃

- **WHEN** Sync 或远程 inference 以 revision N 开始，返回前 mask 已变为 revision N+1
- **THEN** revision N 的返回动作在进入 interpolator 或 robot dispatch 前被丢弃

#### Scenario: transition 中再次切换

- **WHEN** revision N+1 尚在 APPLYING 时操作者提交另一个有效 mask 集合
- **THEN** 系统创建 revision N+2、再次 Inference Flush，并禁止 N+1 的晚到结果进入执行或记录

#### Scenario: reset 恢复启动集合但 revision 不回退

- **WHEN** 当前 mask 不等于 launch mask 且用户执行显式会话 reset
- **THEN** 系统以一个更大的新 revision 恢复 launch mask，禁止把 revision 重置为 0

### Requirement: Mask Application State 必须准确约束动作与查询

有效 transition 后状态 MUST 为 APPLYING；首个匹配最新 revision 的 fresh policy action 可执行时 MUST 变为 ACTIVE。默认 10 秒且可配置的 timeout 内没有 fresh action，或 inference 明确失败时，状态 MUST 锁存为 FAILED。APPLYING 与 FAILED 期间 MUST 禁止新训练帧、VQA 与自动任务规划；禁止执行、记录或解释旧 revision 动作。

#### Scenario: fresh action 激活 revision

- **WHEN** APPLYING 状态收到带最新 revision 的首个有效 policy action
- **THEN** 该 revision 变为 ACTIVE，动作才可进入 robot dispatch，后续 VQA 与自动任务规划恢复

#### Scenario: 过渡超时

- **WHEN** policy inference 已处于可运行阶段，但 transition timeout 内没有得到最新 revision 的 fresh action
- **THEN** 状态进入锁存 FAILED，晚到动作被丢弃，系统不得自动回滚 mask、恢复旧 queue 或自动重试

#### Scenario: 非推理阶段暂停 timeout

- **WHEN** 操作者在尚未 start、Human 或 Paused 且 policy inference 被暂停的阶段切换 mask
- **THEN** 系统接受并记录新 revision，但 timeout 不累计，且不得打断人工控制

#### Scenario: 非交互式 FAILED 结束运行

- **WHEN** 非交互式 rollout 的 Mask Application State 进入 FAILED
- **THEN** 当前 rollout 结束并执行既有 teardown，禁止通过晚到结果恢复运动

#### Scenario: Human 控制期间 FAILED 不夺权

- **WHEN** Human 已是 Control Source 且 Mask Application State 进入 FAILED
- **THEN** Human 控制可继续，Policy 保持禁用，直到显式会话 reset 或进程重启

### Requirement: View Mask 必须跨普通运行边界保持

当前 View Mask MUST 跨普通 episode rotation 与交互式 `/start` control segment 保持。显式 `/reset` 或进程重启才恢复 CLI launch mask；View Mask Control MUST 在会话存活期间可用，包括尚未 start、Policy、Human、Paused 与 segment 间隙。

#### Scenario: episode rotation 不改变 mask

- **WHEN** Sentry/DAgger 因视频或策略规则保存当前 episode 并开始下一 episode
- **THEN** View Mask、revision 与 ACTIVE 状态保持不变

#### Scenario: segment 间隙可预设 mask

- **WHEN** 交互式 rollout 尚未 `/start` 或上一 segment 已结束时操作者切换视角
- **THEN** 新集合被接受并记录，下一次 policy inference 使用该集合

### Requirement: 训练帧必须携带实际动作 provenance

启用 View Ablation 的新 dataset SHALL 定义 `view_mask`（bool，shape `(N,)`，names 为 canonical view keys）与 `view_mask_revision`（int64，shape `(1,)`）。Policy 帧 MUST 使用 dispatched action 携带的 snapshot，禁止在录帧时读取 controller 瞬时状态；Human 帧 MUST 使用该观测时刻的当前 snapshot，并由 intervention/control_source 单独表示动作来源。APPLYING 与 FAILED 期间 MUST NOT 写训练帧。

#### Scenario: RTC 动作记录其生成 revision

- **WHEN** RTC action queue 中的动作由 revision N 生成，而 controller 在录帧前已收到 revision N+1 请求
- **THEN** 若动作尚未 dispatch 则被丢弃；若它已合法 dispatch，则对应帧记录 revision N，禁止错误标为 N+1

#### Scenario: Human 帧记录 Shadow Policy 条件

- **WHEN** Human 是 Control Source 且当前 View Mask 为 revision N
- **THEN** 人工训练帧记录 revision N 与其布尔集合，同时 intervention/control_source 表明动作来自 Human

#### Scenario: APPLYING 不产生训练样本

- **WHEN** transition 后系统保持最后动作并等待 fresh action
- **THEN** 这些 hold/control ticks 只进入事件与 telemetry 时间线，不进入训练 dataset

#### Scenario: 不兼容旧 dataset 禁止续写

- **WHEN** View Ablation 已启用且 `--resume` 指向缺少上述字段、dtype/shape 不同或 names 不一致的 dataset
- **THEN** rollout 在录制前 fail-fast，要求创建兼容 schema 的新 dataset，禁止自动迁移

### Requirement: 切换事件必须形成可对齐的 sidecar

系统 SHALL 记录启动、每次有效 transition、application state 变化与失败事件。事件 MUST 包含 run-relative monotonic time、revision、old/new mask、source 与 application state；可用时 MUST 同时包含 control phase、dataset episode index 与最近 frame index。文件头 MUST 记录一次 run start UTC。默认位置 MUST 为 dataset 模式的 `<dataset.root>/meta/view_mask_events.jsonl`，无 dataset 的 Base rollout 为 `outputs/rollout/<timestamp>/view_mask_events.jsonl`，显式配置可覆盖。

#### Scenario: dataset 事件与帧对齐

- **WHEN** 一次 mask transition 发生在 episode 2 的已录 frame 17 与下一训练帧之间
- **THEN** JSONL 事件记录精确 monotonic time、episode 2、最近 frame 17 与新 revision，而不是伪造一个训练帧 timestamp

#### Scenario: Base rollout 仍有事件记录

- **WHEN** Base rollout 无 dataset 且 View Ablation 已启用
- **THEN** 系统在默认 rollout output 目录创建事件 JSONL，并记录相同的 revision/state 时间线

### Requirement: 可视化必须显示 mask 状态但不得成为控制面

Rerun SHALL 保留原始相机图像，并对 masked camera tile 提供红色 `MASKED` 标识，同时记录每视角 0/1、revision 和 APPLYING/ACTIVE/FAILED。显示 annotation MUST 与 policy/dataset 原始像素隔离。View Mask 推理、热键和记录 MUST 不依赖 Rerun；Viewer selection、button 或 Blueprint 变化禁止改变 View Mask。

#### Scenario: Rerun 显示原图和 MASKED 标识

- **WHEN** 一个视角被 mask 且 Rerun display 已启用
- **THEN** 对应 tile 仍显示原始相机画面并带红色 MASKED 标识，状态时间线显示当前 revision

#### Scenario: 关闭可视化仍可切换

- **WHEN** display_data=false 且 View Ablation 已启用
- **THEN** CLI 初始 mask、运行时热键、Inference Flush、dataset/JSONL 记录均正常工作

#### Scenario: Foxglove 模式不改变推理语义

- **WHEN** display mode 为 Foxglove
- **THEN** 系统尽可能发布等价 mask 状态，且 black-view inference、热键与记录行为与 Rerun 模式一致
