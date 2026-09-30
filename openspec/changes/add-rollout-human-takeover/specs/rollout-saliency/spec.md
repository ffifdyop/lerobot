## ADDED Requirements

### Requirement: Shadow Saliency 语义与标识

当 Policy 不拥有 Control Source 时，系统 SHALL 将基于 Shadow Policy Proposal 生成的热力图标记为 Shadow Saliency。显示、视频或 telemetry 中 MUST 能区分 queued-policy saliency 与 shadow saliency；系统禁止把 Shadow Saliency 描述为人工动作解释或已执行动作解释。Policy 入队也不保证最终执行。

#### Scenario: Human 阶段显示反事实标识

- **WHEN** Control Source 为 Human 且 shadow inference 产生 saliency
- **THEN** 相机叠加图与状态 telemetry 将其标记为 shadow/counterfactual，并保持 Human 来源标识

#### Scenario: Policy 阶段保持执行语义

- **WHEN** Control Source 为 Policy
- **THEN** saliency 表示 RTC 接受后实际保留的首个策略动作 a_d，d 来自队列裁切结果，且不显示 shadow 标识；入队不等于最终执行

#### Scenario: Shadow 独立提交

- **WHEN** 当前 epoch 和 View Mask revision 下生成有效 Shadow Policy Proposal
- **THEN** 可直接提交带 shadow mode、观测/chunk 身份及显式 target_action_index=0 的快照，解释 proposal 的 a₀；禁止等待执行队列接受或伪造 first_queued_action_index，proposal 永不进入执行队列

### Requirement: Shadow Saliency latest-only 刷新

Shadow Saliency SHALL 使用最新观测优先的非阻塞刷新语义。推理速度低于相机或控制节拍时 MUST 丢弃过时请求，禁止阻塞机器人控制、相机 telemetry 或积压旧画面的热力图。

#### Scenario: Shadow 推理落后时丢弃旧请求

- **WHEN** shadow explainer 尚未处理完上一观测而多个新观测到达
- **THEN** 等待队列最多保留最新请求，旧请求被丢弃，控制循环和相机刷新不等待 explainer

### Requirement: 控制阶段切换不重启 Saliency 生命周期

Policy、Human、Paused、Resuming 与 Fault 之间切换时，系统 MUST 保持 saliency manager/recorder 和可视化会话的既有生命周期，禁止为切换控制来源而 stop/start sidecar。Saliency 故障只能沿用既有降级路径，禁止改变 Control Source 或 Fault 状态。

#### Scenario: Human 往返不重建 Explainer

- **WHEN** rollout 从 Policy 进入 Human 后再回到 Policy
- **THEN** saliency manager 和 explainer 实例未因阶段切换重建，输出模式从 queued-policy 切到 shadow 再切回 queued-policy；旧 epoch 的快照不得跨阶段发布

#### Scenario: Shadow Saliency 失败不触发机械臂 Fault

- **WHEN** shadow capture 或渲染连续失败并触发 saliency 禁用
- **THEN** 控制来源和机器人 phase 不变，仅记录 saliency 降级告警

## MODIFIED Requirements

### Requirement: 像素敏感度图语义（grad_x_input，既有行为记录）

`grad_x_input` 方法 SHALL 对每个相机产出像素级地图：在 Policy 控制期间，以与接受入队 chunk 相同的 batch、noise 与生成时 RTC 参数重放，解释 a_d；在 Shadow Policy 期间，以与同帧 Policy Proposal 相同的 batch、noise 与生成参数重放，显式选择 proposal 的 a₀。分数为所选策略空间动作实际维度的平方 L2 范数，地图为 `(grad * img).abs().sum(channel)`；禁止替换为人工动作或后处理电机命令。两模式均须验证完整输入梯度，包括适用的 guided leftover 路径。attention 采用同一目标索引，π0 query 另加 state token 偏移 1，其余架构遵循各自 adapter。层/步骤选择、原始预算及聚合规则沿用 add-pi-rollout-saliency。

#### Scenario: 动作值加权语义保持

- **WHEN** 指令幅度接近零（姿态过均值点）
- **THEN** 热力图整体变暗，且实现不产生告警或 NaN

#### Scenario: Shadow 回放对应同帧 Proposal

- **WHEN** Human 阶段记录一条有效 Policy Proposal 及其 Shadow Saliency
- **THEN** explainer 使用生成该 proposal 的同一 batch、noise 与 chunk 参数，且热力图标记为 shadow

#### Scenario: 非零目标索引

- **WHEN** Policy 阶段队列保留的首个动作在原块中索引为 4
- **THEN** score 取 actions[0,4]，而非 actions[0,0]；生成时 inference_delay 保持原值

#### Scenario: guided RTC 梯度完整性

- **WHEN** 微型模型以非空 leftover 做重放及有限差分比较
- **THEN** 输入梯度在声明容差内一致，截断梯度不得作为完整敏感度验收通过

### Requirement: 精确回放、降级与防护

`attention` 方法 MUST 使用与 `grad_x_input` 相同的快照与噪声注入机制：Policy 控制期间回放 MUST 对应接受入队 chunk 的 a_d，Shadow Policy 期间 MUST 对应同帧 proposal 的显式目标 a₀。attention 外层使用 `torch.no_grad`，允许 RTC 原有内部必要梯度上下文。重放一致性仅为测试/离线验收，不要求默认快照携带目标动作或执行逐帧门禁。capture 或计算单次失败 MUST 丢帧告警，连续两次 MUST 禁用旁路，禁止打断控制。非有限地图 MUST 丢帧；仅预算非有限（包括 prompt/action 等非视觉分类）MUST warn-once 并丢弃该帧预算/曲线，保留有效地图。阶段、epoch 或 View Mask revision 失效属于正常丢弃，不增加连续失败计数。

#### Scenario: 回放与直接前向一致

- **WHEN** 在同一 explainer 副本、同一设备上，比较快照回放与生成目标 executable output 或 shadow proposal 的直接前向结果
- **THEN** 测试声明 dtype、设备、compiled/eager 与 atol/rtol；同副本同设备 float32 capture 基准绝对误差为 1e-5，其他路径单独声明容差，不承诺跨设备逐位一致，也不增加运行时逐帧比较门禁

#### Scenario: capture 失败不中断 rollout

- **WHEN** capture callback 抛出异常
- **THEN** 当帧被丢弃并告警；若连续 2 次失败则 saliency 禁用；策略控制回路始终不受影响
