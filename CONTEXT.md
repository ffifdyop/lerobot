# LeRobot（lehome 分支）

真机操作技术栈：遥操作、数据录制、策略训练与策略上机 rollout（SO-101 级机械臂、三相机）。本词汇表固定 rollout、推理与可视化相关的项目用语。

## 术语表

### Rollout

**Rollout**：
在真实机器人上现场运行已训练的策略，观察其行为。
_避免_：评测（评测隐含指标对比）、部署

**Rollout Run（一次 Rollout 运行）**：
一次 rollout 策略的完整执行，从启动到停止。base 模式下没有 episode 概念——这次运行就是 rollout 期间产生的一切（视频、日志）的天然边界。
_避免_：episode（仅对录制类策略才有意义）

**Policy Tick（策略节拍）**：
一次策略推理：观测进、动作块出。

**Control Tick（控制节拍）**：
控制循环的一步（在动作块内做插值）。较慢的策略节拍驱动较快的控制节拍。

**Control Source（控制来源）**：
当前拥有机器人动作决定权的一方，只能是 Policy 或 Human；同一控制节拍不得混合两个来源。
_避免_：控制模式（容易与机器人的关节/末端执行器控制模式混淆）

**Control Phase（控制阶段）**：
一次 rollout 运行中控制权所处的阶段：Policy、Human、Resuming、Paused 或 Fault。阶段描述运行状态，Control Source 描述有效动作的来源，两者不可互换。

### 人工接管

**Human Takeover（人工接管）**：
rollout 不结束的前提下，Control Source 从 Policy 切换到 Human，使主臂动作取代策略动作成为机器人指令。
_避免_：急停、混合控制

**Leader–Follower Synchronization（主从臂同步）**：
Policy 控制期间让主臂跟随从臂实际姿态的关系，使人工接管开始时两者已经对齐。它不是人工控制，也不表示主臂动作正在驱动从臂。
_避免_：遥操作、主臂接管

**Policy Proposal（策略候选动作）**：
策略针对当前观测提出、但尚未获得执行权的动作。Human、Paused 或 Fault 阶段产生的 Policy Proposal 不得进入机器人可执行动作队列。
_避免_：Policy Action（无法区分候选动作与已选中的策略动作）

**Selected Action（选中动作）**：
控制仲裁器在 Policy Proposal 与人工动作之间选出的训练动作；它位于策略动作空间，尚未包含机器人侧安全裁剪或变换。
_避免_：Executed Action

**Executed Action（实际发送动作）**：
机器人动作处理和安全裁剪之后，最终发送到机器人接口的动作。它可能与 Selected Action 数值或动作空间不同。
_避免_：Selected Action、Policy Proposal

**Shadow Policy（影子策略）**：
在 Policy 不拥有控制权时继续观察并产生 Policy Proposal、但永远不向机器人发送动作的策略运行方式。
_避免_：并行控制、备用控制器

**Rollout Fault（Rollout 故障）**：
控制完整性无法继续保证、必须停止产生新机器人动作的锁存状态。故障可分为允许人工确认后恢复的 Recoverable Fault，以及必须结束当前 episode 的 Fatal Fault。
_避免_：Pause（主动暂停不是故障）、Warning

### 策略与推理

**View Ablation（视角消融）**：
在 `lerobot-rollout` 中刻意屏蔽任意数量（包括全部）的相机视角，用于比较缺少指定视觉信息时的策略行为。动作推理、VQA 和 saliency 必须共享同一消融条件；它不扩展到 eval 或 record，也不表示相机发生故障。本概念以策略至少具有一个真实视觉输入为前提，无视觉策略不在其讨论范围内。
_避免_：相机故障、丢帧

**View Mask（视角遮罩）**：
一次 rollout 运行中对策略输入的指定相机视角施加的像素遮罩；遮罩的视角集合和空间范围共同构成消融条件。当前标准控制面使用整幅全零遮罩，但全图只是一种空间范围；相机原始采集、人工监看和记录始终保留原图。
_避免_：Camera Failure（相机仍可正常采集）、Display Mask（不表示对操作者隐藏画面）

**Canonical View Key（规范视角键）**：
应用视角名称映射后、策略用于区分相机视角的稳定名称。操作者可用原始相机名或规范视角键选择视角，但有效 View Mask 和所有记录一律使用无歧义的规范视角键。
_避免_：Camera Index（顺序不是视角身份）、模糊名称匹配

**Mask Transition（遮罩切换）**：
运行中 View Mask 改变的边界。同一控制循环内收到的遮罩变更合并为一次原子切换；切换过程中继续收到的变更采用 latest-wins，并使较早版本的推理结果失效。切换会废弃旧遮罩条件下尚未执行的策略动作和策略视觉历史，但保留当前任务文本，并等待基于最新遮罩生成的 fresh policy chunk。
_避免_：Camera Recovery（相机并未恢复）、自然延迟生效

**View Mask Revision（视角遮罩版本）**：
一次有效 View Mask 状态的单调递增标识，与该状态产生的策略候选动作绑定。启动状态为 revision 0；此后每次有效变更（包括会话复位恢复启动遮罩）都产生更大的新版本，旧编号不得复用。它使逐帧记录和切换事件能够准确说明动作是在何种消融条件下生成的。
_避免_：录帧时临时读取的当前遮罩状态

**View Ablation Record（视角消融记录）**：
把有效 View Mask、revision 和切换事件与 rollout 时间线关联起来的实验记录。Policy 与 Human 产生的训练帧都携带当时的消融条件；动作来源由 Control Source 单独说明。
_避免_：仅有可视化状态、把 Human 帧写成未遮罩

**View Mask Control（视角遮罩控制）**：
操作者在 rollout 运行中改变 View Mask 的控制面。每个视角使用锁存切换语义：一次独立按键将其加入遮罩集合，再按一次将其移除，长按不得重复触发。它在会话存活期间始终可用，包括策略、人工、暂停和尚未开始运动的阶段；非策略阶段的变更不能打断人工控制。可视化只报告当前状态而不拥有控制权；非法或有歧义的视角请求整体拒绝，不能部分生效或自动猜测。
_避免_：可视化控制、部分成功的遮罩更新

**Mask Application State（遮罩应用状态）**：
当前 View Mask Revision 与机器人可执行策略动作之间的就绪关系。`APPLYING` 表示正在等待该版本的 fresh policy chunk，期间不产生训练帧、VQA 或自动任务规划；`ACTIVE` 表示该版本已可产生动作；`FAILED` 表示 fresh inference 报错或超时，策略动作保持停用且不产生训练帧。失败状态锁存，晚到结果失效，不得自动回滚到旧版本或自动恢复策略控制；若 Human 已拥有控制权，人工控制可继续。非交互式 rollout 进入 `FAILED` 后结束当前运行。
_避免_：Control Phase（它描述控制权，而不是遮罩版本是否就绪）

**Inference Flush（推理清空）**：
Mask Transition 时废弃旧遮罩条件下的待执行动作和策略视觉历史，同时保留当前任务文本。它只清理推理状态，不移动机器人，也不让机器人回到初始姿态。
_避免_：Robot Reset（机器人复位）、Session Reset（会话复位）

**Next Queued Policy Action（下一入队策略动作）**：
新动作块被 RTC 接受后，扣除推理延迟期间已过时的前 `d` 项，保留下来的第一个策略空间动作 `a_d`。它对应下一个可执行入队动作，但本身尚未经机器人侧后处理或安全裁剪。
_避免_：First Chunk Action（RTC 下的 `a₀` 可能因推理延迟被丢弃）、Executed Action

**Local Policy（本地策略）**：
网络在本进程内运行的策略，其内部结构（梯度、attention）可直接检查。SmolVLA 是本地策略。
_避免_：离线策略

**Remote Policy（远程策略）**：
经网络（HTTP/websocket）服务的策略；rollout 进程只能看到观测 → 动作对，永远接触不到网络内部。DM05 与 openpi 策略是远程策略。

**Observation Memory（MEM 观测记忆）**：
π0.5 的可选扩展，使当前 Policy Proposal 可依赖抽样的历史相机观测和/或本体状态。它不是基础 π0.5 的默认语义。
_避免_：π0.5 历史（容易误解为所有 π0.5 都含历史观测）

### 显著性可视化

**Saliency Map（热力图）**：
每个相机一张的重要性图；Policy 控制期间，表示本地策略的 Next Queued Policy Action `a_d` 与产生它的输入之间的关系，被 RTC 拒绝的动作块不作为解释目标。Policy 不拥有控制权时，只能将该图解释为 Shadow Saliency；旁路处理允许跳过观测，不保证每个策略节拍都有图。受 View Mask 影响的视角仍保留模型对遮罩后图像实际产生的 saliency，不得人为清零或隐藏。

**Shadow Saliency（影子热力图）**：
Shadow Policy 对当前观测产生的反事实热力图，表示“若此刻由策略控制，它会关注哪里”。它不解释人工动作，也不表示图中的 Policy Proposal 已经或即将被执行。
_避免_：人工动作热力图、Executed-Action Saliency

**Pixel Sensitivity Map（像素敏感度图）**：
热力图的第一种方法（`grad_x_input`）：像素级，表示每个输入像素对 Next Queued Policy Action `a_d` 的贡献量。"贡献"的精确定义（2026-09-11 确认）：该像素放大/缩小当前指令幅度的能力，即 ∂‖a_d‖²/∂x——因此动作值加权是有意的：指令幅度接近零（姿态过均值点）时全图变暗是正确行为，幅度大的关节主导也是正确行为。
_避免_：关节等权敏感度（曾被考虑并否决——舵机角度值加权正是所需语义）

**Action Token（动作 token）**：
模型内部对动作块中一个时间位置的整体表示，一个 token 同时承载该时间位置的全部关节和夹爪动作，不表示单个关节。
_避免_：Action Dimension（动作向量的一个分量）

**Patch Attention Map（Patch 注意力图）**：
热力图的第二种方法（`attention`）：patch 级，表示 Next Queued Policy Action `a_d` 的动作 token 对各图像 patch token 的 attention 份额。heads 始终取均值；可选择参与聚合的模型层和 denoise steps，默认使用全部可用层与全部 steps。原始份额不做重归一化——模型看非视觉上下文多时整图变暗是正确行为。不同策略的 patch 网格尺寸可不同。

**Attention Budget Bar（注意力预算条）**：
Next Queued Policy Action `a_d` 的动作 token 在各类 token 上的原始 attention 质量分配，包括各相机、语言或 prompt、独立 state（若存在）以及 action。action 段合并该 token 对自身和其他动作 token 的份额，不再细分。不丢弃 action 份额，不对上下文重归一化，总和恒等于 1。π0.5 将任务文本和离散化状态统一标为 `prompt`。预算以堆叠横条叠加在叠加热力图视频每帧顶部。

**Saliency Overlay Video（叠加热力图视频）**：
原始相机画面与其热力图（任一 method）混叠后的视频，每个相机每次 rollout 运行各一个文件，供事后检查。
_避免_：调试视频

**Explainer（解释器）**：
面向特定本地策略及解释方法的组件，将一次策略决策及其输入关联到对应的热力图。本地策略身份本身不意味着已经具有可用的解释器。
_避免_：可视化器（专指 rerun/foxglove 显示后端）
