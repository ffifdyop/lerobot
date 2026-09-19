## Context

`lerobot-rollout` 的相机观测先由 robot observation processor 生成 dataset frame，再经过 `prepare_observation_for_inference`、checkpoint policy preprocessor 和 policy。policy preprocessor 的常见顺序包含 rename、batch/device、normalize；Sync 与 RTC、动作推理与 VQA 均复用该管线。原始 observation 同时供 dataset 与 telemetry 使用，因此禁止在 robot observation processor 中原地改黑图。

现有策略对“缺失相机 key”的行为不一致：ACT、Diffusion、VQBeT 与 DM05 会报错；部分 VLA 会补 dummy view，但可能把后续视角左移。DM05 远程协议也没有 image-valid mask。实现必须保留每个相机 key、shape、dtype 与槽位，只改变指定视角的像素。

RTC 已有 reset epoch、带 task provenance 的 action queue 和 saliency snapshot；Sync 已有 policy/pre/post reset 与 task 切换清队列。新能力必须扩展这些边界，而不是另建一套推理循环。当前工作树另有 human-takeover change 会修改相邻控制阶段；本设计必须以独立的 View Mask revision 与 Mask Application State 表达消融状态，禁止把它混同为 Control Phase。

## Goals / Non-Goals

**Goals:**

- 所有 `lerobot-rollout` strategy、Sync/RTC、本地与远程 policy 必须获得一致的黑图消融语义。
- 相机采集、人工监看、dataset 原图与 telemetry 必须保持原始内容；动作、VQA、自动任务规划与 saliency 必须共享实际 masked batch。
- 运行时切换必须是原子、latest-wins、可追溯的；旧 revision 的动作、history 与 saliency 禁止泄漏。
- 训练帧与事件 sidecar 必须能还原每个有效动作对应的 View Mask。
- 功能关闭时必须保持当前 rollout 行为与数据集 schema 不变。

**Non-Goals:**

- 不实现 token/attention 层的 logical missing view，不删除 observation key，不控制物理相机开关。
- 不扩展 `lerobot-eval`、`lerobot-record`，不实现 Rerun Rust 自定义控制面板。
- 不定义无视觉策略，也不改变现有记录器 I/O 故障策略。
- 不自动迁移缺少 View Ablation 字段的旧数据集。

## Decisions

### 1. 使用嵌套的显式 opt-in 配置

新增 `ViewMaskConfig`，由 `RolloutConfig.view_mask` 持有：

- `enabled: bool = False`
- `initial_views: list[str] = []`
- `hotkeys: dict[str, str] = {}`（空值表示按 canonical policy view 顺序分配 F1、F2、F3……）
- `transition_timeout_s: float = 10.0`
- `event_log_path: str | None = None`

当 `enabled=false` 时，任何非默认子配置必须 fail-fast，禁止隐式启用。启用后，context builder 必须在启动控制循环前完成以下解析：

1. 从 policy visual input features 与 robot camera features 建立 raw key → canonical key 映射；
2. 同时接受 raw camera key 与 canonical key 输入，但歧义、未知或非视觉 key 必须整次拒绝；
3. canonical view 顺序必须取 policy visual feature 的声明顺序，禁止按字符串排序改变模型槽位身份；
4. 默认热键按该顺序绑定 F1…，自定义映射必须覆盖到 canonical key；
5. 与 strategy、teleop 及已有 rollout 控制键冲突时必须报出冲突双方，禁止按注册顺序抢占；
6. 启用录制且 `--resume` 时，旧数据集必须已经具有完全一致的 View Mask feature schema 与 names，否则禁止续写。

选择显式 opt-in 而不是始终注册热键，是为了防止普通 rollout 因误按功能键改变策略输入。选择同时接受 raw/canonical 名称，是为了兼顾现场可用性与记录稳定性；内部和持久化数据必须只使用 canonical key。

### 2. 在 rename 后、normalization 前执行可扩展的空间 mask

新增运行时 `ViewMaskProcessorStep`，插入 checkpoint preprocessor 的 rename step 之后、任何 normalization 或 policy-specific image transform 之前。推理引擎必须把不可变的 `{revision, mask_operator}` snapshot 作为保留的运行时 metadata 随单次 observation 送入 preprocessor；mask step 应用后必须移除该 metadata，禁止传入 policy。

snapshot 中的 operator 必须实现统一的 observation 变换接口，声明受影响的 canonical views，并提供 JSON-serializable metadata 描述。当前默认实现 `FullFrameViewMask` 对每个目标 tensor 使用 `zeros_like`，使整幅黑图成为空间 mask 的一个特例。operator 必须返回派生 mapping，禁止删除 key、改变 dtype/shape/device、重排槽位或修改调用者持有的原始图像。未来矩形、圆形、polygon 或 bitmap operator 可以只替换目标 tensor 的局部像素，而无需修改 processor、Sync/RTC、VQA 或 saliency 路径；其 metadata 通过 revision sidecar 记录，逐帧 dataset 仍以 bool 向量表示哪些视角受到 mask。

`ViewMaskController` 接受一个 operator factory；默认 factory 根据当前锁存的 view 集合创建 immutable `FullFrameViewMask`。本 change 不公开局部 geometry 的 CLI schema，也不实现具体局部形状。后续能力只需提供新的 immutable operator/factory，并继续通过同一 revision snapshot 传播。

选择 policy preprocessor 而不是 robot observation processor，可使 dataset、Rerun/Foxglove 与相机诊断继续读取原图。选择像素 mask 而不是删除 key，是因为后者会破坏固定 schema 策略，并在部分 VLA 中造成 view slot 左移。该步骤必须是 rollout 运行时注入项，禁止写回 checkpoint processor 配置。

### 3. 以独立 controller 管理 mask、revision 与应用状态

新增线程安全 `ViewMaskController`，持有：

- launch mask、target mask 与单调递增 revision（启动为 0）；
- `APPLYING / ACTIVE / FAILED`；
- 待合并的 key-edge toggles；
- 当前 transition deadline 与失败原因；
- 每个 revision 的 immutable snapshot。

键盘后端必须复用 `create_key_listener`。按键采用 key-down edge 锁存语义：一次加入、再次移除，长按自动重复必须被抑制到 key-up 后。控制循环每 tick 原子消费全部 pending toggles，只对最终集合执行一次 transition；无实际集合变化时禁止增加 revision。

一次有效 transition 必须：

1. revision 加一、状态进入 `APPLYING`、记录 old/new mask 与 source；
2. 触发 Inference Flush，清 policy、pre/postprocessor、action queue、插值缓存和视觉/history 状态，但保留 task；
3. 使已开始但未完成的旧 revision inference、action chunk 与 saliency snapshot 失效；
4. 保持最后一条已发送机器人动作，禁止发送旧 queue 中的新动作；
5. 第一个匹配最新 revision 的 fresh policy action 到达时进入 `ACTIVE`。

`APPLYING` 期间的新 toggles 必须 latest-wins：产生更高 revision，再次 flush；任何较早结果即使随后返回也必须丢弃。10 秒 timeout 只在 policy inference 实际有机会运行时累计；尚未 `/start`、Human 或 Paused 且推理被暂停时必须暂停 deadline，禁止因操作者提前设置 mask 而假失败。若 shadow inference 正在运行，则可正常完成 transition。

普通 episode 和 `/start` 必须保留当前 mask。显式 `/reset` 若当前集合不同于 launch mask，必须产生一个新的 revision 并恢复 launch mask，禁止把 revision 倒回 0。

### 4. Sync 与 RTC 都必须携带 revision provenance

InferenceEngine 的结果 provenance 必须扩展为 task + mask revision。禁止由录帧线程临时读取 controller 当前值来给动作贴标签。

- **Sync**：一次推理开始时捕获 mask snapshot；preprocessor 使用该 snapshot。policy 返回后、动作进入 interpolator/robot dispatch 前必须重新校验 revision；若已过期则丢弃结果。首次返回最新 revision 的动作时标记 `ACTIVE`。
- **RTC**：observation holder 必须保存 observation 与 mask snapshot；reset epoch 与 mask revision 都必须参与 merge guard。ActionQueue 条目必须绑定 task 与 mask revision；旧 revision chunk 禁止 merge 或 pop。`reset()` 必须继续以原子方式清 queue、清 observation、增加 epoch。

action interpolator 必须能在 transition 时清除尚未发送的插值序列，同时保存最后一条已实际发送动作作为 hold 值。选择 revision provenance 而不只依赖 queue clear，是为了覆盖“按键发生在 inference 中途”及远程请求无法取消的竞态。

`APPLYING` 时必须拒绝新 VQA，并暂停 autosteer；`FAILED` 时二者保持不可用。VQA 恢复后必须用最新 mask snapshot。默认 10 秒内未得到 fresh action或 inference 明确报错时进入锁存 `FAILED`：晚到结果必须丢弃，禁止自动回滚或重试。非交互式 rollout 必须结束；交互式 rollout 只能通过显式 `/reset` 恢复。Human 已拥有控制权时，FAILED 禁止夺走人工控制，但 Policy 不得恢复下发。

### 5. 记录 action provenance，而不是 UI 瞬时状态

启用 View Ablation 且存在 dataset 时，创建或验证以下 features：

- `view_mask`: `bool`, shape `(N,)`, names 为 canonical view keys；
- `view_mask_revision`: `int64`, shape `(1,)`。

Policy 训练帧必须写入 dispatched action 自带的 mask snapshot。Human 训练帧必须写入当时 controller snapshot，并继续由 `intervention/control_source` 表明动作来源；该 mask 描述同一观测上的 Shadow Policy 条件。`APPLYING` 与 `FAILED` 禁止产生训练帧。

每次启动、有效 transition、状态变化与失败必须追加 JSONL 事件，至少包含 run-relative monotonic time、revision、old/new mask、application state、source、control phase（可用时）以及 dataset episode/最近 frame index（可用时）。文件头只记录一次 UTC run start。默认路径：有 dataset 时为 `<dataset.root>/meta/view_mask_events.jsonl`；无 dataset 时为 `outputs/rollout/<timestamp>/view_mask_events.jsonl`；显式路径覆盖默认值。事件记录必须复用现有记录系统的故障处理，禁止为本 change 发明新的机器人控制语义。

### 6. 可视化报告状态，但不拥有控制权

Rerun 必须继续记录原始相机实体；另以 display-only annotation layer 或显示副本在每个 masked camera tile 上显示红色 `MASKED`，并记录每视角 0/1、revision 与 `APPLYING/ACTIVE/FAILED`。annotation 禁止回流至 dataset 或 policy。全部视角 masked 与普通多视角 mask 使用相同行为，不增加确认或特殊告警。

Foxglove 或关闭可视化时，mask、热键、记录与推理语义必须继续工作；Foxglove 应发布可表达的等价状态，终端必须打印 mapping、transition 与失败。Viewer UI 禁止作为控制输入；X11/trusted macOS/Windows 使用全局 pynput，Wayland/headless 沿用 terminal fallback。

### 7. Saliency 使用已 masked batch并受 revision 隔离

saliency snapshot 必须在 `ViewMaskProcessorStep` 之后捕获，因此 explainer 回放与实际策略输入一致。被 mask 视角仍是有效黑图 token，grad/attention 结果可能非零；recorder 必须保留真实结果并标注 `MASKED`，禁止人为清零或隐藏。

snapshot 必须携带 mask revision。transition 后，旧 revision snapshot 即使已排入 recorder 队列也必须在回放/发布前丢弃。`APPLYING/FAILED` 不得为未执行动作发布新的 saliency；saliency 自身失败仍沿用既有降级规则，禁止影响控制回路。

## Risks / Trade-offs

- [黑图不是 logical missing view，模型仍可能关注黑图 token] → 文档和字段统一称为 View Ablation/black-view mask；saliency 保留真实非零结果，禁止宣称 attention 已屏蔽。
- [transition 清空 history 会带来短暂停顿] → 使用 `APPLYING` 明示、保持最后实际动作、默认 10 秒可配置超时，禁止继续旧 queue。
- [异步或远程 inference 无法真正取消] → 以 reset epoch + revision 双重 guard 丢弃晚到结果，队列条目携带 provenance。
- [多个 keyboard listener 或策略快捷键冲突] → context build 时集中解析并 fail-fast；禁止静默覆盖。
- [Wayland 无法在 Rerun 前台捕获全局键] → 沿用 terminal fallback，并在启动日志明确当前 backend；不把 Viewer UI 扩展纳入本 change。
- [dataset schema 增加后无法无损续写旧 run] → enabled 模式严格校验 schema，要求创建新 dataset；disabled 模式完全不增加字段。
- [human-takeover change 修改相邻 phase/queue 代码] → mask application state 保持正交；实现时必须复用最终 Control Phase/Shadow Policy API，并补跨 phase 集成测试，禁止复制第二套仲裁状态机。

## Migration Plan

1. 先落地 disabled-by-default 配置、纯 processor step 与 unit tests，确认所有现有 rollout 测试逐位保持。
2. 接入 controller、Sync/RTC revision provenance、interpolator flush 与键盘输入，再补竞态/超时测试。
3. 接入 dataset schema、JSONL 与各 strategy 的录帧 gating；旧 dataset 仅在 enabled 模式拒绝 resume。
4. 最后接入 Rerun/Foxglove 与 saliency revision，进行 DM05 Sync、SmolVLA RTC 和 DAgger/Human phase 集成验证。
5. 回滚时关闭 `--view_mask.enabled` 即恢复旧行为；若完全回滚代码，新 schema dataset 仍可保留额外顶层字段，但禁止用旧代码继续录制该 run。

## Open Questions

无。实现必须遵循 specs；若 human-takeover change 先合并，仅允许适配其最终公共 phase/shadow API，不得改变本设计的 View Mask 语义。
