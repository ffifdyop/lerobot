## 1. 配置、名称解析与黑图处理

- [x] 1.1 新增默认关闭的 View Mask 配置，并实现 initial views、timeout、event path 与 disabled/non-default 组合的启动期校验
- [x] 1.2 实现 raw camera key 到 canonical policy view key 的解析、policy 声明顺序保留、非法/歧义名称整次拒绝及默认 F1/F2/F3 映射
- [x] 1.3 实现运行时 View Mask processor，在 rename 后、normalization 前按 snapshot 生成同 shape/dtype/device 的黑图且不污染原始 observation
- [x] 1.4 为配置解析和 processor 增加单元测试，覆盖全部视角、DM05 固定键、原图不变及功能关闭零行为

## 2. View Mask controller 与输入

- [x] 2.1 实现线程安全 ViewMaskController、revision 0、APPLYING/ACTIVE/FAILED、pending toggle 合并与 immutable snapshot
- [x] 2.2 扩展键盘输入以支持 key-down/key-up edge 和长按去重，并接入锁存式每视角热键
- [x] 2.3 集中校验 View Mask、strategy、teleop 与现有 rollout 快捷键冲突，并输出启动 mapping 与状态变更日志
- [x] 2.4 实现 latest-wins transition、普通 episode/segment 状态保持及显式 reset 恢复 launch mask 且 revision 不回退
- [x] 2.5 增加 controller/input 并发测试，覆盖同 tick 多键、长按、transition 中再次切换、无变化不增 revision 与热键冲突

## 3. Sync/RTC provenance 与过渡状态

- [x] 3.1 扩展 InferenceEngine 公共 provenance，使 dispatched action 同时携带 task 与 View Mask snapshot，并提供 transition flush/状态回报接口
- [x] 3.2 在 Sync engine 中绑定单次 inference snapshot、执行前复核 revision，并丢弃切换期间返回的旧本地或远程动作
- [x] 3.3 扩展 RTC observation holder、reset epoch、chunk merge guard 与 ActionQueue 条目，使 observation/chunk/action 全链路携带 revision
- [x] 3.4 接入 interpolator flush 与 hold-last-action，确保 transition 后禁止旧 queue/插值动作继续下发，fresh action 才进入 ACTIVE
- [x] 3.5 实现可运行时间累计的默认 10 秒 timeout、锁存 FAILED、晚到结果丢弃与非交互式结束流程
- [x] 3.6 在 APPLYING/FAILED 阻止 VQA 和 autosteer，并保证 Human/Paused/尚未 start 阶段暂停 timeout 且不打断人工控制
- [x] 3.7 增加 Sync 与 RTC 测试，覆盖远程请求晚到、queue merge 竞态、连续 revision、timeout 暂停/失败及 reset 后 fresh action

## 4. 数据集与事件记录

- [x] 4.1 在启用模式的新 dataset schema 中加入具名 bool `view_mask` 与 int64 `view_mask_revision`，并严格校验 resume schema/names
- [x] 4.2 修改各录制 strategy，使 Policy 帧使用 dispatched provenance、Human 帧使用当前 Shadow Policy snapshot，APPLYING/FAILED 不写训练帧
- [x] 4.3 实现 View Mask JSONL recorder、单次 UTC run header、单调相对时间与可用的 phase/episode/最近 frame 对齐字段
- [x] 4.4 实现 dataset 与 Base rollout 的默认 sidecar 路径及显式覆盖，并沿用现有记录故障处理
- [x] 4.5 增加 dataset/sidecar 测试，覆盖 episode rotation、segment 保持、Human 帧、hold ticks 排除、Base 无 dataset 与不兼容 resume

## 5. Telemetry、Rerun/Foxglove 与 saliency

- [x] 5.1 扩展 rollout telemetry，发布每视角 mask、revision 与 application state，确保 display 关闭时不影响控制能力
- [x] 5.2 在 Rerun 原始相机 tile 上增加 display-only 红色 MASKED 标识和状态时间线，并为 Foxglove 发布可表达的等价状态
- [x] 5.3 扩展 saliency snapshot 携带 masked batch、canonical views 与 revision，并在 replay/recorder 前正常丢弃过期 snapshot
- [x] 5.4 更新 saliency overlay/视频/Rerun 标识，使黑图视角保留真实非零 gradient/attention，禁止人为清零或隐藏
- [x] 5.5 增加可视化与 saliency 测试，覆盖原图不变、MASKED 标识、过期 snapshot 不计失败、APPLYING 不展示旧解释与 display backend 独立性

## 6. 集成验证与用户文档

- [x] 6.1 增加全 strategy 的 disabled-by-default 回归测试，确认现有 CLI、dataset schema、动作与显示行为不变
- [x] 6.2 增加 DM05 Sync 三视角集成测试，验证完整 camera keys、单槽黑 JPEG、切换清 history/queue 与旧响应丢弃
- [x] 6.3 增加 SmolVLA RTC 集成测试，验证动态多视角 mask、revision action provenance、VQA 与 saliency 使用同一 batch
- [x] 6.4 增加 DAgger/人工控制集成测试，验证 Human 帧标记、非策略阶段切换、FAILED 不夺权及恢复边界
- [x] 6.5 更新 rollout 用户文档与示例命令，说明 CLI 配置、F 键映射、黑图而非 logical missing-view 语义、状态与输出文件
- [x] 6.6 运行相关 rollout、interactive、DM05、processor、dataset、visualization 与 saliency 测试，并执行受影响文件的 pre-commit 检查

## 7. 空间 mask 扩展边界

- [x] 7.1 定义 immutable `ViewMaskOperator` 接口与 `FullFrameViewMask` 默认实现，使整幅黑图成为空间 mask 的特例
- [x] 7.2 让 snapshot/controller 通过 operator factory 携带 mask，并让 processor 只依赖 operator 接口，同时保持现有 masked_views、CLI、dataset 与 telemetry 兼容
- [x] 7.3 增加 operator/factory/processor 回归测试，验证自定义局部 operator 可接入且原 observation 不被污染，并运行相关测试与 strict validation
