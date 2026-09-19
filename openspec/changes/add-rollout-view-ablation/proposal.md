## Why

真实机器人 rollout 目前无法在保持相机采集、显示与原始数据完整的同时，受控地移除指定视角对策略的影响。为比较策略对各视角的依赖，需要一个跨本地/远程策略、可在运行中切换且能准确记录实验条件的视角消融能力。

## What Changes

- 为 `lerobot-rollout` 增加显式启用的 View Ablation 配置：CLI 指定初始视角，F1/F2/F3 等单键在运行中锁存切换。
- 将指定 canonical view 的策略输入替换为同形状全零图，保留相机 key、槽位、原始采集、显示与数据集图像。
- 在 snapshot 与 processor 之间定义可扩展的空间 mask operator；当前整幅黑图作为默认实现，未来不同形状、尺寸的局部 mask 无需修改推理与 processor 边界。
- 为 Sync/RTC 建立 revision 化的原子 Mask Transition：latest-wins，清空旧动作、插值与视觉历史，等待 fresh chunk，并提供 `APPLYING/ACTIVE/FAILED` 状态及默认 10 秒超时。
- 让动作推理、VQA、自动任务规划与 saliency 共享同一 mask；过渡期暂停文本推理，被屏蔽视角保留真实 saliency。
- 数据帧记录具名 `view_mask` 与 `view_mask_revision`；切换事件写入可对齐 rollout/dataset 时间线的 JSONL；过渡或失败帧不进入训练数据。
- Rerun 保留原始相机画面并标注 `MASKED`、状态和 revision；控制仍由跨平台键盘输入负责，不由 Viewer UI 负责。
- 对非法/歧义视角、热键冲突、未显式启用的子配置及不兼容的续写数据集执行启动期校验。

## Non-goals

- 不实现模型 attention/token 层的缺失视角协议，不删除 observation key，也不停止物理相机。
- 不扩展到 `lerobot-eval` 或 `lerobot-record`，不为 Rerun 开发 Rust 自定义控制面板。
- 不定义无视觉策略或事件记录器自身故障的新增行为。
- 本 change 不规定矩形、圆形、polygon 或外部 mask bitmap 的用户配置格式，只保留实现这些空间 mask 的稳定接口。

## Capabilities

### New Capabilities

- `rollout-view-ablation`: 定义 rollout 视角选择、黑图语义、动态切换、状态机、记录、显示、校验及跨策略一致性。

### Modified Capabilities

- `rollout-saliency`: 规定 saliency 使用实际 masked policy batch，保留被屏蔽视角的真实结果，并丢弃过期 revision 的快照。

## Impact

- 影响 rollout 配置、上下文构建、键盘输入、策略 preprocessor、Sync/RTC inference、动作插值、交互控制器、各录制 strategy、Rerun/Foxglove telemetry 与 saliency 快照。
- 数据集 schema 新增逐帧 mask 字段，并增加 `meta/view_mask_events.jsonl`；旧 schema 数据集不能在启用该能力时续写。
- 不新增第三方依赖；DM05 等远程策略继续接收完整相机键，只是指定槽位的 JPEG 内容变为黑图。
