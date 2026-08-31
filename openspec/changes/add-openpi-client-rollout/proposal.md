# Proposal: add-openpi-client-rollout

## Why

当前在双臂 SO-101 实物上部署策略只能使用 `lerobot-rollout` 做本地推理。但策略训练在 openpi 仓库完成（JAX 格式 checkpoint，如 `pi05_so101_folding`），其推理 Server 通过 `~/Dev/Manipulation/openpi/deploy.sh` 启动（`serve_policy.py`，websocket + msgpack 协议）。本仓库自带的 `lerobot.async_inference` 是 gRPC 协议，与 openpi server 不兼容，无法复用。因此需要一个跑在机器人侧、连接 openpi server 的 client 脚本，实现 server-client 方式的策略部署。

## What Changes

- 新增一个 rollout client 脚本（Python）：使用 lerobot 的 `bi_so_follower` 机器人接口采集观测（三路相机 + 12 维关节状态），通过 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 向 openpi server 请求动作，并将返回的 action chunk 按关节顺序下发到双臂。
- 新增启动 shell 脚本 `rollout_openpi.sh`：复用现有 `rollout.sh` 的串口/相机配置风格，增加 server host/port 参数。
- 观测键按 openpi 推理管线的约定组织（`images: {cam_high, cam_left_wrist, cam_right_wrist}`、`state`、`prompt`——RepackTransform 仅训练时生效，推理由 client 发送 repack 后的键），由 server 端完成归一化。
- 相机命名映射：左臂 `top`→`cam_high`、左臂 `wrist`→`cam_left_wrist`、右臂 `wrist`→`cam_right_wrist`。

## Capabilities

### New Capabilities

- `openpi-server-client-rollout`: 双臂 SO-101 通过 websocket 连接 openpi 推理 server 进行闭环策略部署，包括观测采集与组织、动作请求与分块执行、运行时长/频率控制与安全退出。

### Modified Capabilities

（无）

## Non-goals

- 不修改 `lerobot.async_inference`（gRPC 方案），两者并存、互不影响。
- 不做 openpi checkpoint 到 lerobot 格式的转换。
- 不在 client 端做动作后处理（如归一化/反归一化），该逻辑由 openpi server 端 transforms 负责。
- 不实现 RTC（real-time chunking）等推理加速策略；首版采用简单的 action chunk 顺序执行。

## Impact

- **新增文件**：`src/lerobot/scripts/lerobot_openpi_rollout.py`（或等价位置）、`rollout_openpi.sh`。
- **依赖**：需要 `openpi-client` Python 包（websocket client，纯 Python，依赖轻）；仅在运行该脚本时需要，不作为 lerobot 的强制依赖（guard import）。
- **不影响**：现有 `lerobot-rollout`、`lerobot-record`、训练流程与数据集代码均不受影响。
