# Spec: openpi-server-client-rollout

## ADDED Requirements

### Requirement: Server 连接与探活

系统 SHALL 通过 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 连接用户指定的 host 与 port（默认 port=8000）。在连接机器人之前，系统 MUST 先调用 server 元数据接口探活；探活失败时 MUST 输出包含 deploy.sh 检查提示的错误信息并以非零状态退出。

#### Scenario: server 正常运行

- **WHEN** openpi server 已通过 deploy.sh 启动在 8000 端口，且用户运行 client 脚本
- **THEN** client 打印 server 元数据并继续连接机器人

#### Scenario: server 未启动

- **WHEN** 目标 host:port 上没有 openpi server 监听
- **THEN** client 输出明确错误提示（含检查 deploy.sh / host / port），且不会连接或使能机器人电机

### Requirement: 观测组织与发送

openpi 的 RepackTransform 只在训练 dataloader 中执行，推理时不生效；因此系统 SHALL 将机器人观测组织为 repack 之后的 dict 发送给 server，包含 `images`（dict，键为 `cam_high`/`cam_left_wrist`/`cam_right_wrist`）、`state`、`prompt` 三个键。图像 MUST 为 uint8 (H, W, C) 格式原样发送；`state` MUST 为 float32 的 12 维向量，顺序为左臂 6 关节（按左臂电机顺序）后接右臂 6 关节。相机源键名 SHALL 可通过 CLI 参数覆盖，默认值 MUST 为 `cam_high`←`left_top`、`cam_left_wrist`←`left_wrist`、`cam_right_wrist`←`right_wrist`。

#### Scenario: 默认键映射

- **WHEN** 使用默认配置运行，且机器人观测包含 `left_top`、`left_wrist`、`right_wrist` 三帧图像
- **THEN** 发送给 server 的 dict 中 `images` 包含 `cam_high/cam_left_wrist/cam_right_wrist` 三个 uint8 (H,W,C) 图像，且 `state` 为 float32 (12,)

#### Scenario: 关节顺序可验证

- **WHEN** client 从机器人读到 12 个关节位置
- **THEN** `state` 的前 6 维依次等于左臂 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper 的读数，后 6 维为右臂同序读数

### Requirement: 动作 chunk 执行

系统 SHALL 维护一个动作队列存放 server 返回的 action chunk。每个控制周期 MUST 从队列头部取出一个 12 维动作并拆分为左臂前 6 维、右臂后 6 维下发；队列耗尽时 MUST 先采集新观测并请求 server 获得新 chunk 后再继续。系统 MUST NOT 乱序执行或跨 chunk 混合动作。

#### Scenario: chunk 内顺序执行

- **WHEN** server 返回形状为 (10, 12) 的动作 chunk
- **THEN** 接下来的 10 个控制周期依次执行该 chunk 的第 0 至第 9 帧动作，每帧前 6 维发至左臂、后 6 维发至右臂

#### Scenario: chunk 耗尽后重新请求

- **WHEN** 动作队列已空且控制周期到达
- **THEN** client 采集当前观测、向 server 请求新 chunk，并在获得后继续执行

### Requirement: 控制频率与运行时长

系统 SHALL 以用户指定的 FPS（默认 30）运行控制循环，并 SHALL 支持可选的运行时长上限（秒）；到达上限后 MUST 停止发送动作并退出循环。

#### Scenario: 按时长退出

- **WHEN** 用户指定 duration=60 并启动 rollout
- **THEN** 约 60 秒后控制循环结束，client 断开机器人并正常退出

### Requirement: 安全断开

无论正常结束、键盘中断还是异常，系统 MUST 保证调用 `robot.disconnect()`；键盘中断时 MUST 立即停止发送新动作。

#### Scenario: Ctrl-C 中断

- **WHEN** rollout 运行中用户按下 Ctrl-C
- **THEN** client 停止发送动作、断开双臂与相机连接，进程退出，电机按机器人配置失能

#### Scenario: 推理异常

- **WHEN** server 在推理请求中途断开连接或抛出异常
- **THEN** client 不再发送任何动作，执行机器人断开后以非零状态退出

### Requirement: 可选依赖隔离

系统 MUST 惰性导入 `openpi-client` 包；未安装时 MUST 输出包含安装命令的明确错误提示。`openpi-client` MUST NOT 出现在 pyproject.toml 的强制依赖中。

#### Scenario: 未安装 openpi-client

- **WHEN** 环境中没有安装 `openpi-client` 且用户运行 client 脚本
- **THEN** 输出包含 `uv pip install <openpi>/packages/openpi-client` 安装提示的错误信息并退出，机器人不会被连接
