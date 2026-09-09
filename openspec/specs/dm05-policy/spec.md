## Purpose

以 lerobot 策略插件形式对接 DM05 HTTP 推理服务（OpenDM，POST /v1/infer），使 lerobot-rollout 可在真机双臂 SO-101 上 rollout 远端 DM05 策略。

## Requirements

### Requirement: dm05 策略可被 lerobot-rollout 加载
系统 SHALL 提供 `dm05` 策略插件（注册名 `dm05`），使其可被 `lerobot-rollout --policy.path=<dir>` 加载；该 `<dir>` 仅为包含连接配置（host/port/timeout/camera_keys）的 lerobot 格式 config 目录，MUST NOT 加载任何模型权重（权重在 DM05 服务端）。

#### Scenario: config-only 目录加载
- **WHEN** 用户执行 `lerobot-rollout --strategy.type=base --policy.path=<dm05-config-dir> --inference.type=sync ...`
- **THEN** rollout 上下文成功构建，策略类型显示为 `dm05`，进程不加载 safetensors 权重

#### Scenario: 缺少 type 字段报错
- **WHEN** `<dm05-config-dir>/config.json` 缺失或 `type` 不是 `dm05`
- **THEN** 加载阶段抛出包含路径与期望字段的明确错误，机器人不被连接

### Requirement: 服务不可达时 fail-fast
策略初始化时 MUST 探测 DM05 服务 TCP 可达性，不可达时 MUST 在 rollout 上下文构建阶段抛错退出；由于上下文构建是策略-first、硬件-last 顺序，失败时 MUST NOT 触碰任何机器人硬件。

#### Scenario: 服务未启动
- **WHEN** config 指定的 host:port 在超时内无法建立 TCP 连接
- **THEN** 上下文构建抛出含服务地址与启动提示（deploy.sh）的 RuntimeError，进程在任何 `robot.connect()` 之前退出

### Requirement: 观测编码为 /v1/infer 请求体
`DM05Policy.select_action` SHALL 把预处理后的观测编码为 `/v1/infer` JSON：`observation.state` 为 12 维 percent 单位浮点数组（关节顺序由 `DM05Config.action_feature_names` 显式声明，经 rollout 上下文的特征对齐机制与硬件顺序联动——DM05 训练顺序与 `bi_so_follower` 输出不一致时由该字段纠正，禁止静默错位）；`observation.images` 的键 MUST 为 `camera_keys` 配置的槽位名（默认 `"1"/"2"/"3"`），值 MUST 为 base64 JPEG（uint8 HWC RGB 编码），顺序与部署时 `--inference-config.image-prompts`（Head/Left wrist/Right wrist）一一对应；`observation.prompt` 为当前任务文本。

#### Scenario: 请求体字段正确
- **WHEN** 一次推理发生
- **THEN** POST 请求体中 `state` 长度为 12、`images` 恰含 `camera_keys` 配置的全部槽位键且值为可解码 base64 JPEG、`prompt` 等于当前任务字符串

#### Scenario: 相机键缺失时报错
- **WHEN** camera_keys 映射的任一相机键不在预处理后的观测里
- **THEN** 抛出包含缺失键与可用键列表的错误，控制环停止

### Requirement: chunk 截断与逐 tick 执行
`DM05Config` SHALL 提供 `execution_horizon` 字段（None 或正整数，默认 None 表示整段执行）。`DM05Policy.select_action` SHALL 把服务返回的 `actions`（形状 (chunk_size, 12)）按 `execution_horizon` 截断后入队（None 或超过实际长度时整段），并在后续 tick 逐帧返回（不重复请求服务），队列耗尽时再以最新观测请求新 chunk。动作队列属性 MUST 命名为 `_action_queue`（基类 `drop_queued_actions` 的默认清理契约只认该名与 `_queues`）。`reset()` MUST 清空队列；`drop_queued_actions()` MUST 丢弃未消费的 chunk 帧（供 interactive 任务切换使用）。

#### Scenario: 一次请求服务多个 tick
- **WHEN** 服务返回 (50, 12) 的 actions、`execution_horizon` 为 None 且控制环以 fps 运行
- **THEN** 第 1 次 select_action 触发 1 次 HTTP 请求，其后 49 次 select_action 不发请求、依次返回第 2..50 帧，第 51 次触发新请求

#### Scenario: 截断后段
- **WHEN** 服务返回 (50, 12) 的 actions 且 `execution_horizon=25`
- **THEN** 第 1 次 select_action 触发 1 次 HTTP 请求，其后 24 次 select_action 依次返回第 2..25 帧，后 25 帧被丢弃不下发，第 26 次触发新请求

#### Scenario: 返回维度不符时报错
- **WHEN** 服务返回的 actions 第二维不等于 12
- **THEN** 抛出包含实际与期望维度的错误，控制环停止

#### Scenario: 任务切换丢弃旧 chunk
- **WHEN** `drop_queued_actions()` 被调用后再次 select_action
- **THEN** 立即以当前观测请求新 chunk，旧 chunk 的剩余帧不再下发

### Requirement: 可选 history_images 队列
`DM05Config` SHALL 提供 `use_history` 开关。开启时，策略 MUST 维护一个主相机（`camera_keys` 中槽位 `"1"` 对应的相机）帧的 deque（maxlen=5），语义为：每次请求新 chunk 时，请求体携带 `observation.history_images` = deque 当前内容（旧→新，不含本次新帧），发送后再把本次主相机帧 append 入队——正常稳态下 history 帧间隔等于 `execution_horizon`（实际执行的帧数）；`drop_queued_actions`（任务切换）触发的提前请求会使该次间隔短于配置值，属已接受的偏差（见 design 风险节）。`reset()` MUST 清空 deque（复现训练侧 episode 开头历史变短的分布）。关闭时（默认）MUST NOT 在请求体中携带 `history_images` 字段（旧 checkpoint 的服务端未开 `--data-config.is-history`，收到该字段会 400 报错）。

#### Scenario: 开启 history 时请求携带历史帧
- **WHEN** `use_history=true` 且第 N 次（N>1）请求新 chunk，且前 N-1 次均为队列耗尽后的正常请求
- **THEN** 请求体 `observation.history_images` 为 min(N-1, 5) 帧 base64 图像、顺序旧→新、均来自主相机，不含本次当前帧；相邻历史帧的间隔为实际执行的帧数（如 `execution_horizon=25` 时隔 25 帧）

#### Scenario: episode 开头历史为空
- **WHEN** `use_history=true` 且策略刚经过 reset 后首次请求 chunk
- **THEN** 请求体 `observation.history_images` 为空数组 `[]`

#### Scenario: 关闭 history 时不携带字段
- **WHEN** `use_history=false`（默认）
- **THEN** 请求体不含 `history_images` 键，与未开 is-history 的服务端兼容

### Requirement: 网络故障时安全停机
HTTP 请求失败（断连/超时/非 200）时 `select_action` SHALL 抛出 RuntimeError（含服务地址、HTTP 状态码、响应体摘要），控制环 MUST 终止（远端策略失联时继续执行旧动作比停机更危险）；机器人断开由 rollout 引擎的既有 teardown 保证。

#### Scenario: 服务中途断连
- **WHEN** 控制环运行中 DM05 服务进程退出，下一次 chunk 请求连接失败
- **THEN** select_action 抛出含服务地址的 RuntimeError，控制环停止下发动作，机器人被安全断开

#### Scenario: 服务返回错误状态码
- **WHEN** 服务返回 HTTP 500（如服务端 OOM）
- **THEN** 抛出含状态码与响应体摘要的 RuntimeError，控制环停止

### Requirement: dm05 processor 管线近似恒等
dm05 策略的 pre/post processor SHALL 为近似恒等管线（不含归一化，归一化由服务端完成），且 MUST 复用通用 `rename_observations_processor`（`--rename_map`）完成相机键名映射。

#### Scenario: rename_map 生效
- **WHEN** 用户传入 `--rename_map='{"observation.images.top": "observation.images.camera1"}'`
- **THEN** 预处理后观测按 rename 后的键进入策略，策略按 camera_keys 取图并发往服务端

### Requirement: rollout_dm05.sh 启动脚本
仓库根目录 SHALL 提供 `rollout_dm05.sh`，以 `lerobot-rollout --strategy.type=base --policy.path=<dm05-config-dir> --inference.type=sync` 驱动 DM05 服务 rollout，MUST 显式传 `use_degrees=false`（percent 单位），并保留串口检查/赋权与相机配置。

#### Scenario: 脚本一键启动
- **WHEN** DM05 服务已启动且串口相机就绪时执行 `bash rollout_dm05.sh`
- **THEN** rollout 正常运行，日志显示策略类型 `dm05`、推理后端 `sync`