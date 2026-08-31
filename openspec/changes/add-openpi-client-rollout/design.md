# Design: add-openpi-client-rollout

## Context

策略训练在 openpi 仓库（JAX checkpoint），推理 Server 由 `~/Dev/Manipulation/openpi/deploy.sh` 启动（`serve_policy.py`，websocket + msgpack 协议，端口 8000）。lerobot 自带的 `async_inference` 是 gRPC 协议，两边互不兼容，且 checkpoint 为 openpi 格式，无法用 lerobot 的 PolicyServer 加载。

机器人侧已具备：`bi_so_follower` 机器人（[bi_so_follower.py](../../../../src/lerobot/robots/bi_so_follower/bi_so_follower.py)），观测键为 `left_<关节>.pos` / `right_<关节>.pos` 与按臂加前缀的相机键（如 `left_top`、`left_wrist`、`right_wrist`），动作键与电机键同构。openpi server 端由 `LeRobotSO101BimanualDataConfig` + `SO101Inputs/SO101Outputs` 负责 repack、图像解析与动作切片，client 只需发送原始键。

## Goals / Non-Goals

**Goals:**

- 在双臂 SO-101 实物上以固定频率闭环运行：采观测 → 请求 openpi server → 执行 action chunk。
- 复用 `rollout.sh` 的串口/相机配置方式，提供一个等价的一键启动脚本 `rollout_openpi.sh`。
- 依赖最小化：client 仅额外依赖 `openpi-client` 包。

**Non-Goals:**

- 不改动 `lerobot.async_inference` 与现有 `lerobot-rollout`。
- 不做 RTC / 动作平滑 / 异步推理流水线；首版为同步阻塞式 chunk 执行。
- 不在 client 端做归一化、图像 resize 之外的数据变换（归一化全部由 server 端 transforms 完成）。

## Decisions

### 决策 1：通信层直接复用 `openpi-client`

client 端**必须**使用 `openpi_client.websocket_client_policy.WebsocketClientPolicy`（pip 包，来自 `~/Dev/Manipulation/openpi/packages/openpi-client`），**禁止**在 lerobot 里重新实现 websocket + msgpack 协议。

备选：把 openpi checkpoint 转成 lerobot 格式后走 `async_inference`——被否，引入额外的权重转换与验证成本，且与 openpi 训练工作流脱节。

### 决策 2：代码位置与入口

新增 `src/lerobot/scripts/lerobot_openpi_rollout.py`，用 draccus dataclass 做 CLI 配置，并在 `pyproject.toml [project.scripts]` 注册 `lerobot-openpi-rollout`，与仓库其他 CLI（`lerobot-rollout` 等）保持一致。配套新增根目录 `rollout_openpi.sh`。

### 决策 3：观测组织约定（与 server 端推理管线对齐）

openpi 的 `RepackTransform` **只在训练 dataloader 中执行**；推理时 `Policy.infer` 只应用 `data_transforms` 与 `model_transforms`。因此 client **必须**直接发送 repack 之后的键（即 `SO101Inputs` 的输入约定）：

- `images`: dict，`cam_high` ← 机器人观测 `left_top`、`cam_left_wrist` ← `left_wrist`、`cam_right_wrist` ← `right_wrist`（uint8, HWC）
- `state`: 按 `left_arm` 电机顺序拼接左臂 6 关节、再按同序拼接右臂 6 关节的 `np.float32[12]`
- `prompt`: CLI `--task`

图像**必须**以 uint8 (H, W, C) 原样发送（server 端 `_parse_image` 负责转 CHW 与归一化）；client **禁止**自行归一化或转 CHW。相机源键名通过 CLI 映射参数可配，默认值如上。

### 决策 4：动作 chunk 同步执行

server 返回 `{"actions": (10, 12)}`。client 维护一个动作队列：每控制周期弹出一个 12 维动作，前 6 维按左臂电机顺序构造 `{left_<m>.pos: v}`，后 6 维构造右臂动作，调用 `robot.send_action`。队列耗尽时**必须**先采集新观测、请求 server，再继续循环。**禁止**跨 chunk 混合新旧动作或乱序执行。

### 决策 5：控制循环与频率

固定 FPS（默认 30，与数据采集一致），用 deadline 式计时（参考 `lerobot_record` 主循环）保证周期稳定。支持 `--duration` 秒数上限；不设上限时运行至 Ctrl-C。

### 决策 6：依赖隔离

`openpi-client` **必须**作为惰性导入（在入口函数内 import），未安装时给出明确的安装提示（`uv pip install <openpi>/packages/openpi-client`）；**禁止**将其加入 `pyproject.toml` 的强制依赖。

### 决策 7：安全退出

主循环**必须**包在 try/finally 中：无论正常结束、Ctrl-C 还是异常，都执行 `robot.disconnect()`；Ctrl-C 时立即停止发送新动作。机器人复用已有标定（`robot.connect()` 默认行为），不在本脚本中重新标定。

## Risks / Trade-offs

- [chunk 边界的推理阻塞导致机器人短暂停顿（pi05 推理约数百 ms）] → 同机/局域网部署 server；文档注明该停顿为首版已知行为；后续如需平滑再引入异步队列或 RTC。
- [观测键名或关节顺序与训练数据不一致 → 策略输出乱动] → 关节顺序直接取自 `bus.motors`（与录制数据集时同源）；相机映射做成显式 CLI 参数并在启动日志中打印实际映射。
- [server 未启动或断连 → websocket 报错] → 启动时先做一次 `get_server_metadata()` 探活并打印，失败即退出并提示检查 `deploy.sh`。
- [动作单位不一致（弧度 vs 角度）] → 保持与 `rollout.sh` 录制链路一致：`SOFollower` 默认 `use_degrees=True`，训练数据即为度数，client 不做任何单位变换。

## Open Questions

- 是否在 chunk 内对动作做插值/滤波以减轻边界抖动？（首版不做，实测后再评估）
