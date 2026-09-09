## Why

DM05（OpenDM 仓库）训练的 SO-101 双臂折叠策略通过 Flask HTTP 服务（`POST /v1/infer`）提供推理。本仓库 `lerobot-rollout` 已有完善的部署引擎（录制 strategies、interactive 会话、安全停机、`--rename_map`），若把 DM05 服务封装为一个 lerobot 策略插件，即可整体复用这套功能；而不必像 openpi 客户端那样另写独立 CLI。

## What Changes

- 新增 `dm05` 策略插件（`src/lerobot/policies/dm05/`）：
  - `configuration_dm05.py`：`DM05Config`（注册名 `dm05`，字段 host/port/timeout/camera_keys 等）
  - `modeling_dm05.py`：`DM05Policy.select_action` 将观测编码为 `/v1/infer` JSON（12 维 percent state、base64 JPEG 槽位 `"1"/"2"/"3"`、prompt），返回的 50 帧 chunk 按 `execution_horizon` 截断后入队逐 tick 下发（None=整段执行，25=只执行前 25 帧）；`drop_queued_actions` 支持任务切换；无权重加载（权重在服务端）
  - 可选 `history_images` 支持：`use_history=true` 时，策略维护主相机帧的 deque（maxlen=5），每次请求新 chunk 时把历史帧（旧→新，不含当前帧）随请求发送——帧间隔自动等于 `execution_horizon`；旧 checkpoint（服务端未开 `--data-config.is-history`）保持不发送
  - `processor_dm05.py`：近似恒等的 pre/post processor 工厂（归一化在服务端）
- 服务不可达时在 rollout 上下文构建阶段 fail-fast（策略-first、硬件-last 的现有顺序天然保证机器人不被触碰）
- 更新 `rollout_dm05.sh`（新建，原独立客户端方案改为 `lerobot-rollout --policy.path=<dm05-config-dir> --inference.type=sync`）

## Capabilities

### New Capabilities

- `dm05-policy`: 以 lerobot 策略插件形式对接 DM05 HTTP 推理服务：策略加载（config-only）、观测编码、chunk 缓存与逐 tick 执行、任务切换语义、fail-fast 连接、可选 history_images 队列。

### Modified Capabilities

（无——`lerobot-rollout` 引擎、其它策略、openpi 客户端均不改动）

## Non-goals

- 不支持 `speed`/`control_mode`、`robot_type` 覆盖（SFT 服务端有默认值且与训练一致）
- 不做 fast backend（TensorRT）特有处理、不实现文本头（`supports_text_generation` 返回 False）
- 不在客户端做归一化/单位换算（percent 原样收发，已验证与服务端 norm_stats 一致）
- 不支持 `--inference.type=rtc`（chunk 逻辑内置于策略，sync 已覆盖）

## Impact

- 新目录：`src/lerobot/policies/dm05/`（3 个文件）
- 修改：`src/lerobot/policies/__init__.py`（**必须**加一行 `from .dm05.configuration_dm05 import DM05Config as DM05Config`——`@register_subclass("dm05")` 只在模块被 import 时执行，内置策略靠这里的显式 import 完成注册，漏掉则加载即报 "Policy type 'dm05' is not registered"）、`src/lerobot/policies/factory.py`（`make_pre_post_processors` 增加 dm05 分支）、`rollout_dm05.sh`（新建）
- 新增本地小目录 `models/opendm/deploy/dm05_folding/` 存放 lerobot 格式 `config.json`（仅连接配置，无权重）；注意 `models` 在 .gitignore 中，该目录为本地产物不入库
- 依赖：仅 `requests`（已有）；服务端 `image-prompts` 顺序约束由 `camera_keys` 映射满足
- state/action 关节顺序：`DM05Config` **必须**定义 `action_feature_names` 字段（仿 pi05），供 rollout 上下文对齐硬件特征顺序（[context.py:426](src/lerobot/rollout/context.py#L426)）——漏了且两侧关节顺序不一致时是静默错位