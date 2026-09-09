## Context

`lerobot-rollout` 是策略部署引擎：`build_rollout_context`（context.py）按「策略-first、硬件-last」顺序装配 `RolloutConfig → policy → processors → robot → dataset → inference engine`，strategies 只消费 `InferenceEngine.get_action` 返回的动作张量。策略插件是约定驱动的：`@PreTrainedConfig.register_subclass("dm05")` 后，`get_policy_class` 自动从 `modeling_dm05.py` 导入 `DM05Policy`，无需改工厂主逻辑。

DM05 服务端（OpenDM，deploy.sh 启动，端口 7891）：`POST /v1/infer`，JSON 含 `observation.prompt/state/images`（base64，槽位 `"1"/"2"/"3"` 对应 image-prompts Head/Left wrist/Right wrist），返回 `{"actions": (50, 12)}`。已验证 state/action 均为 percent 单位（与 `use_degrees=false` 从臂一致），归一化全部在服务端完成。

## Goals / Non-Goals

**Goals:**
- `dm05` 作为标准 lerobot 策略插件接入 `lerobot-rollout`（`--inference.type=sync`）
- 复用全部现有 rollout 能力：录制 strategies、interactive、`return_to_initial_position`、`--rename_map`、可视化
- 服务不可达时 fail-fast，机器人零触碰

**Non-Goals:**
- 不支持 `--inference.type=rtc`（chunk 语义内置在策略里，sync 已覆盖；rollout 上下文的 `supports_rtc_inference(policy)` 检查（[context.py:333](src/lerobot/rollout/context.py#L333)）会正确拒绝未实现 RTC 语义的 dm05——trained 模式下另有 `_validate_trained_rtc_rollout_config` 把关 pi05）
- 不做文本头（`supports_text_generation() → False`）、不支持 `speed`/`control_mode`/`robot_type` 覆盖（SFT 服务端默认值与训练逐字一致，已核验）
- 不改 `lerobot-rollout` 引擎与 `lerobot-openpi-rollout`

**已核验的 history 语义（训练侧 `BuildHistoryImages`，[transforms.py:218](/data/opendm/opendm/data/transforms.py#L218)）**：仅主相机（image_keys[0]）；stride 默认 chunk_size（50）、可经 `history_stride` 显式覆盖；max 5 帧、旧→新、episode 开头变短；不含当前帧（取 frame_index - k*stride, k=1..5）。

## Decisions

### D1: 做成策略插件，而非独立 CLI（或新推理后端）
备选有三：
1. **策略插件（选定）**：`select_action` 内 POST `/v1/infer` + 本地 chunk 队列。chunking 策略的既有模式（pi0 等）完全同构；录制/interactive/reset 全部免费复用；fail-fast 由「策略-first、硬件-last」装配顺序天然保证。
2. 新增 `InferenceEngine` 后端（`--inference.type=dm05`）：被否决——推理后端抽象携带 `policy/preprocessor/postprocessor` 等大量 lerobot 本地权重语义，DM05 客户端全用不上，塞进去反而要给 factory 加一堆跳过分支。
3. 独立 CLI（原方案）：被否决——重复造控制环/安全停机/相机管理，且无法享受录制 strategies。

### D2: config-only 策略目录，`from_pretrained` 强制重写 + 插件注册 import
`--policy.path` 指向一个小目录（`models/opendm/deploy/dm05_folding/`，与本地训练权重 `models/lehome/train/...` 的惯例并列：`opendm` 标明策略出处，`deploy` 对应远端服务部署），内含 `config.json`：`{"type": "dm05", "host": "127.0.0.1", "port": 7891, "timeout": 10, "camera_keys": {...}, ...}`。`DM05Config` **必须**只存连接配置。服务端 checkpoint 路径只存在于 deploy.sh，客户端目录不含任何模型文件。

两个**必须**满足的加载契约（否则 config-only 目录加载即失败/无法实例化）：
1. **`DM05Policy.from_pretrained` 重写是强制的**：基类对本地目录无条件加载 `model.safetensors`（[pretrained.py:180-183](src/lerobot/policies/pretrained.py#L180-L183)），config-only 目录必炸。重写为：`config = PreTrainedConfig.from_pretrained(path)` → `cls(config)`，跳过权重加载。
2. **插件注册 import**：`@register_subclass("dm05")` 只在模块被 import 时执行；dm05 不是 `lerobot_policy_*` 第三方包，**必须**在 `src/lerobot/policies/__init__.py` 显式 `from .dm05.configuration_dm05 import DM05Config as DM05Config`（与内置策略同款，[__init__.py:17-38](src/lerobot/policies/__init__.py#L17-L38)），否则 `PreTrainedConfig.from_pretrained` 报 "not registered"。

基类抽象成员的必填样板（**必须**实现，缺失则类无法实例化）：
- `DM05Config`：`observation_delta_indices`/`action_delta_indices`/`reward_delta_indices`（返回 None 即可）、`get_optimizer_preset`/`get_scheduler_preset`（远端策略无训练，返回占位/None）、`validate_features`（[configs/policies.py:107-129](src/lerobot/configs/policies.py#L107-L129)）
- `DM05Policy`：类属性 `config_class = DM05Config`（`__init_subclass__` 强制校验，[pretrained.py:99-102](src/lerobot/policies/pretrained.py#L99-L102)）；抽象方法 `forward`（直接委托 `predict_action_chunk`）、`predict_action_chunk`、`select_action`、`reset`、`get_optim_params`（占位返回空 dict）

### D3: chunk 语义在 `select_action` 内实现，`execution_horizon` 截断后段
- 队列空时以最新观测 POST，取回 `actions`（(chunk_size, 12)）**必须**截断为前 `execution_horizon` 帧入队，其余帧丢弃；每 tick 弹出一帧转 tensor 返回。默认 `execution_horizon=None` 表示整段执行（与返回的 chunk 等长）
- `execution_horizon` 为 `DM05Config` 字段（per-deployment 配置，与 use_history 同层）：25 表示 50 帧 chunk 只执行前 25 帧，消费完即以最新观测请求新 chunk。校验**必须**为"None 或正整数"——客户端无法预知服务端 chunk_size，超出场景由对首个响应实际长度的截断自然处理（slice 越界即整段）
- **队列命名契约**：chunk 队列属性**必须**命名为 `_action_queue`——基类 `drop_queued_actions` 默认实现只清 `_action_queue_attrs = ("_queues", "_action_queue")` 中的属性（[pretrained.py:87](src/lerobot/policies/pretrained.py#L87)），用别的名字任务切换时是静默 no-op
- `reset()` 清动作队列（interactive `/reset` 语义）；history deque 由 D6 单独处理
- **必须**与 history stride 联动（见 D6）：history 帧间隔 = 实际执行的帧数，`execution_horizon=25` 时 history 间隔也是 25 帧而非 50——否则 history deque 的采样节奏与执行节奏脱钩，破坏"每次 chunk 请求时刻入队一帧"的语义

### D4: 观测编码与关节顺序对齐
- **state 顺序**：`observation.state` 的顺序来自硬件特征顺序，**必须**由 `DM05Config.action_feature_names` 字段显式声明 DM05 训练的关节顺序（仿 [configuration_pi05.py:72](src/lerobot/policies/pi05/configuration_pi05.py#L72)），rollout 上下文用它对齐硬件特征（[context.py:426](src/lerobot/rollout/context.py#L426)）。漏了且两侧顺序不一致时是**静默错位**（无报错，机器人动作错乱）
- `state`：取预处理后观测中 action 特征对应的 12 个标量，percent 原样，**禁止**换算
- `images`：`camera_keys` 配置为「服务端槽位名 `"1"/"2"/"3"` → rename 后的观测相机**完整键名**（如 `observation.images.top`）」，默认 `{"1": "observation.images.top", "2": "observation.images.left_wrist", "3": "observation.images.right_wrist"}`；帧从 float32 CHW [0,1] 张量转 uint8 HWC（`torch.inference_mode` 下转 numpy 前**必须** `.clone()`），再 `cv2.imencode(".jpg")` → base64。**必须**用 JPEG（体积小，30Hz 下编码快），**禁止**发 PNG
- `prompt`：batch 中的 `task` 字段（`prepare_observation_for_inference` 已注入）

### D5: processor 管线近似恒等，但必须带两个适配义务
`processor_dm05.py` 提供 `make_dm05_pre_post_processors`：pre 含通用 `rename_observations_processor`（接受 `--rename_map` 覆盖），post 为空管线。归一化在服务端（norm_stats 随 checkpoint），客户端**禁止**加 Normalize step。`factory.py` 的 `make_pre_post_processors` **必须**增加 dm05 分支（与 groot 同样的 dispatch 模式）。

两个**必须**满足的管线义务（缺一即 sync 引擎调用直接断）：
1. **transition 适配器**：管线构造**必须**传 `to_transition=batch_to_transition` / `to_output=transition_to_batch`（pre）和 `to_transition=policy_action_to_transition` / `to_output=transition_to_policy_action`（post），与内置策略同款（[factory.py:226-238](src/lerobot/policies/factory.py#L226-L238)）
2. **override 过滤**：rollout 总会传 `preprocessor_overrides`（含 `device_processor`、`rename_observations_processor`）；近恒等管线没有 `device_processor` step，**必须**仿 groot 的 `_drop_groot_absent_standard_overrides`（[processor_groot.py:382](src/lerobot/policies/groot/processor_groot.py#L382)）把管线中不存在的 step 的 override 丢弃，否则 KeyError

### D6: history_images 以 `use_history` 开关支持，deque 间隔 = 实际执行帧数
DM05 新代码（`BuildHistoryImages`）的 history 语义：仅主相机、旧→新、不含当前帧、max 5 帧、episode 开头变短；训练时 stride 默认取 action_horizon（chunk_size），但支持显式 `history_stride` 覆盖。客户端实现：
- 策略内维护 `deque(maxlen=5)`；每次请求新 chunk 时**必须**先发送 deque 现有内容（旧→新）再 append 本次主相机帧——**入队节奏与 chunk 请求节奏天然相等**，因此 history 帧间隔自动等于 `execution_horizon`（截断后实际执行的帧数）：整段执行时间隔 50 帧，`execution_horizon=25` 时间隔 25 帧
- 与训练 stride 的对齐由部署参数共同决定：训练侧设 `history_stride=25` 且部署 `execution_horizon=25`，则 rollout 时的 history 间隔与训练分布完全吻合（训练侧代码已支持 `history_stride` 字段，[dm05_so101_bimanual_sft.py:44](/data/opendm/playground/dm05_so101_bimanual_sft.py#L44)）
- `reset()` 清空 deque，复现训练侧 episode 开头分布
- **必须**做成 `use_history` 配置开关且默认 False：当前 checkpoint-40000 的服务未开 `--data-config.is-history`，携带该字段会 400（[dm05_exp.py:738](/data/opendm/opendm/exp/dm05_exp.py#L738)）；新 checkpoint 的服务开该开关
- 与 chunk 队列共用 `drop_queued_actions` 语义：任务切换丢弃 chunk 时 history deque **禁止**清空——历史是物理时间上的观测记录，与任务无关

### D7: fail-fast TCP 探测放在 `DM05Policy.__init__` 或 `start`
`build_rollout_context` 在连接机器人之前加载策略，所以 `__init__` 里 `socket.create_connection((host, port), timeout=5)` 失败即抛错 → 机器人在此之前绝不会被触碰。**必须**在错误信息里带上 deploy.sh 启动提示。

## Risks / Trade-offs

- [sync 引擎每 tick 调一次 `select_action`，若 HTTP 慢（>1/fps）会拖慢控制环] → DM05 服务单请求推理 ~百 ms 级，30Hz 下偶发慢帧由 chunk 摊平（50 帧只需 50 帧一次请求）；实测后再评估是否需要伪 RTC

### 网络故障语义（Important 3 的 spec 补充由 specs 承载）
HTTP 断连/超时/非 200 时 `select_action` **必须**抛 RuntimeError（含服务地址、HTTP 状态码、响应体摘要），控制环终止——远端策略失联时继续执行旧动作比停机更危险。spec 中有对应的可测试 scenario。
- [`select_action` 抛错（如 HTTP 断连）会终止控制环] → 这是预期行为：远端策略失联时继续执行旧动作比停机更危险；错误信息**必须**含服务地址便于排查
- [base64 JPEG 编码开销] → 1280×720 JPEG 编码 ~ms 级，可接受；若成瓶颈再降分辨率/调 quality
- [`execution_horizon` 截断 chunk 后段 = 放弃模型规划的部分未来] → 与 RTC 同源的正确性权衡：越靠近 chunk 尾的动作基于越旧的观测，截断后段换响应性；且截断使重规划频率翻倍（25 帧一次），history 间隔随之变密。**必须**保证服务端吞吐扛得住 2 倍请求频率；模型效果与训练 stride 对齐依赖训练侧 `history_stride` 配合（见 D6）
- [history deque 与 chunk 节奏耦合：若 chunk 请求提前（如手动 drop_queued_actions），实际间隔会短于配置的 execution_horizon] → drop 场景仅在任务切换时发生（罕见）；且 deque 语义与训练侧"变短历史"一致，影响有限。实测新 checkpoint 效果后再评估
- [config-only 目录绕过 `--policy.path` 的常规权重校验] → `DM05Config` 校验 host/port 合法性；`from_pretrained` 重写后跳过权重加载（见 D2）并打印"权重在服务端"提示

## Open Questions

- 无（`drop_queued_actions` 在 `PreTrainedPolicy` 基类已有默认契约，[pretrained.py:230](src/lerobot/policies/pretrained.py#L230)，直接复用）