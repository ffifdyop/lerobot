## 1. dm05 策略插件

- [x] 1.1 新建 `src/lerobot/policies/dm05/configuration_dm05.py`：`DM05Config`（`@register_subclass("dm05")`；字段 host/port/timeout/camera_keys/execution_horizon/use_history/**action_feature_names**/input_features 等；校验 host/port 合法、execution_horizon 为 None 或正整数、camera_keys 槽位键完整）；实现基类抽象成员 `observation_delta_indices`/`action_delta_indices`/`reward_delta_indices`（None）、`get_optimizer_preset`/`get_scheduler_preset`（占位）、`validate_features`
- [x] 1.2 新建 `src/lerobot/policies/dm05/modeling_dm05.py`：`DM05Policy`（类属性 `config_class = DM05Config`、`name`；重写 `from_pretrained` 只读 config 不载权重；`__init__` TCP 探测 fail-fast；`select_action` 编码 `/v1/infer` 请求（float32 CHW→uint8 HWC→JPEG→base64，inference_mode 下转 numpy 前 `.clone()`），chunk 按 `execution_horizon` 截断入队**命名为 `_action_queue`**，逐 tick 下发；`use_history=true` 时维护主相机 deque（maxlen=5，发后 append，reset 清空）；`drop_queued_actions`/`reset` 清动作队列；`forward`/`predict_action_chunk`/`get_optim_params` 实现抽象契约；`supports_text_generation → False`）
- [x] 1.3 新建 `src/lerobot/policies/dm05/processor_dm05.py`：近似恒等 pre（仅 rename step）/ 空 post 工厂；管线**必须**带 transition 适配器（`batch_to_transition`/`transition_to_batch`/`policy_action_to_transition`/`transition_to_policy_action`），**必须**仿 groot `_drop_groot_absent_standard_overrides` 过滤管线中不存在 step 的 rollout override（如 `device_processor`）；在 `src/lerobot/policies/factory.py` 的 `make_pre_post_processors` 注册 dm05 分支
- [x] 1.4 在 `src/lerobot/policies/__init__.py` 加 `from .dm05.configuration_dm05 import DM05Config as DM05Config`（注册副作用，漏掉则加载报 "not registered"）

## 2. 部署配置与脚本

- [x] 2.1 新建 dm05 config-only 目录（`models/opendm/deploy/dm05_folding/config.json`：type=dm05、host/port/timeout/camera_keys/action_feature_names；注意 `models` 在 .gitignore 中，为本地产物不入库），验证 `lerobot-rollout --policy.path=<dir>` 能解析出 dm05 策略类型
- [x] 2.2 新建 `rollout_dm05.sh`：`lerobot-rollout --strategy.type=base --policy.path=<dir> --inference.type=sync`，保留串口检查/赋权、相机配置、`use_degrees=false`；注意相机默认键与 rename_map 的配合（若沿用 rollout.sh 的 rename_map 改成 camera1/2/3，需同步设置 config 的 camera_keys 指向 rename 后的键）

## 3. 验证

- [x] 3.1 单元测试（`tests/policies/dm05/`，mock `requests.post`，不依赖真实服务）：chunk 截断（None 整段 / 25 只留前 25 帧）、`_action_queue` 节奏（1 次请求服务 N tick）、`drop_queued_actions` 清队、history deque（reset 后 `[]`、逐次填充至 5 帧、旧→新、间隔=execution_horizon）、请求体字段断言（state 12 维、槽位与 image-prompts 对应、图像编码可解码）、网络故障抛 RuntimeError
- [x] 3.2 启动 DM05 服务（deploy.sh）后联调 dry-check：端到端一次推理返回形状 (chunk_size, 12)、延迟可接受
- [x] 3.3 真机联调：`lerobot-rollout` 全链路跑通（base strategy、到时/Ctrl+C 安全停机、`return_to_initial_position`），全程从臂无异常动作