# 2026-09-22 验收记录

## 5.6 规格协调

已修改人工接管 change 的 saliency spec、design 和任务 6.1：Policy 接受入队后解释 a_d；Shadow 只解释同观测 proposal 的显式 a₀，独立提交，不能伪造队列裁切元数据或进入执行队列。两者明确区分 queued-policy/shadow，均不能宣称动作已经执行。保留层/步筛选、原始预算、完整梯度、生命周期、epoch 失效和旁路失败隔离。Shadow 的 recorder 模式分支仍由人工接管变更实现。

两份 change 均通过 `openspec validate <change> --strict`。

## 5.7 训练基线

固定改动前 Git 版本 `9706a20c0116ff99f48d4d24af98dc50704df786`。测试通过 `git show` 读取不可变源码，当前工作树保持原状；运行测试需要此 Git 对象。

`tests/policies/test_saliency_native_baseline.py` 缩小配置维度，保留真实视觉编码器、Gemma/SmolVLM、双流 attention、flow matching 和反向传播。无预训练权重/网络依赖。固定初始化、batch、noise、time，float32 CPU 下验证：

- π0、π0.5：默认/冻结视觉/仅 expert 三种配置，各自开启和关闭梯度检查点，共 12 组。
- SmolVLA：同样三种冻结配置，共 3 组；该原生 core 没有上述 π 梯度检查点开关。
- 所有组 loss 和每个非空参数梯度均以 `atol=rtol=0` 通过；参数 requires_grad、梯度 None 状态和 state_dict 键一致。
- 共享 Euler/RTC 默认路径分别有/无 leftover，输出和 denoiser 参数梯度与旧版严格相同。
- SmolVLA 的训练、冻结与 attention 函数另有源码 AST 一致性检查。

这验证训练代码兼容性，不是完整模型长期训练质量对比。

## 5.8 默认路径与资源

三类微型真实模型的固定噪声 eager 采样与改动前版本严格相同，随机状态不变；queue replace/append 的裁切及来源信息与旧版一致。无 saliency sink 的 RTC engine 测试将额外噪声采样和快照构造设为禁止操作，并确认正常入队。代码审查确认副本加载仅在 `cfg.saliency.enabled` 分支中，默认 capture 为 None。

GPU 编译测试使用宿主 RTX 5090、PyTorch 2.11.0+cu128、float32，`torch.compile(backend="inductor", fullgraph=True)`。π0/π0.5 首轮 default mode 均通过；随后在 `max-autotune` 下对比新旧 compiled 以及旧版 eager，最终 3 passed：

| 模型 | 旧 compiled 对 eager 最大绝对误差 | 新 compiled 对 eager 最大绝对误差 | 结果 |
| --- | ---: | ---: | --- |
| π0 | 2.98e-7 | 3.58e-7 | 通过 |
| π0.5 | 1.19e-7 | 2.38e-7 | 通过 |
| SmolVLA | 0.337852478 | 0.337852478 | 新旧 compiled 回归通过；旧版已有 eager/compiled 偏差 |

三个模型的新旧 eager 均严格相同，新旧 compiled 均通过 atol=rtol=1e-5 比较。SmolVLA 偏差在改动前代码同样复现，尚未定位原因；仅为缩小配置的复现，不能直接断言真实 checkpoint 有同样问题。编译回归的判据是同一编译入口的新旧版本比较，故 5.8 通过；另保留旧版 eager/compiled 偏差为独立待定位问题，不提高容差也不设 xfail。

π0/π0.5 本轮编译及首次执行分别约 9.21/10.20 秒，随后三次平均 2.74/2.54 毫秒（微型模型，已有磁盘编译缓存）。这不是完整 checkpoint 或机器人控制性能数据。CUDA graph 输出在测试中显式 clone，并标记调用边界，避免后续执行覆盖输出。

CPU 回归命令：

```bash
UV_CACHE_DIR=/tmp/lerobot-uv uv run --no-sync env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/policies/test_saliency_native_baseline.py tests/test_saliency.py tests/test_interactive_rollout.py tests/rollout/test_view_mask_integration.py tests/policies/rtc/test_action_queue.py tests/policies/rtc/test_modeling_rtc.py tests/policies/common/test_flow_matching.py tests/policies/test_pi_gemma.py tests/policies/pi0_pi05/test_pi05_training_time_rtc.py -q -k 'not replays_real_policy and not replay_real_policy and not real_policy'
```

最新结果：306 passed，3 skipped（opt-in GPU 编译测试），2 deselected（需模型资产的 SmolVLA 测试）。

GPU 编译复现（需要宿主 GPU 可见，卡号按实际空闲情况选择）：

```bash
CUDA_VISIBLE_DEVICES=3 SALIENCY_COMPILE_TEST=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/policies/test_saliency_native_baseline.py -q -s -k compile
```

### 真实 π0.5 十步资源测量

checkpoint：`models/lerobot/pi05_so101_bimanual_folding_cloth_20260921_224353/checkpoints/030000/pretrained_model`，全部键成功加载。GPU 为 RTX 5090，float32，10 steps，chunk=50，模型动作维度=32；三路合成 224×224 输入和 16 个合成 prompt token，seed=421，目标 d=2。独立测试实例 eval，所有参数冻结，与 manager 的副本设置一致。每段用 CUDA synchronize 测量 wall time，并重置 peak memory 计数。

| 路径 | 单次耗时（秒） | max_memory_allocated（MiB） |
| --- | ---: | ---: |
| 原生 predict_action_chunk | 0.539 | 16067.13 |
| attention replay | 0.285 | 16067.13 |
| grad_x_input replay | 0.590 | 21283.26 |

attention 捕获 180 个事件（三相机地图均有限），预算合计 1.000000108；grad 三相机地图均有限且非零。表内耗时为顺序单次测量，第一段包含冷启动效应，不能据此比较加速比或推出完整 rollout 控制帧率。峰值包含单模型驻留，不含同时加载的控制副本、720p 图像、Rerun/视频和机器人控制。

完整 guided 多步及 trained 成功路径的新验收见下；trained 路径仅有微型随机权重，不能声称真实 checkpoint 具有该训练能力。

## 5.2 真实 π0 guided 验收

checkpoint：`/home/gsp/Dev/Manipulation/lehome-challenge/outputs/train/pi0/pi0_top_long_batch10_no_lora_no_freeze/checkpoints/200000/pretrained_model`，权重键全部加载。NVIDIA RTX PRO 6000 Blackwell Workstation Edition（GPU 0），checkpoint dtype=bfloat16，10 个去噪步、50 动作、模型动作维度 32，三路合成 224×224 图像、16 个合成 prompt token，seed=421；guided delay=2、leftover=4、目标 a₂。独立 eval 模型参数冻结。CUDA 同步的单次耗时/峰值显存：直接推理 0.557 s / 8616.54 MiB，grad_x_input 0.533 s / 12558.60 MiB，attention 0.211 s / 8624.67 MiB。两种方法的三相机地图均有限；梯度图非零。同输入/噪声/条件下直接前向与可微重放的动作最大、平均绝对误差均为 0，a₂ 对三相机梯度均有限且非零。单次顺序测量不代表速度收益或整机节拍。

## 5.3 真实 π0.5 guided 与微型 trained 验收

真实 checkpoint：`models/lerobot/pi05_so101_bimanual_folding_cloth_20260921_224353/checkpoints/030000/pretrained_model`，权重键全部加载。NVIDIA RTX PRO 6000 Blackwell Workstation Edition（GPU 2），float32，10 个去噪步、50 动作、模型动作维度 32，三路合成 720×1280 图像（模型预处理至 224×224）、16 个合成 prompt token，seed=421；guided delay=2、leftover=4、目标 a₂。CUDA 同步单次直接/grad/attention 耗时分别 0.433/0.581/0.268 s，峰值显存分别 16097.04/21346.54/16105.17 MiB。三相机梯度均有限且非零，attention 地图均有限；同输入/噪声/条件下直接前向与可微重放动作最大、平均绝对误差均为 0（另一次 GPU 0 测量）。单次顺序测量不代表控制节拍。

trained RTC 成功路径由 `test_pi05_trained_rtc_explainers_with_miniature_policy` 验证：微型随机初始化的真实 π0.5 架构设置 `rtc_training_max_delay=2`、delay=2，hard prefix 严格等于 leftover，直接与 grad 重放 `atol=1e-5, rtol=0`，attention 捕获完整且预算和为 1。测试暴露了原先仅在 capture 开启时对每动作 timestep 调用 `.item()` 的错误；现从未被清零的末尾 postfix 动作读取去噪时间步。改动不触及无 capture 的训练/默认推理路径，回归套件通过。现有真实 π0.5 checkpoint 的 `rtc_training_max_delay` 均为 0，不是 trained RTC 权重；本项只验证 trained 算法/解释器链路，不声称其真实权重质量。

## 5.4 SmolVLA 离线与实机集成（离线显存替代验收）

真实 checkpoint：`/home/gsp/Dev/Manipulation/lehome-challenge/outputs/train/smolvla/smolvla-real_720p_ft_20260830_142257/checkpoints/080000/pretrained_model`，VLM 从本地 HF 缓存离线加载。NVIDIA RTX PRO 6000 Blackwell Workstation Edition（GPU 0），10 去噪步、50 动作、三路合成 256×256 输入、48 token prompt，guided delay=2、leftover=4、目标 a₂。直接/grad/attention 单次分别 0.681/0.498/0.159 s，峰值 928.94/2292.07/937.21 MiB。梯度三相机图有限且非零；默认 attention 80 事件、预算和 1.000000053；显式全选与默认地图/预算最大误差均 0，单层单步恰好 1 事件。

同一真实模型的 recorder 离线集成：100 次快照提交平均 0.0026 ms，容量 1 的队列保留最新；Rerun 帧和预算日志钩子各收到 1 次，三路相机 MP4 均生成且 OpenCV 可解码为 1 帧。硬件 H.264 编码器初始化失败后仍有可播放的回退输出。另以独立控制/解释副本同卡运行 20 次推理：无旁路均值/p95 154.03/160.66 ms；有 attention 旁路 272.67/294.09 ms，控制推理明显受到 GPU 竞争。此为离线推理压力测试，不含真实机器人动作、传感器/Rerun 实例与真实 RTC 调度，也未统计设备短运行的丢帧率。此段是离线历史记录，实机结果见下。

### 2026-09-23 实机前预检（尚未连接机器人）

`rollout.sh` 当前仍使用 π0.5、360 秒且 saliency 参数被注释，不能直接用于 SmolVLA 短测。前述离线 SmolVLA checkpoint 的 state=6，与双臂设备预期不匹配；本次改选脚本已列出的 `models/lehome/train/smolvla/smolvla-real_720p_ft_freeze_20260920_174448/checkpoints/006000/pretrained_model`，其 state/action 均为 12、三相机训练 rename_map 为 top/left_wrist/right_wrist→camera1/2/3。在未连接机器人、GPU 2、固定合成输入下，直接/grad/attention 推理分别为 0.656/0.554/0.161 秒，峰值 928.94/2292.07/937.21 MiB，两种地图均有限。 该 checkpoint 的 `model.safetensors` 含 474 个 BF16、26 个 F32 张量；当时未单独记录 activation dtype 或 autocast 模式，因此这里仅声明权重存储精度，不推定全部算子精度。

只读检查见四个固定 by-id 串口和 `/dev/video0,6,8`，未发现占用进程；本地左右臂标定文件存在，但无法由文件验证电机中的标定匹配。`SOFollower.connect()` 默认在不匹配时进入可写电机标定的交互提示，且连接本身会配置电机。因此尚未打开串口或发送动作；需现场操作者确认安全工作区、急停/断电可用、标定状态与受限短测参数后再启动。5.4 保持未完成。

### 2026-09-23 已授权 10 秒实机短测

第一次启动在连接机器人前因 Rerun Viewer 不在 PATH 退出；第二次仅把当前虚拟环境自带的 0.33.1 Viewer 加入本次命令的 PATH。使用 state/action=12 的三相机 SmolVLA checkpoint、RTC execution_horizon=10、控制 GPU 2/解释 GPU 3、30 Hz、attention 全层全步、10 秒；现场安全条件由操作者确认。左右从臂固定 by-id 串口、相机 /dev/video0/6/8 均正常连接，未出现标定提示。程序按 duration 自动结束，执行了 3 秒回初始位流程且双臂/相机断开；该流程未报错，但因限幅仍生效，未核对最终关节位置与起点误差；没有再发送第二段动作。

- 控制：300 ticks、299 judged，有效 29.92 Hz；0/299 超过 33.3 ms 工作预算，工作均值/最差 10.2/17.2 ms；24 tick 无可发动作，276 tick 发送动作。该节拍是有 saliency 旁路时的结果，不代表策略动作质量。
- 解释：recorder 接收 16 份已入队快照、处理 16 份，队列丢弃/过时/失败均为 0；就已提交快照而言丢帧率 0/16。处理均值/最差 366.6/562.9 ms。首帧标注 a[24]、layers=all、steps=all，证明实机 RTC 选择了非零裁切索引。此次只实机运行 attention；grad_x_input 已有离线真实 checkpoint 验证。
- 输出：Rerun 0.33.1 查看器显示三路 saliency 图和 saliency/budget 五类曲线；三路 MP4 均可解码，MPEG-4、1280×720、各 16 帧，首帧可见块状网格、五类原始份额预算条和目标动作/层/步标签。视频及 Rerun 查看器截图路径：`outputs/saliency/validation-smolvla-20260923-short/`（截图文件 `rerun-viewer.png`）。
- 安全与局限：双臂设 max_relative_target=0.01；代码核对表明 use_degrees=false 使用 [-100,100] 归一化单位（gripper 为 [0,100]），故此限值不是 0.01 rad，而是每次目标最多变化 0.01 个归一化单位。模型目标与初始位差距很大，安全裁剪连续触发；本次只能证明限幅下的管线/节拍，不证明策略可安全完成任务。该单位更正应在后续实机命令中保持明确。
- 显存替代验收：本次未持续采样 GPU 2/3 的实机运行峰值，驱动 Accounting Mode 为 Disabled，无法事后恢复。2026-09-23 用户明确接受同 checkpoint 的离线单模型峰值替代本项实机峰值；上文 GPU 2、固定合成输入测得直接/grad/attention 分别为 928.94/2292.07/937.21 MiB，本次实机使用 attention。这些数字不含控制与解释器在不同 GPU 上同时运行、720p 相机与 Rerun 的整体占用，也不是 GPU 3 的实机峰值；不得用于承诺整机显存上限。依据修订验收口径，5.4 完成；若未来需要部署容量评估，仍须另测双 GPU 实机峰值。

本次增补的 recorder 计数/耗时日志只在启用 saliency 的旁路线程运行；`tests/test_saliency.py` 为 77 passed，`git diff --check` 通过。

## 2026-09-23 Review 修复验证

修复范围限于 saliency adapter/聚合/recorder、测试及文档；本轮没有再修改共享模型定义、训练 forward 或原生 RTC。

- π0/π0.5 的 attention 与 grad_x_input 均迁移 RTC 张量到解释器设备。微型真实架构在 CPU 同设备和物理 GPU 2→3（`CUDA_VISIBLE_DEVICES=2,3`）分别测试：float32、eager、2 去噪步、4×4 动作块，leftover 与动作块尺寸完全匹配，因此不依赖 RTC padding 隐式迁移。两方法均完成，原快照未修改。
- 上述 8 个用例同时计数 `embed_prefix`：每次重放恰好 1 次，不再为 attention 布局检查额外执行视觉编码。实际 key/query 不匹配另有合成回归测试。
- 三模型共用单组 `AttentionAccumulator`。合成事件覆盖单层单步、多层多步、默认全选、同一事件分发给两个独立组、重复/遗漏/缺失 step 和布局变化。SmolVLA 默认配置不再兼容无 step 的捕获事件。head 均值和事件累加使用 float32，不恢复低精度捕获中已经丢失的信息。
- recorder 分离绘图与发布；分别在计算、渲染、writer 初始化中触发 epoch/revision 更新，确认没有旧视频或 Rerun 输出，不增加失败计数，也不更新色阶状态。多相机测试覆盖最后一张图渲染时失效，以及视频/Rerun 写入期间 reset 后停止后续输出。已经开始的外部写入不追溯撤销，也不持有控制线程锁等待外部 I/O。
- 完整相关套件：`tests/test_saliency.py`、`tests/policies/test_saliency_native_baseline.py`、RTC action_queue/modeling_rtc、pi_gemma 和 interactive_rollout，`-k 'not compile'`：273 passed、4 deselected，66.08 s；随后增加的多输出端失效用例单独运行 3 passed。合计 276 个用例通过，包含改动前训练/默认推理基线、冻结配置与梯度检查点覆盖。
- Ruff 0.11.12 检查与格式检查、`git diff --check`、`openspec validate add-pi-rollout-saliency --strict` 通过。本轮没有重跑 compiled、真实 checkpoint 性能测量或实机；此前测量保留为历史证据，不能当成本轮实现的重新测量结果。用户接受的离线显存替代验收口径保持不变。
