# 验证记录

变更：`add-fineart-interactive-teleop-takeover`。日期：2026-11-06。

## 完整性、正确性和一致性

| 维度 | 结果 |
| --- | --- |
| 完整性 | 11 项实现任务完成；7 项需求、16 个场景都有实现或检查证据 |
| 正确性 | 新增 33 项无硬件测试；最新相关回归合计 328 passed、1 skipped |
| 一致性 | 复用现有交互控制器和推理引擎；库配置默认关闭，本机脚本默认开启；未修改 DAgger 或训练依赖 |

## 需求到实现和验证的映射

| 需求 | 实现 | 验证 |
| --- | --- | --- |
| 显式启用和兼容检查 | `configs.py` 的 `TeleopTakeoverConfig`、配置校验；`context.py` 的设备能力校验；`takeover.py` 的关节和反馈校验 | 默认交互回归；缺少配置、单位不匹配、关节不匹配、无反馈/保持能力测试 |
| 实测位置持续跟随 | `LeaderFollower.align/sync`；`BaseStrategy.run` 和控制器启动规划循环 | 实测值不同于策略目标；使能力矩前写入当前位置；双关节误差检查；非有限位置拒绝 |
| 完成对齐再接管 | `RolloutController.teleop/_run_teleop`；`LeaderFollower.align/release` | 运行和规划期间接管；对齐超时不放行；对齐期间停止 |
| 恢复自动规划 | `RolloutController.resume/_run_segment`，沿用整体目标和启动规划 | 人工纠正后在新观测上重新规划；手工 subtask 恢复；真实 RTC 引擎在途 chunk 被丢弃 |
| 生命周期和故障 | 控制器取消事件、base 发送前取消检查、故障退出抑制回初始姿态 | reset/stop/父事件取消人工控制；reset 取消待处理 resume；反馈、主臂读取和从臂发送失败；晚到动作和规划无效 |
| 部署入口 | `rollout.sh`、FineART 和 inference 文档 | 脚本默认启用、显式关闭和硬件环境变量覆盖配置检查；启用后 RTC/sync 配置检查；shell 语法检查 |
| 空格即时切换 | `StdinCommandListener` 的可选 TTY cbreak 输入、`RolloutController.toggle_teleop`、交互会话空格回调 | 伪终端无需回车触发、命令空格和 UTF-8 编辑、连发抑制、终端恢复、管道兼容、过渡期重复请求、完整按键接管恢复循环 |

## 已执行检查

- pytest：`tests/test_interactive_takeover.py`、`tests/test_interactive_rollout.py`、`tests/test_rollout.py`、`tests/test_rollout_action_ordering.py`、`tests/test_recording_inference_pause.py`、`tests/test_remote_rollout.py`、`tests/inference/test_local_rtc_regressions.py`、`tests/inference/test_dispatch_helpers.py`、`tests/common/test_control_utils.py`，结果 **315 passed, 1 skipped**。
- 当前训练虚拟环境没有 pytest，网络解析不可用。测试通过 `/tmp/fineart-takeover-pytest.sh` 使用 `uv run --no-sync python -m pytest`，临时注入本机缓存的 pytest 8.3.5、pluggy、iniconfig 和 pygments，关闭第三方 pytest 插件自动加载；未安装或升级训练环境依赖。
- Ruff 0.16.7：改动源码和测试的检查、格式化检查通过。
- Mypy 2.1.0：检查 7 个改动源码文件，仅发现 `context.py` 已存在的 `unused-ignore`。对改动前保存的同一文件复查，也在原位置复现相同告警；不改动该基线代码。使用 `--no-warn-unused-ignores` 后 7 个文件通过。没有声称通过仓库完整 pre-commit mypy 2.3.1 检查。
- `bash -n rollout.sh`、`git diff --check` 通过。
- `bash rollout.sh check --inference.type=sync` 通过（接管关闭）。
- 使用 `TELEOP_TAKEOVER=1 LEADER_ID=check_only LEFT_LEADER_PORT=/dev/check-left RIGHT_LEADER_PORT=/dev/check-right` 分别检查 RTC 和 sync 配置，均通过。这里使用的是明确虚构的检查串口，没有连接设备。
- `openspec validate add-fineart-interactive-teleop-takeover --strict` 通过。
- 对照改动前快照审阅 diff；原有训练脚本、推理基类和既有交互测试改动保持原样。

## 空格快捷键追加验证

- 在上述 9 个测试文件基础上加入 `tests/utils/test_stdin_input.py`，最新完整结果为 **328 passed, 1 skipped**。
- 使用真实 POSIX 伪终端验证空格无需回车即可触发，并覆盖空格→接管→人工移动→空格→在新观测上重新规划的完整链路。
- 已覆盖连续空格防抖、命令参数空格、UTF-8 跨读取边界、退格、Ctrl-U、Ctrl-D、管道输入和输入故障后的终端恢复；检查 cbreak 保留 ISIG，因此 Ctrl-C 仍由既有信号处理路径处理。
- 本次修改的 `stdin_input.py`、`controller.py`、`interactive.py` 使用完整现有 mypy 配置通过，不需要排除 `unused-ignore`；Ruff 检查、格式检查、shell 语法检查、`git diff --check` 和 OpenSpec 严格校验通过。
- TTY 不提供按键释放事件，采用 0.75 秒空格输入间隔抑制常见连发。使用时聚焦 rollout 终端、保持输入行为空、短按空格；未启用 takeover 或非 TTY 不启用快捷键。

## 本机硬件默认值追加验证

- 对照 `~/Dev/Manipulation/lerobot/teleoperate.sh`，保留 `bimanual_follower` 和从臂固定串口，补齐 `bimanual_leader` 及左右主臂 `5B3E089335` / `5B3D044325` 的 `/dev/serial/by-id/` 路径；四条臂均使用 `use_degrees=false`。
- 本机脚本默认 `TELEOP_TAKEOVER=1`；库 `TeleopTakeoverConfig.enabled` 仍为 `False`。脚本不再要求手工填写已知主臂默认值，所有硬件环境变量仍可覆盖。
- `bash rollout.sh check` 通过，输出 `enabled=True` 和上述主臂配置；`TELEOP_TAKEOVER=0 bash rollout.sh check` 通过，输出 `enabled=False`；覆盖主从臂 ID 和四条检查用虚构串口的配置检查通过。三种配置均保留 RTC 和启动自动规划目标。
- 当前工具环境无可用 CUDA，配置检查将 device 回退到 CPU；这些结果仅证明配置解析，未加载权重或执行推理。
- 按当前 `HF_HOME=/data/.cache/huggingface` 只读查找，`lerobot/calibration/robots/so_follower/` 下存在 `bimanual_follower_left.json` / `bimanual_follower_right.json`，`lerobot/calibration/teleoperators/so_leader/` 下存在 `bimanual_leader_left.json` / `bimanual_leader_right.json`。只确认文件存在，不验证其与实物电机一致，也未修改标定。
- shell 语法检查、`git diff --check`、OpenSpec 严格校验通过。本次仅修改脚本默认值和文档，沿用上一节控制逻辑回归结果，未重复运行整套测试。

## 待真机验证

没有加载 FineART 权重、连接机器人、启动遥操作或实际运动。主臂串口和标定 ID 已取自参考脚本，仍需真机验证标定一致性、跟随频率、主从臂误差阈值、力矩切换手感和夹持行为。sync 推理及设备 I/O 不能被强制中断；当前脚本默认 RTC，但软件测试不证明硬实时接管或物理零突变。

## 审阅结论

未发现未实现的规格项。软件实现和无硬件验证完成；保留上述类型检查基线告警和真机验证边界，变更保持未归档，供审阅和后续真机验收。
