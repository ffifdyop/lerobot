## Why

当前 FineART 交互式 rollout 可以自动生成 subtask，但无法在同一会话内切到人工遥操作。已有 DAgger 只在暂停后对齐主臂，而且不能与当前 autosteer 入口组合。需要让主臂在策略控制期间持续跟随从臂，并允许人工纠正后继续自动规划。

## What Changes

- 为交互式 base rollout 增加可选的遥操作接管配置，以及 `/teleop`、`/resume` 命令。
- 在聚焦的 POSIX 终端中支持空格立即切换接管/恢复，无需回车，同时保留文字命令输入。
- 策略执行和规划期间，将从臂实测关节位置持续反馈给主臂，支持同构双臂 SO-101。
- 接管时暂停策略、取消旧规划，保持从臂并检查主从臂对齐，再释放主臂供人操作。
- 恢复时保留整体目标，清空旧动作和规划上下文，使用当前画面重新生成 subtask 后执行。
- 扩展启动脚本、配置检查、文档和无硬件回归测试。

## Capabilities

### New Capabilities

- `interactive-teleop-takeover`: 主臂持续跟随、交互接管、人工控制和恢复 autosteer 的完整生命周期。

### Modified Capabilities

无。现有 OpenSpec 尚无 rollout 能力规格。

## Impact

涉及 rollout 配置、控制器、base 策略、动作发送边界、遥操作反馈接口及 `rollout.sh`。复用已有推理引擎和 SO 主臂驱动，不新增依赖。库配置默认不开启；本机 `rollout.sh` 沿用参考 `teleoperate.sh` 的标定 ID、固定串口和单位，默认启用 takeover，支持环境变量关闭或覆盖。不改变原有 DAgger 行为。

## Non-goals

不增加 DAgger 训练或干预数据集录制，不改模型权重，不实现力反馈或任意异构关节映射，不把软件测试当作真机无突变或实时性验证。
