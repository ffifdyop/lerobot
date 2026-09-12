# LeRobot（lehome 分支）

真机操作技术栈：遥操作、数据录制、策略训练与策略上机 rollout（SO-101 级机械臂、三相机）。本词汇表固定 rollout、推理与可视化相关的项目用语。

## 术语表

### Rollout

**Rollout**：
在真实机器人上现场运行已训练的策略，观察其行为。
_避免_：评测（评测隐含指标对比）、部署

**Rollout Run（一次 Rollout 运行）**：
一次 rollout 策略的完整执行，从启动到停止。base 模式下没有 episode 概念——这次运行就是 rollout 期间产生的一切（视频、日志）的天然边界。
_避免_：episode（仅对录制类策略才有意义）

**Policy Tick（策略节拍）**：
一次策略推理：观测进、动作块出。

**Control Tick（控制节拍）**：
控制循环的一步（在动作块内做插值）。较慢的策略节拍驱动较快的控制节拍。

### 策略与推理

**Local Policy（本地策略）**：
网络在本进程内运行的策略，其内部结构（梯度、attention）可直接检查。SmolVLA 是本地策略。
_避免_：离线策略

**Remote Policy（远程策略）**：
经网络（HTTP/websocket）服务的策略；rollout 进程只能看到观测 → 动作对，永远接触不到网络内部。DM05 与 openpi 策略是远程策略。

### 显著性可视化

**Saliency Map（热力图）**：
每个相机一张的重要性图，表示策略当前正在执行的那个动作（a₀ = chunk 首个被执行动作）与输入的关系。每个相机每个策略节拍一张。现有两种实现方法（method），通过 `--saliency.method` 选择。

**Pixel Sensitivity Map（像素敏感度图）**：
热力图的第一种方法（`grad_x_input`）：像素级，表示每个输入像素对 a₀ 的贡献量。"贡献"的精确定义（2026-09-11 确认）：该像素放大/缩小当前指令幅度的能力，即 ∂‖a₀‖²/∂x——因此动作值加权是有意的：指令幅度接近零（姿态过均值点）时全图变暗是正确行为，幅度大的关节主导也是正确行为。
_避免_：关节等权敏感度（曾被考虑并否决——舵机角度值加权正是所需语义）

**Patch Attention Map（Patch 注意力图）**：
热力图的第二种方法（`attention`）：patch 级（每相机 8×8），表示 a₀ 动作 token 对各图像 patch token 的 cross-attention 份额（heads 与 denoise steps 取均值，原始份额不做重归一化——模型看语言多时整图变暗是正确行为）。

**Attention Budget Bar（注意力预算条）**：
a₀ 动作 token 的 cross-attention 质量在五份上的分配——cam1 / cam2 / cam3 / 语言 / state，总和恒等于 1。以堆叠横条叠加在叠加热力图视频每帧顶部。

**Saliency Overlay Video（叠加热力图视频）**：
原始相机画面与其热力图（任一 method）混叠后的视频，每个相机每次 rollout 运行各一个文件，供事后检查。
_避免_：调试视频

**Explainer（解释器）**：
策略无关的组件：输入一份观测和本地策略，产出热力图。本地策略默认获得热力图能力；特殊架构通过按策略注册的解释器覆盖通用实现。
_避免_：可视化器（专指 rerun/foxglove 显示后端）