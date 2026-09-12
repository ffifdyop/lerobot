# Proposal: add-attention-saliency

## Why

现有 rollout 热力图(`grad_x_input`）回答「哪些像素能放大/缩小当前动作」，但回答不了「动作在**看**哪里」——梯度敏感度被像素值与局部对比度调制，而动作 token 对图像 patch 的注意力分配才是模型感知决策的直接读数。两种视角互补，需要并存。

## What Changes

- `SaliencyConfig.method` 新增 `"attention"` 选项，与 `grad_x_input` 共存；默认值不变（向后兼容，非 BREAKING)。
- SmolVLA 注意力截取：`eager_attention_forward` 增加可选 capture callback（默认关闭，仅 saliency 第二副本开启，控制回路零开销）。
- 新增 attention 解释器：取奇数层纯 cross-attn,a₀ query,heads × denoise steps 均值，原始份额（不重归一化），产出每相机 8×8 patch 图。
- 渲染：块状最近邻放大 + patch 网格线 + letterbox 忠实映射（黑边 token 不虚构）；新增**注意力预算条**(cam1/cam2/cam3/语言/state 五份，总和=1）烧录每帧顶部，rerun 同步记录 5 条 scalar 曲线。
- `CONTEXT.md` 术语更新随本变更一并提交：热力图改为家族术语，新增像素敏感度图 / Patch 注意力图 / 注意力预算条。
- `rollout.sh` 切换为 `--saliency.method=attention` 并更新注释。

## Non-goals

- 不删除、不改变 `grad_x_input` 的任何行为与语义。
- 不做远程策略（DM05 / openpi）的解释——其内部不可达。
- 不做 vision encoder 内部 patch-to-patch 注意力、integrated gradients（保留预留位）、关节等权敏感度、per-dim 分解。

## Capabilities

### New Capabilities

- `rollout-saliency`:rollout 显著性可视化——两种 method（像素敏感度图、Patch 注意力图）的语义与产物、叠加热力图视频、注意力预算条、rerun 实时流、精确回放与噪声注入约束。saliency 功能此前无 openspec 记录，本规格补齐两个 method 的完整语义。

### Modified Capabilities

（无。`dm05-policy` 不涉及。)

## Impact

- **代码**:`src/lerobot/rollout/saliency/`(config、新 attention explainer、overlay、recorder)、`src/lerobot/policies/smolvla/smolvlm_with_expert.py`(capture callback)、`src/lerobot/utils/rerun_visualization.py`（预算条 scalars)。
- **测试**:`tests/test_saliency.py`（镜像现有覆盖）、`tests/utils/test_rerun_visualization.py`。
- **配置/脚本**:`rollout.sh`、`CONTEXT.md`。
- **依赖**：无新增；注意力为手写 eager 路径，权重本已实体化。
