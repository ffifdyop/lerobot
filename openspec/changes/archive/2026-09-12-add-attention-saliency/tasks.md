# Tasks: add-attention-saliency

## 1. 注意力截取机制

- [x] 1.1 `smolvlm_with_expert.py`:`SmolVLMWithExpertModel` 增加 per-instance `attention_capture` 属性（默认 None),`eager_attention_forward` 穿入 layer_idx/kind（三处调用点：denoise self、训练 prefix、denoise cross),tap 点为 softmax 输出 cast 前的 float32 probs,None 时仅一次空判断
- [x] 1.2 单测：默认不触发；per-instance 隔离（未设置的实例无调用）；设置后收到 (layer_idx, float32 probs, kind) 且 kind 正确；复位后停止

## 2. Attention 解释器

- [x] 2.1 `SaliencyConfig.method` 增加 `"attention"` 合法值与校验（默认仍 `grad_x_input`，非法值 fail-fast,attention 必须 RTC)
- [x] 2.2 explainer registry 扩展为 `(policy_type, method)` 键（`("smolvla","grad_x_input")` 行为不变）,manager 按 method 取解释器并在副本实例上设置/复位 `attention_capture`（单卡回退同设备可运行）
- [x] 2.3 新增 attention explainer：只消费 kind=="cross" 事件，a₀ query 行 head 维均值后 float32 原地累加，事件均值；聚合数量取自 policy config；按列归属切出每相机 8×8 图与五份预算
- [x] 2.4 布局断言：每相机 64 token、`Lk == 64×相机数 + tokenizer_max_length + 1`，不匹配 fail-fast 走 sink 降级
- [x] 2.5 单测：列归属（cam2 独亮）、原始份额不重归一化、176 触发/80 消费、五份总和=1(1e-4)、布局不匹配降级、同副本同设备回放与直接前向差异 <1e-5（镜像现有 bitwise 测试）、单次失败丢帧/连续 2 次禁用

## 3. 渲染与录制

- [x] 3.1 `overlay.py`：块状最近邻放大 + patch 网格线；letterbox 忠实映射按 `resize_with_pad` 左上 padding 几何从输入尺寸推导（720p：仅帧顶 80px 半高条带、网格线 y=80/240/400/560、底部无条带）,禁止重采样对齐
- [x] 3.2 `overlay.py`：五份堆叠预算条绘制（帧顶 ~24px，百分比标注）
- [x] 3.3 `recorder.py`：按 method 接入新解释器输出；预算条数据随帧传入渲染；单次失败丢帧告警、连续 2 次禁用（沿用现有语义）
- [x] 3.4 `rerun_visualization.py`：新增 5 条预算 scalar 曲线（`saliency/budget/<相机 key>`、`saliency/budget/language`、`saliency/budget/state`）并接入蓝图 TimeSeriesView 使默认可见
- [x] 3.5 单测：热力分量 patch 内分段常数、网格线位置、letterbox 边缘条带几何、预算条数值与分段、rerun 实体与蓝图

## 4. 收尾

- [x] 4.1 `rollout.sh` 切换 `--saliency.method=attention` 并更新注释块
- [x] 4.2 全量测试 `uv run pytest tests/test_saliency.py tests/utils/test_rerun_visualization.py tests/test_interactive_rollout.py` 通过；`uvx ruff` / `bandit` / `typos` 干净
- [x] 4.3 规格回归：逐条对照 `specs/rollout-saliency/spec.md` 场景验证，更新 openspec 状态
