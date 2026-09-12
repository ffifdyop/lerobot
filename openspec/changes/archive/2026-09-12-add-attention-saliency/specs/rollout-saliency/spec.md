# Spec: rollout-saliency

## ADDED Requirements

### Requirement: 方法选择与校验

`--saliency.method` SHALL 接受 `"grad_x_input"` 与 `"attention"` 两个合法值，默认值 MUST 为 `"grad_x_input"`（向后兼容）。非法值 MUST 在配置校验期 fail-fast 报错。`attention` 方法 MUST 沿用现有硬约束：仅 RTC 推理可用（精确回放依赖噪声注入）,display 开启时要求 `--display_mode=rerun`。

#### Scenario: 默认方法不变

- **WHEN** 用户仅传 `--saliency.enabled=true` 而不指定 method
- **THEN** 方法解析为 `grad_x_input`，行为与本变更前逐位一致

#### Scenario: 非法方法值被拒绝

- **WHEN** 用户传 `--saliency.method=foo`
- **THEN** 配置校验抛出明确错误并列出合法值，rollout 不启动

#### Scenario: attention 方法在非 RTC 推理下被拒绝

- **WHEN** 用户传 `--saliency.method=attention` 且推理策略非 RTC
- **THEN** 配置校验 fail-fast，提示 attention 方法需要 RTC 的噪声注入

### Requirement: 像素敏感度图语义（grad_x_input，既有行为记录）

`grad_x_input` 方法 SHALL 对每个相机产出像素级地图：以与执行完全相同的 batch、noise 与 chunk 参数精确回放，分数为 a₀ 的平方 L2 范数，地图为 `(grad * img).abs().sum(channel)`。该语义 MUST 不受本变更影响。

#### Scenario: 动作值加权语义保持

- **WHEN** 指令幅度接近零（姿态过均值点）
- **THEN** 热力图整体变暗，且实现不产生告警或 NaN

### Requirement: 注意力截取与隔离

SmolVLA 的 eager attention 路径 SHALL 支持 per-instance 的可选 capture callback（模型实例属性，签名 `fn(layer_idx, probs, kind)`,kind ∈ {self, cross}),probs MUST 为 softmax 输出、cast 回 value dtype 之前的 float32 张量；callback 默认 MUST 为 None 且此时零拷贝。callback MUST 只设置在 saliency 第二副本的模型实例上，**禁止**设置在控制回路模型实例上（per-instance 属性使隔离由构造保证）；禁止模块级/进程级全局注册；manager 停止时 MUST 复位。explainer MUST 只消费 `kind == "cross"` 的事件（cross 调用构造上只出现在 denoise 奇数层），其余事件直接忽略；每次消费 MUST 原地归约（head 维均值后累加进 float32 累加器并计数），禁止保留全量注意力张量。

#### Scenario: 控制回路零开销

- **WHEN** saliency 未启用，或控制回路模型实例上未设置 callback
- **THEN** 该实例的每次注意力调用不触发任何 callback，每次调用仅一次 None 判断

#### Scenario: 只消费 cross 事件

- **WHEN** 对 explainer 副本回放一个 chunk（当前 checkpoint:num_steps=10、16 层、self_attn_every_n_layers=2)
- **THEN** callback 共触发 176 次（16 次 prefill self + 80 次 denoise self + 80 次 denoise cross)，其中恰好 80 次 cross 事件被消费，prefill 与偶数层事件被过滤，每次只消费 a₀ query 行

### Requirement: Patch 注意力图语义

`attention` 方法 SHALL 对每个相机产出 8×8 patch 地图：a₀ query 的 cross-attn 概率对全部 head、全部 denoise step、全部奇数层取均值（分母取自实际消费的 cross 事件数——每奇数层每 step 恰一次事件，构造上等价且禁止硬编码；当前 checkpoint 对应 15×10×8)，按列归属切片（每相机 64 列连续排布，顺序与 prefix 中相机顺序一致）。地图 MUST 为原始份额，禁止在图像 token 内重归一化。实现 MUST 运行时断言每相机恰好 64 个 token 且 `Lk == 64 × 相机数 + tokenizer_max_length + 1`，不匹配 MUST fail-fast 并走 sink 降级（禁用 saliency、告警、控制回路继续）。letterbox 目标尺寸 MUST 从 policy config 的 `resize_imgs_with_padding` 读取并注意其 (width, height) 约定；该方法为 `None` 时 attention 方法 MUST 在启动期 fail-fast（无定义的 patch→pixel 映射），禁止静默错位渲染。

#### Scenario: 列归属正确

- **WHEN** 构造假注意力使第 2 个相机的 64 列质量为 1、其余为 0
- **THEN** 仅 camera2 的 8×8 图全亮，camera1/camera3 全暗

#### Scenario: 原始份额保留绝对量

- **WHEN** 某 chunk 中 a₀ 对全部 192 个图像列的总质量仅 0.2（其余看向语言/state)
- **THEN** 三个相机地图整体偏暗（均值 ≈ 0.2/64 量级），不被拉满对比度

#### Scenario: 布局不匹配时 fail-fast

- **WHEN** 注意力 key 长度与「64×相机数 + 语言长 + 1」不符
- **THEN** 抛出明确错误，saliency 降级为禁用并告警，控制回路继续运行

### Requirement: 注意力预算条

`attention` 方法 SHALL 计算 a₀ query 的 cross-attn 质量五份分配：cam1 / cam2 / cam3 / 语言 / state；以 float32 截取与累加，五份之和 MUST 等于 1（容差 1e-4)。预算条 MUST 以堆叠横条（带百分比标注）烧录进每路相机帧顶部，并 MUST 在 rerun 记录 5 条 scalar 曲线（`saliency/budget/<相机 key>` × 3、`saliency/budget/language`、`saliency/budget/state`）且在默认蓝图中可见。取值为每 chunk 原始值，禁止时间平滑。

#### Scenario: 五份总和为 1

- **WHEN** 任一 chunk 的预算条计算完成
- **THEN** cam1+cam2+cam3+语言+state 五份之和与 1 的偏差不超过 1e-4

#### Scenario: 视频自包含

- **WHEN** 播放任一相机的 saliency mp4
- **THEN** 每帧顶部可见五段堆叠条与百分比，无需打开 rerun 即可读注意力预算

### Requirement: Patch 图渲染

Patch 注意力图 SHALL 以最近邻块状放大叠加（禁止双线性/高斯模糊），热力分量在 patch 内分段常数、patch 间无插值渐变，patch 边界 MUST 画网格线。letterbox 映射 MUST 按 `resize_with_pad` 的真实几何（左、上 padding，从输入尺寸与 `resize_imgs_with_padding` 推导，禁止假设居中 padding、禁止硬编码）：每个 patch 的值只渲染它与显示帧内容区的交集，禁止将 token 网格对齐内容区重采样；纯黑边行 token 不渲染，但其质量仍计入对应相机的预算条份额。色标 MUST 沿用 `norm_scope`(run/chunk）机制。

#### Scenario: 块状而非平滑

- **WHEN** 8×8 地图叠加到 720p 帧
- **THEN** 热力分量在 patch 内分段常数，patch 间有可见网格线，patch 边界两侧无插值渐变

#### Scenario: letterbox 忠实映射

- **WHEN** 相机为 1280×720(resize_with_pad 到 512×512：仅顶部 224px 黑边，内容底对齐，覆盖 patch 行 3.5–8)
- **THEN** 仅帧顶出现半 patch 高度（80px）的窄条带（patch 行 3 的下半），帧内水平网格线位于 y = 80/240/400/560（真实 token 边界），底部无条带、无整体网格平移

### Requirement: 精确回放、降级与防护

`attention` 方法 MUST 使用与 `grad_x_input` 相同的快照与噪声注入机制，回放与执行一致；回放 MUST 在 `torch.no_grad` 下进行。capture 或计算失败 MUST 沿用 sink 降级路径：单次失败丢帧并告警，连续 2 次失败禁用 saliency，绝不打断控制回路；非有限地图 MUST 沿用 NaN guard 丢帧；非有限预算（NaN 落在语言/state 列而相机地图有限）MUST 仅丢弃该帧的预算条与预算曲线（warn-once 告警），热力图帧 MUST 保留。

#### Scenario: 回放与直接前向一致

- **WHEN** 在同一 explainer 副本、同一设备上，比较快照回放与直接前向两种方式的 capture 结果
- **THEN** 二者差异小于 1e-5（同一 batch、同一 noise、同一 chunk 参数；跨设备存在 ulp 级漂移，不做跨设备逐位断言）

#### Scenario: capture 失败不中断 rollout

- **WHEN** capture callback 抛出异常
- **THEN** 当帧被丢弃并告警；若连续 2 次失败则 saliency 禁用；策略控制回路始终不受影响
