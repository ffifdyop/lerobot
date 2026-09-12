# Design: add-attention-saliency

## Context

`feat/rollout-saliency` 已实现 `grad_x_input` 热力图：RTC 引擎在 `predict_action_chunk` 前注入噪声并打快照（`inference/rtc.py:551-591`),`SaliencyManager` 加载策略第二副本（`saliency/manager.py:62-67`，设备解析 `resolve_explainer_device` 优先第二块 GPU、**单卡时回退到策略设备**,attention 方法无多卡硬性要求）,`SmolVLAExplainer` 精确回放并以 ∂‖a₀‖²/∂x 产出像素级地图，`SaliencyRecorder` 侧线程渲染叠加视频与 rerun 流。

SmolVLA 的 action→image 注意力是手写 eager 路径（`smolvlm_with_expert.py:515-560`),`probs` 每次调用都实体化为 (B, 15, Lq, Lk)，且 softmax 输出为 float32（第 551 行，第 552 行才 cast 回 value dtype)。每个 chunk 的去噪阶段（`config.num_steps` 个 Euler 步，当前 checkpoint 为 10)：偶数层走 joint self-attn(softmax 分母含 50 个 action token 自身），奇数层走纯 cross-attn(softmax 只覆盖 prefix key);prefill 阶段全部 16 层都走 self-attn 路径，**构造上不产生 cross-attn 调用**。图像 token 按相机连续排布：camera1 [0:64]、camera2 [64:128]、camera3 [128:192]，每相机 8×8 patch 网格（letterbox 512×512 经 pixel_shuffle scale 4)；语言 48 token（带 padding mask，依赖 `pad_language_to="max_length"`),state 1 token。

**letterbox 几何（已对照代码核实）**:SmolVLA 的 `resize_with_pad`(`policies/common/vla_utils.py:218-242`）在**左、上** padding。1280×720 → 512×512:ratio=2.5，内容 288×512 底对齐，**仅顶部 224px 黑边**；内容区覆盖 patch 行 3.5–8(4.5 行）——patch 行 0-2 纯黑边，行 3 上半黑边/下半内容，行 4-7 纯内容；底边恰与 patch 边界对齐。显示帧（720p）精确对应内容区 ×2.5，帧顶出现半 patch 高度（80px）窄条带，帧内水平网格线在 y = 80, 240, 400, 560，底部无条带。

## Goals / Non-Goals

**Goals:**

- `--saliency.method=attention` 产出每相机 8×8 的 Patch 注意力图与五份注意力预算条，与 `grad_x_input` 共存且默认不变。
- 控制回路零开销：capture 必须只在 explainer 副本实例上生效。
- 与现有 saliency 基础设施（快照、噪声注入、recorder、色标、mp4/rerun 输出）完全复用。

**Non-Goals:**

- 不改 `grad_x_input` 行为；不做远程策略解释；不做 vision encoder 内部注意力、integrated gradients、per-dim 分解。

## Decisions

### D1 — 共存而非替换：`method` 字段新增 `"attention"`

`SaliencyConfig.method` 已有预留位设计。新增 `"attention"` 为合法值；默认保持 `grad_x_input`（向后兼容）。`rollout.sh` 显式切换并更新注释。

### D2 — 截取机制：per-instance capture callback，禁止全局注册与 monkeypatch

在 `SmolVLMWithExpertModel` 上增加**实例属性** `attention_capture`（默认 `None`)；签名 `fn(layer_idx, probs, kind)`,`kind ∈ {"self", "cross"}`,`probs` 为 softmax 输出、**cast 回 value dtype 之前**的 float32 张量。`eager_attention_forward` 需穿入 `layer_idx` 与 `kind`（调用点：`forward_attn_layer` 的 denoise 分支与训练用 prefix 分支、`forward_cross_attn_layer`，共三处）;callback 为 `None` 时零开销（一次 `is None` 判断，无拷贝）。

- callback **必须**只设置在 saliency 第二副本的模型实例上，**禁止**设置在控制回路模型实例上——per-instance 属性使隔离由构造保证（控制回路实例属性恒为 `None`)，不依赖约定；manager 停止时将副本实例属性复位为 `None`。
- 禁止模块级/进程级全局 registry:recorder 线程与控制线程并发跑同一个 16 层前向，全局 callback 会被控制回路触发并污染 explainer 累加器。
- 备选（monkeypatch 该函数）被否：隐式、脆、无法静态检查。forward hook 不适用：注意力是普通函数而非 `nn.Module`。

### D3 — 只取奇数层纯 cross-attn

奇数层 softmax 只覆盖 prefix key，注意力质量在 图像 / 语言 / state 间分配，总和恒等于 1——语义干净，且使预算条天然归一。偶数层 joint self-attn 被 action 自注意力稀释，**禁止**混入（备选「全部层平均」「只用偶数层」均被否）。explainer 的过滤规则为 `kind == "cross"`:cross 调用构造上只出现在 denoise 奇数层（prefill 全走 self 路径），无需再按 query 数过滤。

### D4 — 聚合:a₀ query × heads 均值 × 事件均值，原始份额

- query 只取 a₀(50 个 suffix query 的第 0 行）——与 `grad_x_input` 的「当前执行动作」语义对齐（备选「全部 query 均值」被否）。
- 每次 cross 事件先在 head 维取均值，再对全部 cross 事件（奇数层 × denoise steps，当前 checkpoint 为 8×10=80 次）取均值；禁止硬编码 head 数、层数、步数。**实现备注（review 后修订）**：均值分母取自实际消费的 cross 事件数——每个奇数层在每次 denoise step 恰好触发一次 cross 事件，事件均值在构造上等价于「heads × steps × 奇数层」的平铺均值，且与未来 checkpoint 的层结构变化零漂移；这比从 config 字段（`num_attention_heads` / `num_vlm_layers` / `self_attn_every_n_layers` / `num_steps`）推导更稳健，config 只用于布局断言（D6)。
- **禁止**在图像 token 内重归一化：原始份额 ≤ 1，模型看语言多时整图变暗是正确行为，与「弱相机保持暗」的色标语义一致。

### D5 — 原地归约，禁止保留全量注意力

callback 内只做切片累加：`acc += probs[0, :, 0, :].mean(dim=0)`((Lk,),float32)，事件计数 `n += 1`;chunk 结束时 `均值 = acc / n`。当前 checkpoint 每 chunk 仅 80 次 cross 事件，显存开销可忽略；callback 对 `kind == "self"` 的事件直接返回。

### D6 — 列归属：运行时推导 + fail-fast 断言

图像列数 = 64 × 相机数（SmolVLM connector pixel_shuffle scale 4:32×32 patch → 8×8 token，必须断言）；语言列数 = `tokenizer_max_length`(padding 位已被 mask，质量自然为 0);state = 最后 1 列。回放时必须断言 `Lk == 64×相机数 + tokenizer_max_length + 1`，不匹配则抛错走 sink 降级（禁用 saliency 并告警），禁止静默错位。

**部署约束**：该布局依赖 checkpoint 的 `pad_language_to="max_length"`（当前 checkpoint 已设；config 默认 `"longest"` 会产生变长语言块，触发断言 fail-fast——此为有意行为）与 `add_image_special_tokens=false`、`prefix_length=0`。

### D7 — 渲染：块状放大 + 网格线 + letterbox 忠实映射

- 8×8 图最近邻放大为块状叠加（**禁止**双线性/高斯模糊——不虚构像素级精度）,patch 边界画网格线；热力分量在 patch 内分段常数（与底图 alpha 混合后 patch 内仍随底图变化，无插值渐变）。
- letterbox 映射**必须**按 `resize_with_pad` 的真实几何（左、上 padding)：每个 patch 的值只画它与显示帧内容区的交集。720p 相机下：仅帧顶出现半 patch 高度窄条带（patch 行 3 的下半），底部无条带；**禁止**假设居中 padding，禁止把网格强行对齐内容区重采样。
- 纯黑边行 token 不渲染，但其注意力质量仍计入该相机的预算条份额（预算按 token 归属统计，与可见性无关）。
- 渲染几何必须从输入尺寸与 `resize_imgs_with_padding` 配置推导（与 `resize_with_pad` 同一份 ratio/pad 逻辑），禁止为 720p 硬编码。**实现备注（review 后修订）**：该配置按 (width, height) 存储（`modeling_smolvla.py` 传 `[1]`/`[0]` 给 `resize_with_pad`)，而 overlay 数学用 (h, w)——manager 是唯一接缝，负责交换；`None`（关闭 letterbox）时 patch→pixel 映射无定义，attention 方法启动期 fail-fast。

### D8 — 注意力预算条：五份堆叠，烧录帧顶 + rerun scalars

a₀ query 的 cross-attn 质量五份分：cam1 / cam2 / cam3 / 语言 / state（总和=1，容差 1e-4;float32 截取与累加保证可达，测试断言）。以约 24px 高堆叠横条烧录进每路相机帧顶部（带百分比标注；分段配色按**类别**：相机按绘制顺序循环三色，language/state 固定中性灰，与相机数量无关）;rerun 记录 5 条 scalar 曲线（`saliency/budget/<相机 key>` × 3、`saliency/budget/language`、`saliency/budget/state`)，且必须在默认蓝图中可见（TimeSeriesView)。取值为每 chunk 原始值，不做时间平滑。分段命名全链路统一用 `language`/`state`（帧内标注与 rerun 实体一致）。

### D9 — 分派、约束与降级路径

- explainer 分派：现有 registry 按 policy_type 单注册（重复注册抛错），必须扩展为 `(policy_type, method)` 键——`("smolvla", "grad_x_input")` 保持现有注册行为不变，新增 `("smolvla", "attention")`;manager 按 `config.method` 取解释器，method=attention 时在副本实例上设置/复位 `attention_capture`。
- `attention` 方法沿用现有硬约束：必须 RTC（精确回放需要同一 noise)、display 开启时要求 `--display_mode=rerun`；单卡可运行（第二副本与策略同设备）。
- sink 失败降级：单次处理失败丢帧并告警，**连续 2 次失败**禁用 saliency，绝不打断控制回路；非有限值沿用 NaN guard 丢帧；**非有限预算**（NaN 落在语言/state 列而相机图有限）仅丢弃该帧预算条与曲线（warn-once)，热力图帧保留。`norm_scope`(run/chunk）机制不变——attention 份额 ∈[0,1] 天然适配 run-scope 在线最大值。回放在 `torch.no_grad` 下进行。

## Risks / Trade-offs

- [a₀ 单 query 的地图偏噪] → heads × steps × 层三重均值；未来如需可加 query 均值 flag（本期不做）。
- [callback 被误设到控制回路实例] → per-instance 属性 + 设置入口收敛在 `SaliencyManager`；测试断言未设置的实例不触发任何调用。
- [transformers 升级改变 connector/patch 布局，或 checkpoint 改 `pad_language_to`/`add_image_special_tokens`] → D6 的 fail-fast 断言把隐式错位变成显式报错（saliency 禁用、rollout 继续）。
- [8×8 分辨率信息量少，叠加后可能看不出目标物体轮廓] → 接受：patch 级注意力本就只有这个分辨率；网格线让 patch 语义可读。
- [预算条烧录进帧顶会遮挡少量画面] → 24px 条带位于 720p 帧顶部，遮挡 < 4%；不接受只做 rerun（视频需自包含可回看）。
- [probs 在 cast 后被 tap 会退化为 bf16，五份总和可能超出 1e-4 容差] → D2 规定 tap 点为 cast 之前的 float32 softmax 输出，累加保持 float32。

## Migration Plan

无状态迁移。默认 `method=grad_x_input` 不变，存量行为逐位保持；`rollout.sh` 显式加 `--saliency.method=attention`。回滚 = 摘掉该 flag。

## Open Questions

无。术语已于 2026-09-11 随 `CONTEXT.md` 更新决议（热力图 = 家族术语；像素敏感度图 / Patch 注意力图 / 注意力预算条）。
