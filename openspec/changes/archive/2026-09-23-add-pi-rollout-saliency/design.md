# π0 / π0.5 rollout saliency 设计

状态：首版已实现，2026-09-23 review 修复完成。本文记录 2026-09-22 访谈决定及实现边界；任务和验证证据分别见 tasks.md、validation.md。术语见根目录 `CONTEXT.md`。

## Goals / Non-Goals

- 为仓库内本地 `pi0`、`pi05` 同时支持 `grad_x_input` 和 `attention`，复用现有 rollout saliency 管线。
- 本次交付包含实现、回归测试、使用文档与验收记录。
- π0.5 MEM 视觉/本体历史暂不支持；启用任一 memory 开关且请求 saliency 时，加载解释器权重前明确报错。普通策略推理不受影响。
- 两种方法统一解释被 RTC 接受的动作块中，实际裁去前缀后的首个策略动作 `a_d`。它不是经过后处理、安全裁剪后发送到电机的 Executed Action，也不保证入队后一定执行。
- 同步修正 SmolVLA 的目标索引，以免三类策略在同一选项下采用不同时间语义。
- attention 保留原始概率份额；π 系列预算增加单个 `action` 段，合并自身与其他动作 token 的份额。π0.5 文本及离散状态合称 `prompt`。
- 支持选择真实零基层号及零基去噪步骤；默认全选。多个选中事件等权平均，首版每相机仍输出一张聚合图、一条预算。
- 同一次重放的多组对比明确列为后续功能。

## Context

| 部分 | 变更前实现 | 扩展影响 |
|---|---|---|
| 注册与生命周期 | `rollout/saliency/explainer.py` 按 `(policy_type, method)` 注册；manager 加载冻结的第二份模型 | 新增四个注册组合，无通用自动回退 |
| RTC 快照 | `rollout/inference/rtc.py` 在控制推理前构造并立即 submit | 改为预先复制、接受入队后提交 |
| 队列 | `policies/rtc/action_queue.py` 在替换队列时裁掉延迟前缀 | 目标索引必须由队列实际裁切结果确定 |
| SmolVLA 梯度 | `smolvla.py` 使用 `actions[0, 0]` | 改为快照提供的索引 |
| SmolVLA attention | `smolvla_attention.py` 只平均 cross 事件，heads 后平均所有事件 | 不是所有模型层；保留此捕获范围并增加筛选 |
| π0 / π0.5 采样 | 两者 core `sample_actions` 带 `@torch.no_grad()` | 外部 `enable_grad()` 本身不能恢复图 |
| π 系列 attention | denoise expert 已强制 eager；每层 self-attention 可取得权重 | 在解释器副本上注册实例级 hook |
| π0.5 MEM | 控制策略内部堆历史，旁路队列会丢快照 | 独立旁路历史不能保证重放相同输入 |
| 图像几何 | manager 只认 SmolVLA 的 `resize_imgs_with_padding` | 增加 π 系列 `image_resolution` 适配 |

π 系列 prefix 使用缓存：每个去噪步骤复用相机/prompt 的 KV，并与该步骤的动作 KV 一起做 attention。缓存不属于跨观测 MEM。

## Decisions

### 原生训练与默认推理边界

现有 SmolVLA attention 也采用独立副本重放，并非读取控制模型当次 attention。其模型改动是默认 None 的实例回调；统计行上的 detach 不截断原 attention 图，冻结参数与 eval 只作用于副本。本方案沿用这个隔离边界，不以解释需求改变原生训练算法。

- 不删除或改写公共推理入口的 no_grad，不全局替换 attention，不修改原生训练 forward、loss、参数可训练性或 checkpoint 格式。
- 尤其禁止直接删除共享 RTCProcessor 中的 detach、改变默认 create_graph/retain_graph，或调整其梯度开关来满足 saliency。需要完整输入导数的改动只允许位于解释器专用路径；若必须提取共享纯计算函数，原路径调用和数值/梯度语义必须保持，单独进行回归审查。
- 解释器加载使用独立配置副本，不临时修改控制模型共享配置。捕获器、step 观察与梯度上下文只绑定未编译解释器实例，close 时释放。
- saliency 关闭时不新增噪声采样、快照复制、模型加载和解释器依赖导入。允许默认关闭的实例观察入口，但不宣称 Python 条件判断绝对零开销。
- 队列仅暴露现有裁切计算所得元数据，不更改锁、裁切/追加算法或入队决策；优先兼容地增加返回信息并审计调用方，不扩展公共 Policy 推理签名。
- saliency 开启会增加资源消耗、改变噪声生成位置，副本初始化也可能消耗 RNG；不保证相同全局 seed 的整次 rollout 轨迹或 RTC 时序不变。输出/梯度等价测试必须固定输入、噪声和生成条件，与资源及时序测量分开。

2026-09-22 的现有 SmolVLA 审查中，saliency 测试 64 项通过；attention 改动前后的小张量 float32/bfloat16 输出及 Q/K/V 梯度一致。这只是既有实现的证据，不替代新增 π 路径的训练、推理和编译回归。

### RTC 快照与动作身份

实现使用以下显式快照字段：

- `preprocessed`、`noise`、`chunk_kwargs`：不可变输入副本；chunk_kwargs 保存生成时真正使用的参数。
- `first_queued_action_index`：实际入队裁切索引 d，独立于生成时的 `inference_delay`。
- `chunk_id`、观测时间、接受时间、reset epoch、View Mask revision：关联与判断快照是否失效。
- 首版不要求快照携带 `target_action`。测试夹具保存控制输出中的 `actions[0,d]`，用于一致性比较；未来显式诊断模式可选携带，不成为默认逐帧门禁。

顺序：

1. 预处理观测，采样噪声，复制输入到局部 pending snapshot。
2. 控制模型按该输入、噪声、原始 RTC 条件生成完整 chunk。
3. 测量耗时，检查 trained RTC 延迟、reset epoch、遮罩 revision 等接受条件。
4. 在现有队列锁语义下完成 merge，返回实际裁切索引及是否存在保留动作。
5. 仅接受且有保留动作时，将 d 与接受身份补进快照，非阻塞提交解释器。
6. 解释器使用原始生成条件重放，再按 d 选目标。渲染使用该次观测图像，而不是生成完成时的新画面。

生成前估计的 `inference_delay` 和队列最终采用的 d 不能互相覆盖。队列应返回实际结果，不能由 saliency 根据耗时再计算一次；append 模式无前缀裁切时，新块对应 d=0。d 超过块长、整个 chunk 被丢弃时不提交；绝不夹到最后一个动作伪造有效目标。

接受入队不等于已经执行。接受后可能发生暂停、reset 或再次替换；旁路输出也有延迟。界面/文档标记为“观测时刻对应的下一入队策略动作”，不宣称是显示时刻正在执行的电机动作。保留最新优先队列，允许少产热力图而不阻塞控制。

Shadow Saliency 保留其独立反事实语义，不能为了满足本方案而让 shadow proposal 入执行队列。本次正常入队契约不规定 shadow 的目标索引；后续人工接管功能接入时需显式提供 mode 与反事实目标选择，避免套用“入队后提交”导致 shadow 消失。

### 梯度方法

两种 π adapter 分别复用各自的图片、state/prompt 准备逻辑；仅实际存在的相机图像设为梯度叶子，参数冻结。裁剪动作到实际动作维度，再计算：

```text
score = sum_j actions[0, d, j]^2
map[p] = sum_c abs(image[c,p] * d(score)/d(image[c,p]))
```

输入空间、动作归一化与现有 SmolVLA 保持一致；不对后处理后的物理单位动作求梯度。该图是局部敏感度，不能当成不同 checkpoint 之间校准过的绝对贡献。

建议首版只在未编译的解释器副本上显式调用 `sample_actions.__wrapped__`，复用采样函数体并绕过 no-grad 包装；先验证包装契约，不修改控制模型全局梯度设置，不复制一套独立 denoise 算法。

关键验证门槛：guided RTC 的 `RTCProcessor.denoise_step` 有 `detach()` 及内部 autograd，数值重放一致不代表外层梯度是整个采样过程的全导数。必须用微型模型与有限差分检查有 leftover 的路径；若存在梯度截断，提供数值等价、保持外层计算图的解释器专用路径，并测试控制输出不变。原生 RTC 的截断保持不动；解释器不得借调整梯度而修正或更换原算法，前向纠偏值也必须等价。若暂不能同时满足前向等价与完整导数，该梯度路径不得标为支持，也不得静默回退截断结果；记录阻碍并重新评审，不扩大修改训练链路。该门槛也覆盖沿用 guided 路径的 SmolVLA。本文没有声称现有路径已验证通过。

trained RTC 的固定前缀对当前图像可能没有梯度；在实际 d 落入固定区域时允许真实零敏感度，并记录上下文，不擅自改变目标索引。

### Attention 布局与捕获

| 策略 | 目标 query 行 | key 分类 | 捕获层 |
|---|---:|---|---|
| SmolVLA | d | 相机、language、state | 现有 cross-attention 层 |
| π0 | 1+d | 相机、language、连续 state、action | action expert self-attention 层 |
| π0.5（无 MEM） | d | 相机、prompt、action | action expert self-attention 层 |

π0 的额外 query 行来自 suffix 首位连续 state token。action 分类覆盖整个预测 chunk，含对自身的注意力。π0.5 不按文本字符串猜测 state token 边界。

在解释器副本的 expert attention 模块安装 hook，读取权重并即时选 query/平均 heads；不要积存全部矩阵，也不要全局 monkey-patch。close 时卸载 hook。不能依赖当前 `PiGemmaModel.output_attentions` 通路：decoder 丢弃权重的返回约定需要单独核实。

grid 从视觉配置和实际 token 数推导并校验：标准 π 配置通常是 224 图像、14 patch、16×16 网格；SmolVLA 当前为 8×8。相机布局要跟随真实图片准备顺序，包括尾部缺失相机占位，不能只用存在相机数量推断 prefix 总长。padding mask 对应列应无质量；缺失视角不伪造可显示画面。

letterbox 目标由 adapter 提供统一 `(height,width)`，复用 overlay 裁切；显式校验非方形原始相机的空间映射。

### 单组层/步骤选择

已实现配置：

```python
attention_layers: list[int] | None = None
attention_steps: list[int] | None = None
```

- None 代表全部合法事件；空列表、重复、负数、越界或非可捕获层均在启动时报错，列出合法范围。输入顺序不改变权重。
- 层号为真实零基层号；SmolVLA 使用当前捕获回调的真实层号，且只接受产生 cross-attention 的层。
- step 0 为第一次去噪、`num_inference_steps-1` 为最后一次网络求值；这里的 step 不是机器人控制步，也不是动作索引 d。
- 选择器仅适用于 attention；grad 模式配置非默认选择器时报错，避免用户误以为梯度也被筛选。
- 在采样循环的真实 step 边界提供副本专用观察信息。不能用“hook 次数除以层数”猜 step，以免 RTC 包装或重复调用造成错位。

令 S、L 为选中步骤和层，H_l 为该层 head 数：

```text
M[p] = mean_s(mean_l(mean_h(A[s,l,h,query(a_d),p])))
```

即所选层×所选步骤的笛卡尔积，各组合等权。正常路径应每组合恰好一次；遗漏或多次捕获须显式处理/报错，不按未知事件数静默改变权重。预算与 patch 图共享同一聚合结果，预算总和在浮点容差内为 1。

“预算容差”仅表示各类 attention 份额相加与 1 之间允许的数值舍入误差，不是计算/显存预算。SmolVLA 捕获 cast 前 float32 权重；π hook 可能取得 cast 后低精度权重，转回 float32 不能恢复丢失精度。归约统一 float32，但测试阈值按实际捕获 dtype/路径固定并记录：float32 路径沿用 1e-4，低精度阈值根据受控舍入测试确定，不统一强套 1e-4，也不自动放宽阈值掩盖漏列。不增加每帧总和超差自动禁用机制；保留现有非有限值防护，结构正确性以布局检查和测试保证。

捕获、选择和聚合保持轻量分离：捕获输出带 layer/step 身份的目标 query head 均值，单组 accumulator 决定是否接收并累加；预算和地图从同一结果派生。首版不实现多组管理框架、不缓存所有事件矩阵，后续可把同一事件分发给多个 accumulator，无需每组重放模型。

三模型共用 `rollout/saliency/attention.py` 的 `AttentionAccumulator`，无步号、重复和遗漏的选中事件均报错。π 布局直接对照本次重放捕获的 key/query 尺寸校验，不为计数单独运行视觉编码器。跨设备重放在 adapter 中迁移 batch、noise 和 RTC 张量参数，不改原生 RTC 设备处理。

先平均原始 attention，再做显示色阶；不能先把各事件图像份额归一化。保留现有 run/chunk 色阶，但颜色是显示尺度，预算条才表示原始质量；跨运行比较不能仅凭独立自适应色阶判断大小。

画面标注实际层集合、步骤集合、mean、动作索引与观测关联信息。未指定时标为“全程 attention 总览”。全选均值描述平均信息路由，不代表所有阶段相同，也不证明浅层/深层、早期/末期的固定语义分工。

选择器不减少完整模型前向步骤；只减少被捕获/累加的数据。当前 eager 路径仍可能在未选层物化权重。单组筛选的开销原则上不高于同一捕获路径全选，具体时延与显存需测量。

## Migration Plan

按下列顺序实现，具体可勾选任务见 tasks.md。必须保留控制路径原有行为；禁止将新快照索引缺失静默解释为 a₀。回滚时整体回滚快照生产者和消费者；若无法保持兼容，禁用 saliency 后继续控制，禁止混用两代快照。

1. **RTC 身份契约**：让 ActionQueue merge 返回裁切结果；RTC 两阶段提交快照；explainer 接收显式目标索引，默认调用方不可偷偷回退 a₀。更新 SmolVLA 两种方法及原先要求推理前 submit 的测试。
2. **π 梯度 adapter**：新增 π0/π0.5 注册、输入适配、no-grad 绕过、重放比较；验证并解决 guided RTC 外层梯度风险。
3. **π attention adapter**：实例 hook、真实 step 标记、query 偏移、动态 token/grid 校验及生命周期清理。
4. **共同选择器与预算**：为三类策略增加单组筛选；预算分类显式区分 camera/language/state/prompt/action，避免把 action 误当相机。
5. **显示与兼容性**：manager 尺寸适配、选中范围标签、MEM 提前拒绝、保留 nonblocking 与故障隔离。保持轻量注册，避免默认配置 import 引入可选模型依赖。
6. **文档与验证**：更新 CLI 使用说明、输出含义、资源需求和测试。真实模型验证完成前不把功能标为已支持。

继续采用第二份完整模型、eager 重放、最新优先队列；π 模型副本可能占用较多显存，启动打印设备及配置，实测峰值，OOM 时明确失败，不能承诺单卡不影响控制。

重放即“用保存的条件在副本上再算一次”，以在旁路收集 attention 或计算图像梯度，不把第二次动作交给机器人。重放一致性是测试/离线验收：固定观测、噪声及 RTC 条件，对比控制输出和副本输出，确认解释的确对应同一次生成。它不要求默认逐帧保存目标动作、比较、告警并跳过。测试分别声明 dtype、设备、compiled/eager 及 atol/rtol，超差应使该验收失败并定位原因，不能宣称跨设备逐位一致。可选运行时诊断留作后续，不新增配置或门禁。

## Risks / Trade-offs

- guided RTC 梯度截断 → 必须验证有限差分，数值重放和梯度正确性分别验收。
- π 模型显存与旁路延迟 → 测量峰值与控制节拍，允许丢帧，禁止阻塞控制。
- 多阶段平均掩盖差异 → 提供真实层/步骤筛选和画面标签，默认均值只作总览。
- 与 `add-rollout-human-takeover` 同改 saliency 规格 → 2026-09-22 在任务 5.6 中协调其 saliency delta：Policy 使用 a_d，Shadow 独立提交并显式选择 proposal a₀。Shadow 不伪造队列接受元数据，后续实现模式感知的 recorder 入口；本变更仍不实现人工接管。归档任一变更时保留另一变更的层/步筛选、完整梯度与降级要求。

### 验收标准

- 三类策略两种方法的目标均随实际 d 变化；π0 query 为 1+d，π0.5/SmolVLA 为 d。使用非零 d 的合成数据检测错行。
- 生成前 delay 与入队 d 不同时，前者仍用于重放，后者仅用于目标选择。
- reset、过时遮罩、trained RTC 拒绝、无剩余动作均不提交；接受到提交之间的失效按 epoch/revision 丢弃。旁路失败不改变控制结果。
- 测试中固定观测、噪声、RTC 条件，副本目标动作与夹具保存的控制输出一致；覆盖首块、guided leftover、π0.5 trained 前缀及实际 checkpoint 精度，不要求生产快照携带控制输出。
- 梯度在小模型上通过有限差分检查，包括 guided leftover；无影响/固定目标允许零图，不能把所有零图当成功或失败。
- attention 单层单步、任意多层多步、全选默认与手算一致；未选事件不累加；预算与地图选择一致。测试 heads 数、层数、step 数可变。
- π0 的独立 state、π0.5 prompt、action 自身及其他列都正确计入预算，未掩码总和约为 1；布局不匹配报错。
- 相机顺序、缺失相机占位、不同分辨率、letterbox、View Mask、hook 实例隔离与卸载均有覆盖。
- MEM 两个开关分别测试提前拒绝；关 saliency 不影响 MEM 策略使用。
- 对任何改动到的共享模型/RTC 函数，以改动前基线固定 batch、noise、time 及随机状态，对照训练 loss、应训练参数的梯度和冻结状态；覆盖默认训练与相关冻结/梯度检查点配置。不得新增参数或 state_dict 键。
- 关闭 saliency 时对照默认推理输出、队列裁切结果；测试无副本加载/额外噪声/快照/捕获。另测控制模型与副本的配置、参数、hook 隔离以及 compiled 控制入口回归；编译性能需实测，不以数值一致代替。
- 实际设备短运行记录解释器耗时、控制节拍、丢帧率以及 Rerun/视频结果。显存峰值允许采用同 checkpoint 的离线单模型实测作为本次替代验收（2026-09-23 用户确认），但必须列明设备、精度、方法，并明确它不是控制模型、解释器与 Rerun 同时运行的实机峰值，不能据此承诺整机资源上限。解释器允许丢帧，不要求每个已接受 chunk 都被绘制。

测试覆盖 `tests/test_saliency.py`、`tests/test_interactive_rollout.py`、π0/π0.5 RTC 及训练基线；真实 checkpoint、compiled 入口和受限实机验证证据见 validation.md。不同阶段的精度、设备和替代验收范围分别记录，不将微型模型测试等同于实机验证。

发布边界：先渲染全部相机，再在各视频/Rerun 写入入口复查 epoch 与 View Mask revision。计算、渲染或 writer 初始化期间失效的快照丢弃，不计为计算故障。已经开始的外部 I/O 无法撤回；若在一次写入期间发生 reset，停止尚未开始的后续输出，不为跨输出事务而持有控制线程锁或阻塞控制。

## Future Work

**同一次重放的多组对比**：配置多个命名的 `(layers, steps)` 选择组，如 early/late；共享一次前向，每组独立累加器，分别输出热力图与预算，并采用可比较的显示尺度。基于首版事件筛选即可扩展，仍需增加输出命名、视频/Rerun 生命周期和资源上限；不承诺增加组数完全无开销。

**MEM**：控制策略物化历史后，执行与解释共享同一完整 temporal batch。需另行定义历史帧敏感度与当前帧显示关系；不得让会丢帧的旁路自己重建历史。

**实际执行前缀归因**：增加动作消费追踪，待块退出后确定实际执行集合，提供延迟/离线聚合。与首版下一入队动作语义区分。

## Open Questions

用户侧首版范围与输出选择已确定，无需重新访谈。实现阶段需验证 guided RTC 完整梯度、实际 checkpoint 布局/精度与设备资源；这些是验收门槛，不是默认已经满足的事实。Shadow 的独立目标选择由人工接管变更协调，本变更不实现 Shadow 功能。

## References

跨层表示会混合来自不同 token 的信息，原始 attention 均值不是最终动作的严格因果归因。层/步骤选择器用于检查差异，不预设固定的语义/精细分工。参考：[Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)。attention rollout 等方法需单独论证缓存、残差与 action expert 路径，首版不引入。
