## Context

LeRobot 折布脚本使用 v3 版 `so101_bimanual_folding-cloth-720p`；本地 openpi 配置引用其 v2.1 版。按用户确认，两者是同一批数据的格式转换，本变更固定使用 v3，不再比较两版数据。LeRobot 当前脚本使用 seed 1000、验证 episode `[0,95]`，且未启用图像增强和 EMA。openpi 当前启动脚本采用 seed 42、40,000 步、`val_episodes=3`，其验证 episode 为 `[57, 21, 75]`。归档时用户确认默认每卡 batch 6、两卡全局 batch 12，checkpoint 保存间隔为 20,000 步。v3 `meta/stats.json` 当前仅含 min/max/mean/std，缺少 quantile 归一化所需的 q01/q99。

openpi 在模型图像预处理处先缩放补边到 224×224，再逐图增强：非腕部图像随机裁剪宽高的 95%、缩回原尺寸、随机旋转 ±5°；所有相机调用 `augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)`。本地 augmax 另有默认 `hue=0.1`、`p=0.5`，且 saturation 分支未把返回值赋回，因此当前实际没有饱和度变化。裁剪位置随机带来小幅平移；没有独立 `Translate` 算子。LeRobot 通用 `ImageTransforms` 默认随机抽取变换子集，无法仅靠打开开关复现该固定顺序及腕部相机例外。

动作 loss 也存在目标差异：openpi `Pi0Config.action_dim=32`，SO-101 原始动作 12 维经 `PadStatesAndActions` 补到 32 维；JAX 对 32 维逐元素平方误差先按动作维取均值，再对 batch 和 horizon 取均值。LeRobot 将动作补到 `max_action_dim=32`，但在 reduction 前把 loss 截回真实 12 维。因此 JAX 还对 20 个 padding 维训练，目标为对应噪声，且 12 个真实动作维的梯度按 32 维均值缩放；LeRobot 忽略这些维度，真实维度按 12 维均值。两者差异会改变梯度和共享主干更新，不只是日志尺度。

## Goals / Non-Goals

**Goals:**

- 使用 LeRobot v3 数据，对齐折布配方的学习率、归一化、EMA、随机种子及验证划分。
- 增加明确的增强开关及逐项强度配置：一般 π0.5 默认关闭；折布脚本默认开启并采用 openpi 参数，允许用户关闭或调整裁剪、旋转及有效颜色扰动。
- 增强必须在通用数据图像变换管线逐图采样，只挂载到训练数据集；验证和推理必须保持无增强。
- 折布配方可通过 π0.5 配置项启用 JAX 的 32 维动作 loss，保持其他 π0.5 配方既有目标。
- 保持 LeRobot 通用训练器、数据集公开 API 和策略方法签名不变。

**Non-Goals:**

- 不增加跨实现 batch 校验，也不比较 v2.1 与 v3 数据内容。
- 不追求跨 JAX/PyTorch 同随机种子的逐像素一致；对齐本地 openpi/augmax 的有效变换、顺序、范围、概率及相机规则。
- 不修改 Gemma/SigLIP、attention、通用训练器或数据集公开 API；不改变其他策略或其他 π0.5 配方的默认行为。

## Decisions

1. **将配方限定在 SO-101 折布训练入口。** 保留一般 π0.5 默认值，通过脚本显式覆盖 seed 42、`4e-5` 峰值学习率、1,000 步 warmup、40,000 步衰减到 `4e-6`、40,000 优化步、quantile 状态/动作归一化和 EMA `0.999`；使用现有 AdamW `(0.9,0.95)`、`eps=1e-8`、weight decay `0.01`、clip 1.0。按用户确认将默认每卡 batch 设为 6、两卡全局 batch 12，checkpoint 保存间隔设为 20,000 步。备选是修改全局 `PI05Config` 默认值；会影响其他任务，因此不采用。
2. **增强放在通用数据图像变换管线。** 折布脚本通过 `dataset.image_transforms.enable` 总开关和既有 `ImageTransformsConfig.tfs` 注册的 `OpenPIImageAugmentation` 的 `kwargs` 配置六个强度字段，默认分别为 0.95、5°、0.5、0.3、0.4、0.1。通用默认关闭；折布入口把它设为唯一变换并通过既有随机变换容器执行，先将训练图像缩放补边至 224×224，再按相机键区分：非腕部执行裁剪和旋转，腕部跳过几何变换，所有相机按样本独立采样有效颜色扰动；当前本地 augmax 的 saturation 无效分支保持不生效。现有 `ImageTransforms` 调用路径缺少相机键，因此仅在内部读取器向 LeRobot 的 `ImageTransforms` 传入键，任意其他 callable 仍以单个图像参数调用。训练/验证数据集现有工厂已分别挂载变换和 `None`；模型保留确定性的 resize 与归一化，删除随机增强分支。这样不改数据集公开构造参数或策略方法签名，也不增加 JAX/augmax 必装依赖。增强移到 DataLoader CPU worker 会增加读取端计算与浮点图像传输开销，需用真实数据检查吞吐；由于插值及随机数阶段不同，不保证与 JAX 逐像素一致。
3. **对齐补齐动作维度的 loss。** 仅增加 `PI05Config` 布尔字段 `train_on_padded_action_dims`，默认 `false`，折布对齐脚本通过现有配置项开启。开启时保留模型 32 维逐元素 loss，并对全部维度取均值；关闭时沿用按实际动作维截断的既有行为。这样在 12 维 SO-101 数据上复现 JAX 对 32 维 padded action 的监督，也让训练 loss 与验证 loss 使用同一维度定义。备选是只对 12 个真实维度归一化 JAX loss；该方案会调整 JAX 参考目标，故不采用。
4. **v3 数据和验证划分使用原生比例。** 脚本默认继续使用 LeRobot v3 数据；必须先使用与训练相同的原生 `eval_split` 划分，再从训练 episode 生成状态/动作 q01/q99，并确保训练预处理器使用这些统计。可复用仓库已有的 `RunningQuantileStats` 计算逻辑，但现有 `augment_dataset_quantile_stats.py` 会调用 `push_to_hub()`，禁止直接以其默认入口运行；首次训练前单独运行准备脚本，在本地直接合并到 v3 数据集的 `meta/stats.json`，只补充 q01/q99；折布训练入口仅读取已有统计，不在每次启动时重算；不得删除或覆盖已有 mean/std、min/max、count 等统计字段，确保 MEAN_STD 和 QUANTILES 两种归一化仍可用。默认 `eval_split=3/96`，在当前单任务、96 个 episode 数据集上由 LeRobot 留出末尾 `[93, 94, 95]`；不再人工重排 episode。此验证集与 openpi 的 `[57, 21, 75]` 不同，不能按相同样本直接比较两边的验证 loss。无须在训练前比较 v2.1。openpi 验证 step 和 LeRobot 离线验证 loss 都使用实时参数；EMA checkpoint/策略评估必须单独标识。

## Risks / Trade-offs

- [PyTorch affine grid 与 augmax 的几何阶段重采样、填充边界实现不同] → 对齐有效行为、概率和参数范围；对固定图像做输出检查，并说明无法保证逐像素一致。
- [增强迁移到 DataLoader CPU worker 增加计算和浮点图像传输] → 训练前检查样本形状、数值范围与实际数据读取吞吐；验证集继续读取原始图像。
- [EMA 占用额外显存和存储] → 记录开销，保存及评估时标明实时/EMA 权重身份。
- [v3 数据缺少 q01/q99] → 先基于训练 episode 生成本地分位数，再用数据集原生 `load_stats`/`write_stats` 合并写入 `meta/stats.json`；验证原有 mean/std 未变且仍可用于 MEAN_STD。训练配置继续引用稳定的原数据集路径，便于 checkpoint 恢复。禁止调用会上传数据集的现有统计脚本默认入口。
- [padding 维度的噪声目标不是机器人实际动作] → 仅在明确要求复现当前 JAX 参考目标的折布配方启用；提供配置开关，其他任务保留真实维度 loss。
- [loss 包含 padding 后的标量尺度和梯度发生变化] → 折布 recipe 显式启用并记录该选项；其他任务使用默认真实维度 loss。

## Migration Plan

只更新此折布训练入口的配方默认值；已有 checkpoint、其他任务和通用图像变换的默认行为不迁移。用户可用 `OPENPI_AUGMENTATION=false` 恢复折布脚本原有的无增强图像路径。

## Open Questions

无。使用 LeRobot v3 数据，默认每卡 batch 6、两卡全局 batch 12，checkpoint 保存间隔为 20,000 步。
