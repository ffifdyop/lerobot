# π0.5 SO-101 双臂折布配方

训练入口：[`train_so101_bimanual_folding_cloth.sh`](train_so101_bimanual_folding_cloth.sh)。对齐前脚本默认学习率为 `1e-4`、seed 为 1000、使用 `MEAN_STD`、未启用 EMA/图像增强，验证 episode 为 `[0, 95]`；数据路径已指向 LeRobot v3。v3 `meta/stats.json` 缺少 q01/q99。此次使用 LeRobot 原生 `dataset.eval_split=0.03125`；在当前单任务、96 个 episode 的 v3 数据集上，验证集是末尾的 `[93, 94, 95]`。seed 42 控制训练随机性，不决定验证 episode。默认数据是 LeRobot v3 `ffifdyop/so101_bimanual_folding-cloth-720p`，本机缓存位于 `/home/gsp/.cache/huggingface/lerobot/ffifdyop/so101_bimanual_folding-cloth-720p`。首次训练前单独运行统计准备脚本，将 q01/q99 合并写入本地 v3 数据集的 `meta/stats.json`；不上传 Hub。训练入口只读取已有统计，训练与 checkpoint 均使用稳定的 `DATASET_ROOT` 路径。


首次训练前从仓库根目录运行一次：

```bash
uv run --no-sync python examples/pi05/prepare_train_quantile_stats.py --repo-id ffifdyop/so101_bimanual_folding-cloth-720p --source-root /home/gsp/.cache/huggingface/lerobot/ffifdyop/so101_bimanual_folding-cloth-720p --eval-split 0.03125
```

若覆盖训练脚本的 `EVAL_SPLIT`，须用相同数值重新运行统计准备脚本，避免验证 episode 进入 quantile 统计。

配方值如下。默认每卡 batch 6、两卡、global batch 12；每 20,000 步保存 checkpoint。

| 项目 | 设置 |
| --- | --- |
| 随机种子 | 42 |
| 总训练步数 | 40,000 |
| 峰值学习率 | `4e-5` |
| Scheduler | warmup 1,000 步，40,000 步衰减到 `4e-6` |
| AdamW | betas `(0.9, 0.95)`，eps `1e-8`，weight decay `0.01`，grad clip `1.0` |
| 验证划分 | 原生 `eval_split=0.03125`，当前 96 个 episode 留出末尾 3 个；每 500 步离线验证 |
| EMA | 开启，常数 decay `0.999` |
| 状态/动作归一化 | `QUANTILES`；图像 `IDENTITY` |
| Image augmentation | `OPENPI_AUGMENTATION=true` 默认开启；设为 `false` 关闭；具体强度由下列 `OPENPI_AUG_*` 参数控制 |
| Action loss | `train_on_padded_action_dims=true`，对 32 维求均值 |

v3 数据原有 `mean/std/min/max/count` 不变。准备脚本调用与训练相同的原生划分，排除验证 episode，对剩余 episode 的 `observation.state` 与 `action` 计算 q01/q99，再只把这两个字段合并写入原 `meta/stats.json`。因此 `MEAN_STD` 所需的 mean/std 与 `QUANTILES` 所需的 q01/q99 并存，验证 episode 不参与 quantile 计算。该划分与 openpi 的 `[57, 21, 75]` 不同，两个实现的验证 loss 不能按同一验证集直接比较。

图像先缩放补边至 224×224，再逐图增强。顶视等非 wrist 相机随机裁取 95% 宽高并缩放回原尺寸，然后旋转 ±5°；随机裁剪的位置也带来小幅平移，没有单独的 Translate 操作。所有相机以 0.5 概率同时应用 augmax 有效的 brightness `0.3`、contrast `0.4`、hue `0.1` 范围扰动。当前本地 augmax 的 saturation 分支丢弃了变换返回值，因此不会额外改变饱和度。增强由训练数据集的 `dataset.image_transforms` 在读取阶段执行；离线验证数据集不挂载增强，动作推理也不增强。LeRobot 将裁剪、缩放和旋转合成一个 affine grid 采样；augmax 按阶段重采样，因此插值与边界结果可能不同，随机增强不保证逐像素一致。

增强通过通用 `ImageTransformsConfig.tfs` 注册为 `OpenPIImageAugmentation`；脚本以 `dataset.image_transforms.enable` 作总开关，并将六个环境变量写入 `dataset.image_transforms.tfs.openpi.kwargs`。其他训练入口可用相同的 `tfs` 配置。默认值对应当前 openpi 配方：

| 环境变量 / `tfs.openpi.kwargs` 字段 | 默认值 | 作用范围 |
| --- | --- | --- |
| `OPENPI_AUG_CROP_SCALE` / `crop_scale` | `0.95` | 非腕部相机裁剪后保留的宽高比例；`1.0` 关闭裁剪及其随机位移 |
| `OPENPI_AUG_ROTATION_DEGREES` / `rotation_degrees` | `5.0` | 非腕部相机随机旋转范围 `[-5°, +5°]`；`0` 关闭旋转 |
| `OPENPI_AUG_COLOR_PROBABILITY` / `color_probability` | `0.5` | 所有相机执行整组颜色扰动的概率；`0` 关闭颜色扰动 |
| `OPENPI_AUG_BRIGHTNESS` / `brightness` | `0.3` | 亮度扰动幅度 |
| `OPENPI_AUG_CONTRAST` / `contrast` | `0.4` | 对比度扰动幅度 |
| `OPENPI_AUG_HUE` / `hue` | `0.1` | 色相扰动幅度 |

例如保留增强但缩小几何和颜色扰动：

```bash
OPENPI_AUG_CROP_SCALE=0.98 OPENPI_AUG_ROTATION_DEGREES=2 \
OPENPI_AUG_BRIGHTNESS=0.15 OPENPI_AUG_CONTRAST=0.2 OPENPI_AUG_HUE=0.05 \
./examples/pi05/train_so101_bimanual_folding_cloth.sh
```

SO-101 的 12 维动作在策略内部补齐到 32 维。此配方的训练和离线验证 loss 都对 32 维逐元素误差求均值，20 个补零维度也参与监督和反向梯度；它与默认 `train_on_padded_action_dims=false` 时仅计算真实 12 维 loss 的数值和优化目标均不同。动作推理仍只返回 12 维。跨实现比较 loss 时需要注明维度口径。

LeRobot 的周期离线验证在实时参数上计算，并每 500 步写入训练日志；EMA 不用于此离线验证（`ema.use_for_eval=false`）。训练 checkpoint 中实时模型位于 `pretrained_model`，对应 EMA 模型位于 `pretrained_model_ema`。评估输出应标出所用权重身份。

迁移到数据变换管线前的一次 GPU0 短跑用 batch 1 训练 1 步、验证 2 个样本，记录到 `step 1: eval_loss=0.1286`；这是迁移前的实时参数功能冒烟指标，不代表新数据增强管线的吞吐或策略质量。EMA 已初始化并更新，但 checkpoint 导出时 `/tmp` 空间不足，运行返回错误；此次生成的 56 GB 临时输出已清理，磁盘空间已恢复。单个 checkpoint 约占 56 GiB；默认 `SAVE_FREQ=20000` 会保留 20k 和最终 40k 两份，合计约 112 GiB。若只保留最终 checkpoint，可设 `SAVE_FREQ=0`，但仍需约 56 GiB。实际训练指标保存在用户指定的输出目录/W&B 记录中。

如需关闭增强：

```bash
OPENPI_AUGMENTATION=false ./examples/pi05/train_so101_bimanual_folding_cloth.sh
```
