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
| EMA | 默认关闭；不传 `--ema.*`，采用上游 `ema=None` |
| 状态/动作归一化 | `QUANTILES`；图像 `IDENTITY` |
| Image augmentation | `OPENPI_AUGMENTATION=true` 默认开启；设为 `false` 关闭；具体强度由下列 `OPENPI_AUG_*` 参数控制 |
| Action loss | `train_on_padded_action_dims=true`，对 32 维求均值 |

v3 数据原有 `mean/std/min/max/count` 不变。准备脚本调用与训练相同的原生划分，排除验证 episode，对剩余 episode 的 `observation.state` 与 `action` 计算 q01/q99，再只把这两个字段合并写入原 `meta/stats.json`。因此 `MEAN_STD` 所需的 mean/std 与 `QUANTILES` 所需的 q01/q99 并存，验证 episode 不参与 quantile 计算。该划分与 openpi 的 `[57, 21, 75]` 不同，两个实现的验证 loss 不能按同一验证集直接比较。

图像先缩放补边至 224×224，再逐图增强。顶视等非 wrist 相机随机裁取 95% 宽高并缩放回原尺寸，然后旋转 ±5°；随机裁剪的位置也带来小幅平移，没有单独的 Translate 操作。所有相机以 0.5 概率同时应用 augmax 有效的 brightness `0.3`、contrast `0.4`、hue `0.1` 范围扰动。当前本地 augmax 的 saturation 分支丢弃了变换返回值，因此不会额外改变饱和度。增强通过上游批处理变换框架的 `PerSampleTransform` 适配，在 `dataset.image_transforms.backend=dataloader` 的读取阶段执行；uint8 输入转为归一化 float32，历史帧各自采样增强参数，且相机名称随每次调用传递；离线验证数据集不挂载增强，动作推理也不增强。LeRobot 将裁剪、缩放和旋转合成一个 affine grid 采样；augmax 按阶段重采样，因此插值与边界结果可能不同，随机增强不保证逐像素一致。

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

配方默认关闭 EMA，周期离线验证每 500 步计算实时参数的 loss。上游 EMA 仍可通过 `--ema.type=constant --ema.decay=0.999` 显式启用；启用后的评估与导出使用上游 EMA 语义，不能继续使用旧的 `ema.enable` 或 `ema.use_for_eval` 参数。此配方的默认验证结果仅代表未启用 EMA 的路径。

训练指标默认写入本机 Trackio 项目 `lehome-pi05`，用 `uv run trackio show --project lehome-pi05` 查看。脚本不指定 `space_id`，因此不创建远端 Space；如需另行配置托管记录，可传上游 `--trackio.space_id=用户或组织/Space`。`--policy.push_to_hub=false` 仍控制模型上传。

两卡全参 FSDP 示例：

```bash
USE_LORA=false FSDP_DEVICES=2 NUM_GPUS=2 BATCH_SIZE=6 \
./examples/pi05/train_so101_bimanual_folding_cloth.sh
```

脚本显式设置 `--accelerator.fsdp.wrap_modules='["SiglipEncoderLayer"]'`，其余 VLM 与 action-expert 参数由根 FSDP 单元管理，沿用上游模型结构，不插入本地联合 decoder 层。此脚本的 FSDP 检查限定已验证的全参配方，不代表上游禁止 PEFT 或 EMA。默认两卡 global batch 仍为 12；初始化仍会构造完整模型，激活显存也不会按卡数直接缩小。新方案的吞吐和显存须使用当前代码及相同 batch 重新测量，历史联合层结果不能作为当前方案的性能证据。

旧版本曾完成 GPU0 上 batch 1、训练 1 步和验证 2 个样本的冒烟，记录 `eval_loss=0.1286`；当时 checkpoint 导出因 `/tmp` 空间不足而失败，临时输出已清理。该结果发生在增强管线和上游 FSDP 迁移前，不验证当前训练质量、吞吐或 checkpoint 大小。当前 checkpoint 空间需求随保存格式、优化器与 EMA 设置变化，应以实际导出测量；默认 `SAVE_FREQ=20000` 在 20k 和最终 40k 步保存，`SAVE_FREQ=0` 只保留最终 checkpoint。

如需关闭增强：

```bash
OPENPI_AUGMENTATION=false ./examples/pi05/train_so101_bimanual_folding_cloth.sh
```
