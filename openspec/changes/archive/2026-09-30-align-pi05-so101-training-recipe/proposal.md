## Why

当前 LeRobot 双臂 SO-101 折布 π0.5 配方与本地 openpi JAX 配方在学习率、归一化、EMA、验证划分及图像增强上不一致。先对齐这些训练设置，才能更清楚地判断剩余效果差异。

## What Changes

- 折布训练入口使用 LeRobot v3 版 `so101_bimanual_folding-cloth-720p` 数据集；该版本与 openpi 使用的 v2.1 版本是同一批数据的格式转换，不再增加两版数据的等价性比较。
- 对齐学习率计划、状态/动作 quantile 归一化、EMA、验证集数量（使用 LeRobot 原生 `eval_split`，不指定相同 episode），以及 JAX 对补齐到 32 维的动作执行完整维度 MSE 的训练目标。v3 元数据尚无 q01/q99，实施时须从 v3 训练集计算 q01/q99，直接追加到本地数据集的 `meta/stats.json`，并完整保留已有 mean/std 等统计量。
- 通过 LeRobot 通用的 `dataset.image_transforms` 管线提供可配置的 openpi 风格图像增强。训练图像先缩放补边至 224×224；非腕部图像再随机裁剪 95%、缩回原尺寸、旋转 ±5°，全部图像按本地 augmax 的实际颜色行为增强。通用默认关闭，折布配方明确启用；用户可关闭或分别调整裁剪、旋转及有效颜色扰动的幅度和概率；验证与推理始终不增强。

## Capabilities

### New Capabilities

- `pi05-so101-training-recipe`：基于 LeRobot v3 数据的可复现折布训练与验证配方。
- `pi05-openpi-image-augmentation`：在通用数据图像变换管线中，由开关控制、按相机类型区分的折布训练图像增强。
- `pi05-flow-matching-loss-alignment`：折布配方可选地复现 JAX 对补齐动作维度的 loss 计算。

### Modified Capabilities

无。

## Impact

通过折布脚本的 dataset 图像变换配置和 π0.5 的 loss 配置控制训练配方；训练数据读取时执行增强，模型不执行随机增强。保留数据集公开 API、通用训练器和策略方法签名。

## Non-goals

本变更不修改 Gemma/SigLIP 或 attention；loss 对齐通过折布配方的显式开关启用，不改变其他 π0.5 训练任务的默认行为。
