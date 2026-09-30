## Purpose

通过 LeRobot 通用数据图像变换管线提供可配置的 OpenPI 风格训练增强，并按相机类型区分几何行为。

## Requirements

### Requirement: π0.5 图像增强必须复现当前 openpi 的有效训练配方
启用 openpi 增强时，LeRobot 训练数据的通用 `dataset.image_transforms` 管线 MUST 先将输入图像缩放补边到 224×224，再按图像独立采样变换；π0.5 模型内部 MUST 不再执行随机增强。非腕部相机在默认配置下 MUST 在宽高各裁剪到 `int(224×0.95)` 后缩回 224×224，并随机旋转 -5° 到 +5°；腕部相机 MUST 跳过裁剪与旋转。所有相机在默认配置下 MUST 按本地 augmax 的有效颜色行为实现：`brightness=0.3`、`contrast=0.4`、默认 `hue=0.1`、整组 ColorJitter 应用概率 0.5；本地 augmax 的 `saturation=0.5` 分支实际不改变输出，故对齐模式禁止额外引入饱和度扰动。该增强与 `pi05-so101-training-recipe` 的折布配方关联，但一般 π0.5 默认不启用。

#### Scenario: 顶视相机训练图像
- **WHEN** 对齐增强开启且训练 batch 中有顶视图像
- **THEN** 每张顶视图在缩放补边后独立经历 95% 随机裁剪、恢复尺寸、±5° 旋转，以及按 0.5 概率采样的颜色变换

#### Scenario: 腕部相机训练图像
- **WHEN** 对齐增强开启且训练 batch 中有腕部图像
- **THEN** 腕部图像不裁剪也不旋转，只进入同一颜色变换路径

#### Scenario: 相同 batch 的不同图像
- **WHEN** 一个 batch 含多张图像
- **THEN** 随机裁剪位置、旋转角和颜色参数按样本抽取，不复用单个 batch 共享的参数或统计量

### Requirement: 增强强度必须通过数据图像变换配置调整
`dataset.image_transforms.tfs` MUST 将 `OpenPIImageAugmentation` 注册为自定义变换，并通过该变换的 `kwargs` 将非腕部裁剪比例、旋转幅度，以及所有相机的颜色应用概率、亮度、对比度、色相幅度分别作为配置项，默认保持上述 openpi 配方。折布训练脚本 MUST 提供对应参数供用户逐项覆盖，不修改通用训练器、数据集或公开策略方法签名。

#### Scenario: 覆盖几何增强
- **WHEN** 用户将裁剪比例设为 1.0、旋转幅度设为 0
- **THEN** 非腕部图像不再随机裁剪或旋转，颜色设置保持独立

#### Scenario: 覆盖颜色增强
- **WHEN** 用户将颜色应用概率设为 0
- **THEN** 所有相机均不应用亮度、对比度或色相扰动，几何设置保持独立

### Requirement: 增强必须只作用于训练路径
`dataset.image_transforms.enable` MUST 默认关闭；折布训练脚本 MUST 暴露可显式设置为 `true` 或 `false` 的开关，默认设为 `true` 以对齐 openpi。只有训练数据集读取时才执行增强；验证数据集 MUST 不挂载该变换，动作推理和保存后加载的推理策略 MUST 使用无增强的图像路径。其他通用图像变换和任意已有 callable 的行为 MUST 保持不变。

#### Scenario: 默认 π0.5
- **WHEN** 用户未启用 openpi 增强
- **THEN** π0.5 的训练、验证和推理图像预处理与当前默认行为一致

#### Scenario: 折布脚本关闭增强
- **WHEN** 用户以 `OPENPI_AUGMENTATION=false` 启动折布训练脚本
- **THEN** 训练使用无增强的原始图像路径，其余配方设置保持不变

#### Scenario: 验证与推理
- **WHEN** 增强配置开启且训练器创建训练和验证数据集
- **THEN** 仅训练数据读取器按相机键执行增强；验证数据集与动作推理使用无随机增强的图像路径
