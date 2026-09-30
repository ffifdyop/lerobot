## 1. 确认折布基线

- [x] 1.1 记录当前折布脚本的学习率、seed、归一化、EMA 和验证划分生效值；确认默认数据路径指向 LeRobot v3 版本。归档时按用户最终确认采用每卡 batch 6、两卡全局 batch 12。
- [x] 1.2 确认 seed 42 对应验证 episode `[57, 21, 75]`，并记录 v3 元数据缺少 q01/q99；无需比较 v2.1 数据内容。

## 2. 提供 π0.5 专用增强（可与第 3 组并行）

- [x] 2.1 通过通用 `dataset.image_transforms` 注册默认关闭的 openpi 增强；训练数据读取时缩放补边后按相机类型执行几何变换，并按本地 augmax 有效颜色语义逐图采样。
- [x] 2.2 测试开关开/关、顶视/腕部、逐图随机性、训练/验证数据集隔离及模型推理无增强；用固定图像检查裁剪放大方向、参数范围和输出尺寸，并记录插值差异。
- [x] 2.3 将裁剪比例、旋转幅度、颜色概率、亮度、对比度和色相幅度暴露为 `dataset.image_transforms.tfs.openpi.kwargs` 参数；保持 openpi 默认值，测试覆盖几何和颜色参数的生效。

## 3. 对齐折布训练入口

- [x] 3.1 更新折布脚本的 seed、学习率计划、40,000 步和优化器显式参数；默认每卡 batch 6、两卡全局 batch 12，checkpoint 保存间隔 20,000 步。
- [x] 3.2 首次训练前独立运行准备脚本，从排除验证 episode 的 v3 训练集生成状态/动作 q01/q99，直接合并写入本地数据集 `meta/stats.json`；训练入口仅读取现有统计，训练和 checkpoint 均引用稳定的原数据集路径；合并时保留原有 mean/std 等字段且不调用会推送 Hub 的统计脚本默认入口。验证 `MEAN_STD` 与 `QUANTILES` 两种模式均可用；再启用 quantile 归一化和常数 `0.999` EMA，并使用原生 `dataset.eval_split` 和 500 步离线验证。
- [x] 3.3 通过现有 draccus dataset 配置项控制增强；折布脚本的 `OPENPI_AUGMENTATION` 开关默认开启并映射到 `dataset.image_transforms.enable`，可设 `false` 关闭；六个 `OPENPI_AUG_*` 环境变量可逐项覆盖增强强度。

- [x] 3.4 仅在 `PI05Config` 增加默认关闭的 `train_on_padded_action_dims` 配置项，不改 `forward` 方法签名或通用训练器；折布脚本通过现有 policy 配置启用，loss 和验证 loss 对 32 维动作目标求均值。

## 4. 验证与记录

- [x] 4.1 测试 12 维真实动作补齐到 32 维时，开关开启会将 20 个 padding 维纳入训练和验证 loss/梯度；关闭时保持原有 12 维 loss。
- [x] 4.2 运行受影响的 π0.5、图像增强及训练配置测试，并执行 `openspec validate align-pi05-so101-training-recipe --strict`。
- [x] 4.3 在迁移增强到数据管线前，于 GPU0 完成 1 步训练和 2 样本离线验证（loss=0.1286，仅作功能冒烟）；配置及日志确认增强/loss 开关、验证无增强和实时权重评估，EMA 以 0.999 初始化更新。checkpoint 导出因临时文件系统空间不足失败，未将冒烟 loss 用于策略质量判断。
- [x] 4.4 记录 v3 数据路径、归一化统计、增强/loss 开关、实时/EMA 权重身份及评估指标；loss 比较注明是否包含 padded action 维度。

## 5. 将图像增强迁移到通用数据变换管线

- [x] 5.1 在 `dataset.image_transforms` 中实现可配置的 openpi 增强，使数据读取阶段按相机键选择几何行为，并在增强前缩放补边至 224×224；保留现有通用变换及自定义 callable 的行为。
- [x] 5.2 从 `PI05Config` 和模型 `_preprocess_images` 移除增强分支，将折布脚本的开关和强度参数映射到 dataset 配置；训练集增强，验证集和推理不增强。
- [x] 5.3 更新针对数据变换、相机规则、逐图随机性及训练/验证隔离的测试与配方文档；135 项相关测试通过，并用本地 v3 数据完成三相机单样本读取；执行脚本语法检查和 OpenSpec 严格校验。

## 6. 收敛到 LeRobot 已有组件

- [x] 6.1 将 OpenPI 增强注册为通用 `ImageTransformsConfig.tfs` 变换，通过现有总开关及变换参数配置，保留相机规则和其他变换行为。
- [x] 6.2 复用已有缩放补边函数，删除增强中的重复实现，并保持各策略原导入路径可用。
- [x] 6.3 使用数据集原生 `load_stats`/`write_stats` 合并 q01/q99，保留 mean/std 等原统计。
- [x] 6.4 移除随机 holdout 分支，收敛折布验证配置，更新文档和 OpenSpec，运行相关测试与严格校验。

## 7. 改用原生验证划分

- [x] 7.1 折布训练入口仅设置 `dataset.eval_split`，删除固定 episode 列表与重排逻辑。
- [x] 7.2 分位数准备复用 `make_train_eval_datasets` 的训练 episode，确保验证数据不进入 q01/q99。
- [x] 7.3 更新提案、规格和配方文档，验证 96 个 episode 默认留出 3 个并运行相关检查。

## 归档验证（2026-09-30）

当前代码在 CPU 上通过 160 项相关测试；折布脚本语法检查、差异空白检查及全部主规格的 OpenSpec 严格校验通过。三个增量规格已同步到主规格。checkpoint 导出验证按用户要求不作为归档阻塞条件。

归档后用户确认将每卡 batch=6 和 checkpoint 保存间隔=20,000 步一并纳入原提交；训练脚本、配方文档和主规格已同步。
