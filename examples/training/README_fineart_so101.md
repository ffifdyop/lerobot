# SO101 双卡 FSDP2 / DDP + W&B FineART-VLA 微调

本方案默认使用 `/data/DataCaption/Flora/outputs/so101_bimanual_insert_2/subtask/dataset` 的全部
22 个 episode、17,281 帧，包括原来标记为排除的区间。前置检查依据实际
`language_persistent` 的有效 subtask，忽略旧的 `training_ready` 和区间筛选标记。
不会修改源数据、重新裁剪视频或接入 reward 权重。

## 方案与参数

先检查数据与配置，再显式跑 20 步 smoke，最后执行正式训练。
当前脚本保留你设置的每卡 batch 12、4000 steps、学习率 4e-5 → 4e-6，以及关闭
gradient checkpointing 的设置。全量数据每轮 721 steps，4000 steps 约为 5.55 个 epoch；
恰好 5 轮为 3605 steps。要按 `EPOCHS` 自动计算，将脚本 `STEPS` 的默认值改为空。
这是训练预算，不是已验证的收敛步数或成功率。

| 参数 | smoke | 初始微调 |
| --- | --- | --- |
| GPU | 两张 96 GB 卡，默认 GPU 0、2 | 同左；默认 FSDP2，可切换 DDP |
| batch size | 每卡 1，全局 2 | 每卡 12，全局 24；可按显存调整 |
| steps | 显式 `--steps 20` | 当前显式 4000；留空后按 EPOCHS 自动计算 |
| learning rate | 4e-5 → 4e-6，warmup 2 步 | 4e-5 → 4e-6，warmup 100 步 |
| 精度与省显存 | BF16、FSDP2；gradient checkpointing 可开启 | 同左，当前 gradient checkpointing 关闭 |
| 训练参数 | 默认 VLM、vision、LM head、action expert 全参数微调 | 可切换 LoRA、冻结 VLM 或只冻结 vision |
| 监督 | 30% task→subtask，70% subtask→action；text/FAST/flow 权重 1/1/10 | 同左 |
| action chunk | 50 帧，30 FPS，即约 1.67 秒 | 同左 |
| 数据使用 | 默认全区间、无 episode 留出 | `TRAIN_RATIO` 可留出完整 episode 验证；始终不做区间过滤 |

保持原 checkpoint 的 `fast_skip_tokens`（本地版本为 128），不修改 embedding 权重共享。
12 维 SO101 特征和三路相机键从数据集重新生成，不沿用 ALOHA 的 14 维输入配置。
相机语义顺序保持为 top、left_wrist、right_wrist，对应预训练模型的 overhead、左腕、右腕。
训练用当前数据集的归一化统计，并重新构建 processor；FAST 从 checkpoint 自带的
`action_tokenizer/` 出发，仅在训练 episode 上拟合 1,024 个 action chunk，并验证 256 个 chunk
的重建误差。父进程先完成一次拟合，再启动各 worker 只读加载，避免多卡并发写入
FAST 缓存。相同训练 episode 和配置的 smoke 与正式训练共享缓存；改变划分会使用不同缓存。

## 自己设置训练参数

集中修改 `/data/lerobot-fineart/scripts/train_fineart_so101.sh` 中的“常用训练参数”区。
例如将下面几项的默认值改为你需要的值：

```bash
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-fsdp}" # fsdp 或 ddp
DP_SHARD="${DP_SHARD:-}"          # 每组卡数；空值：fsdp=2，ddp=1
DP_REPLICATE="${DP_REPLICATE:-}"  # 组数；空值：fsdp=1，ddp=2 个副本
FSDP_RESHARD_AFTER_FORWARD="${FSDP_RESHARD_AFTER_FORWARD:-true}"
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-false}"
BATCH_SIZE="${BATCH_SIZE:-12}"      # 每卡 batch，双卡全局为 24
MAX_LR="${MAX_LR:-${LR:-4e-5}}"    # 峰值学习率，兼容旧 LR
FINAL_LR="${FINAL_LR:-${MIN_LR:-4e-6}}" # 最终学习率，兼容旧 MIN_LR
EPOCHS="${EPOCHS:-5}"             # 自动计算训练步数
STEPS="${STEPS:-4000}"           # 非空时优先于 EPOCHS；自动计算时改为空默认值
TRAIN_RATIO="${TRAIN_RATIO:-1.0}" # 1.0 全量；0.8 留出约 20% episode 验证
NUM_WORKERS="${NUM_WORKERS:-4}"   # 每卡 workers
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
USE_LORA="${USE_LORA:-false}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
FREEZE_VLM="${FREEZE_VLM:-false}" # true 自动改为动作专家训练，关闭 text/FAST
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
SAVE_FREQ="${SAVE_FREQ:-1000}"    # 0 只保存最终 checkpoint
EVAL_FREQ="${EVAL_FREQ:-1000}"    # 0 关闭验证 loss；全量训练时自动关闭
LOG_FREQ="${LOG_FREQ:-20}"
```

也可不改文件，在启动命令前设置环境变量。命令行参数优先于环境变量，环境变量优先于脚本默认值：

```bash
# 先检查目标配置，再把 check 改为 train。
GPU_IDS=0,2 BATCH_SIZE=4 MAX_LR=2e-5 FINAL_LR=2e-6 TRAIN_RATIO=0.8 STEPS=8095 \
  bash /data/lerobot-fineart/scripts/train_fineart_so101.sh check

# 同样可以使用命令行参数；--lr/--learning-rate 是 --max-lr 的兼容别名。
bash /data/lerobot-fineart/scripts/train_fineart_so101.sh check \
  --batch-size 4 --max-lr 2e-5 --final-lr 2e-6 --train-ratio 0.8 --steps 8095 \
  --warmup-steps 100 --save-freq 1000 --eval-freq 1000
```

`TRAIN_RATIO` 表示按 episode 划分训练/验证集，余下全部用于验证，不是随机抽取若干帧。
复用原生规则：每个任务末尾留出 `ceil(该任务 episode 数 × (1 - TRAIN_RATIO))` 个 episode。
训练集和验证集的 episode 不重叠；episode 长短和取整会使实际帧比例偏离请求值。
检查报告会列出两组 episode、帧数及实际比例。`SEED` 控制训练随机性，不改变上述固定划分规则。
比例太小时，如果所有 episode 都被留出，检查直接报错。

保留原生归一化行为：使用源数据集 `meta/stats.json`，该统计包含全部源 episode。
FAST 的拟合样本只来自训练 episode；其重建检查也来自训练集，不作为留出集性能。
`MAX_EVAL_SAMPLES=0` 验证全部留出帧，正数则使用原生的按任务抽样上限。

总卡数为 `DP_SHARD × DP_REPLICATE`，
步数计算为 `steps_per_epoch = ceil(训练帧数 / (每卡 batch × 总卡数))`，
`steps = ceil(EPOCHS × steps_per_epoch)`，包含多卡尾 batch 补齐。
显式设置 `STEPS` 或 `--steps` 后，以该步数为准，报告会给出对应 epoch 数。
默认梯度累积为 1；50 帧 action chunk 不会把训练样本数再除以 50。
学习率的衰减周期自动对齐最终 steps；warmup 不超过总步数减 1。
smoke 示例显式传 `--steps 20`，防止脚本中设置的 `STEPS=4000` 覆盖短程预算。
其默认每卡 batch 1、workers 0、warmup 2，可用 `SMOKE_BATCH_SIZE`、
`SMOKE_NUM_WORKERS`、`SMOKE_WARMUP_STEPS` 或对应命令行参数覆盖。

## FSDP 每组卡数与组数

原生 `parallelism.dp_shard` 是每个分片组的卡数，`parallelism.dp_replicate` 是组数。
每组内分片参数、梯度和优化器状态；组间复制相同的分片布局并同步梯度。
脚本暴露 `DP_SHARD`、`DP_REPLICATE`，命令行对应 `--dp-shard`、`--dp-replicate`。
两者的乘积必须等于 `GPU_IDS` 中设备数，父进程按该乘积启动 worker。

| 总卡数 | 每组卡数 `DP_SHARD` | 组数 `DP_REPLICATE` | 模式 |
| --- | --- | --- | --- |
| 2 | 2 | 1 | FSDP2，默认 |
| 4 | 4 | 1 | FSDP2 |
| 4 | 2 | 2 | HSDP（使用 FSDP2） |
| 8 | 4 | 2 | HSDP（使用 FSDP2） |
| 2 | 1 | 2 | DDP；设置 `DISTRIBUTED_BACKEND=ddp` |

两个值留空时，`fsdp` 默认一组两卡，`ddp` 默认两个单卡副本。
原生支持多机拓扑；这个示例的父进程只启动单机 worker，`GPU_IDS` 必须列出同一台机器的卡。

```bash
# 一组两卡；不会加载模型或启动训练。
GPU_IDS=0,2 DP_SHARD=2 DP_REPLICATE=1 bash scripts/train_fineart_so101.sh check
# 四卡示例：两组，每组两卡（HSDP）。
GPU_IDS=0,1,2,3 DP_SHARD=2 DP_REPLICATE=2 bash scripts/train_fineart_so101.sh check
# 四卡也可组成一个分片组；CLI 参数可覆盖环境变量。
GPU_IDS=0,1,2,3 bash scripts/train_fineart_so101.sh check --dp-shard 4 --dp-replicate 1
# 真实训练 smoke：检查后手动执行。
DISTRIBUTED_BACKEND=fsdp bash scripts/train_fineart_so101.sh smoke --steps 20
# 可切回 DDP；也支持 --distributed-backend ddp。
DISTRIBUTED_BACKEND=ddp bash scripts/train_fineart_so101.sh check
```

分组按 `GPU_IDS` 的顺序映射到逻辑 rank。比如 `0,1,2,3` 配合每组两卡时，
分片组是 `[0,1]` 和 `[2,3]`。仅有两卡时，两组各两卡需要四卡，配置检查会直接报错。
全局 batch 为 `每卡 batch × DP_SHARD × DP_REPLICATE`：HSDP 的所有卡都会处理不同的数据，
不会因为只有两组就仅乘以二。切换拓扑后，未显式指定 STEPS 时会重新计算 epoch 预算；
当前脚本显式 `STEPS=4000`，该值仍然优先。

FSDP2 子单元仅选择 `SiglipEncoderLayer`。FineART 联合注意力直接访问 Gemma decoder
内部参数，因此 decoder、action expert 和 LM head 保留在根 FSDP 单元中；不要改成
按 `GemmaDecoderLayer` 独立分片。根单元参数在计算时仍需完整 gather，实际显存峰值
不会简单减半，batch 12 的可行性需要实际 smoke 检查。

`FSDP_RESHARD_AFTER_FORWARD=true` 在 forward 后重新分片参数；false 会增加参数驻留，
减少 backward 前的 gather。`FSDP_CPU_OFFLOAD=true` 开启原生 CPU offload，通常降低速度。
二者支持 `--fsdp-reshard-after-forward false` 和 `--fsdp-cpu-offload true`。
gradient checkpointing、LoRA 和冻结控制继续使用现有参数；CPU 配置检查不证明这些组合
已通过真实双卡 forward/backward。不要另设 `ACCELERATE_USE_FSDP`，由原生配置选择后端。

保留原生 `checkpoint_format=safetensors`：保存时汇总完整模型，分片优化器状态使用 DCP，
保存也可能出现显存峰值。示例续训使用 checkpoint 中保存的同一份配置和相同拓扑。

## 冻结、LoRA 与学习率调度

| 参数 | 作用 |
| --- | --- |
| `GRADIENT_CHECKPOINTING=true/false` | 对应原生 `policy.gradient_checkpointing`，控制用重计算节省训练显存 |
| `USE_LORA=true/false` | 使用原生 PEFT LoRA；`LORA_R` 为 rank，`LORA_ALPHA / LORA_R` 为缩放 |
| `FREEZE_VLM=true/false` | 冻结整个 PaliGemma，包括视觉、语言模型及 LM head，只训练动作专家/投影或其 LoRA |
| `FREEZE_VISION_ENCODER=true/false` | 全参模式只冻结 SigLIP 视觉编码器，保留语言模型、LM head 和多模态投影可训练 |
| `WARMUP_STEPS` | warmup 步数；`0` 关闭 warmup |
| `MAX_LR` | 调度器的 `peak_lr`，同时作为基础 optimizer LR |
| `FINAL_LR` | 调度器的 `decay_lr`，训练结束时的最终 LR；留空为 `MAX_LR / 10` |

全参训练示例：

```bash
GRADIENT_CHECKPOINTING=true USE_LORA=false FREEZE_VLM=false \
MAX_LR=1e-5 FINAL_LR=1e-6 WARMUP_STEPS=100 \
  bash /data/lerobot-fineart/scripts/train_fineart_so101.sh check
```

LoRA 示例（检查后将 `check` 改为 `train`）：

```bash
USE_LORA=true LORA_R=16 LORA_ALPHA=32 FREEZE_VLM=false FREEZE_VISION_ENCODER=true \
GRADIENT_CHECKPOINTING=true MAX_LR=2e-5 FINAL_LR=2e-6 WARMUP_STEPS=100 \
  bash /data/lerobot-fineart/scripts/train_fineart_so101.sh check
```

这些控制同样支持 `--gradient-checkpointing false`、`--use-lora true`、`--lora-r 16`、
`--lora-alpha 32`、`--freeze-vlm true`、`--freeze-vision-encoder true` 等命令行参数。

`FREEZE_VLM=true` 对应原生 `train_expert_only=true`。FineART 原生不允许冻结 VLM
同时使用 text/FAST 监督，因此脚本自动将 text/FAST 权重置零、关闭 FAST 拟合，
将 recipe 改为 100% `low_level_execution`，仍以标注 subtask 作为动作条件。
该模式不训练 task→subtask 生成。`FREEZE_VLM` 同时意味着视觉编码器冻结。

LoRA 模式由原生 PEFT 冻结基础权重，只训练 adapter，默认不全量训练 LM head。
`FREEZE_VLM=false` 时，adapter 覆盖 VLM 语言层的 attention/MLP、动作专家 q/v 和
action/time 投影，text/FAST 可更新 VLM adapter；`FREEZE_VLM=true` 时只给动作侧注入。
目标层不含 vision tower、多模态投影或直接使用权重的 LM head。
因此 LoRA 下 `FREEZE_VLM=false` 表示允许 VLM adapter 更新，基础 VLM 仍然冻结。
检查只验证配置；实际 adapter 注入、权重加载和 GPU 训练由 smoke 验证。

调度保持原生 `cosine_decay_with_warmup`：warmup 从 `MAX_LR / (WARMUP_STEPS + 1)`
附近起步，向峰值方向上升，随后余弦衰减，到总 steps 时达到 `FINAL_LR`。
原生余弦分支使用全局 step，因此 warmup 边界不保证恰好等于 `MAX_LR`；
`MAX_LR` 是调度器的峰值参数，不是第一个 step 的学习率。旧环境变量 `LR`/`MIN_LR`
和命令行 `--lr`/`--min-lr` 仍可使用，新变量非空时优先。

默认基线关闭编译和额外 attention 优化，`flow_num_repeats=1`、knowledge insulation 开启。
默认 text/FAST 仍训练 VLM；knowledge insulation 仅隔离 flow 的梯度，不等于冻结整个 VLM。
启用 W&B 指标和配置日志，未启用模型权重
artifact 上传、Hub 上传、Trackio 和硬件评估。

## 执行

首次补齐当前仓库环境（安装依赖，不启动训练；使用已有 lockfile）：

```bash
cd /data/lerobot-fineart
bash scripts/train_fineart_so101.sh setup
# 已在本机登录 W&B 时可跳过这一行。
uv run --no-sync wandb login
```

`setup` 安装 `training`、`fineart_vla`、`fineart_vla_kernels`、`wandb`、`peft` extras。
登录通过 W&B 自己的交互式命令完成；不要把 API key 写入脚本或训练配置。

默认使用本地已缓存的 checkpoint 和 tokenizer，关闭 Hub 联网。
如果缺少缓存，补下载时可设置 `HF_HUB_OFFLINE=0`；也可用
`--checkpoint /absolute/path/to/pretrained_model` 指定完整本地 checkpoint。
不要关闭数据集离线加载，也不要删除源视频目录；数据集视频通过软链接复用源文件。

先运行检查，不加载模型、不拟合 FAST、不训练：

```bash
bash scripts/train_fineart_so101.sh check
```

检查会遍历每帧 subtask，读取所有 episode 首尾的三路图像，并写出
`train_config.json` 和 `preflight.json`。后者记录全量数据策略、数据文件 SHA256、
checkpoint 来源、代码版本、标注覆盖和解码结果。

然后分别启动 smoke 和初始微调：

```bash
GPU_IDS=0,2 WANDB_PROJECT=fineart-so101 bash scripts/train_fineart_so101.sh smoke --steps 20
GPU_IDS=0,2 WANDB_PROJECT=fineart-so101 bash scripts/train_fineart_so101.sh train
```

smoke 会真实加载权重、拟合 FAST、计算 loss、反向传播并保存最终 checkpoint。
正式运行从原始 midtrained checkpoint 开始，smoke 用来验证训练链路。
两种模式默认生成精确到秒的 UTC 时间戳目录，例如 `train_20261105T054916Z`，
旧训练目录存在时拒绝覆盖。

自定义示例：

```bash
GPU_IDS=0,2 WANDB_PROJECT=fineart-so101 bash scripts/train_fineart_so101.sh train \
  --distributed-backend fsdp --steps 3605 --batch-size 12 \
  --max-lr 4e-5 --final-lr 4e-6 --train-ratio 1.0 \
  --run-dir /data/lerobot-fineart/outputs/fineart_so101/full_5epochs_fsdp_b12
```

`GPU_IDS` 必须列出 `DP_SHARD × DP_REPLICATE` 个不同的设备；默认 `0,2` 是本机两张 96 GB GPU。
脚本自身是父进程，会按拓扑设置 `torch.distributed.run --nproc-per-node`，不要再套一层 torchrun。
`--batch-size` 始终指每卡 batch size，梯度累积为 1，当前默认全局 batch size 为 `12×2=24`。

W&B 默认配置如下，团队沿用 PI05 叠布训练脚本的 W&B 配置，项目与任务名使用本项目的命名：

| 字段 | 值 |
| --- | --- |
| entity / 团队 | `2943872339-zhejiang-university` |
| project | `fineart-so101` |
| job name | `fineart_so101_full_<check\|smoke\|train>` |
| W&B run name | `fineart-so101-<run-dir 的目录名>`，默认含模式和时间戳 |
| mode | `online` |
| artifacts | `wandb_logging.disable_artifact=true`，显式关闭模型 artifact 上传 |

参考脚本的实际路径为
`/home/gsp/Dev/Manipulation/lerobot/examples/pi05/train_so101_bimanual_folding_cloth.sh`。
该脚本当前已改用 Trackio；这里核对并沿用其 Git 提交 `6c8e56fe` 中的 W&B 团队设置，
保留本项目的原生 W&B 后端，不使用参考脚本的 `lehome-pi05` 项目或叠布任务名。
认证复用本机现有 W&B 登录或 `WANDB_API_KEY` 环境变量，不读取或复制密钥到项目文件。

可设置 `WANDB_ENTITY` 覆盖团队、`WANDB_NAME` 设置 run 名称，
或传 `--wandb-project`、`--wandb-entity`、`--wandb-name`。
没有网络时可以显式 `WANDB_MODE=offline`，日志保存在 `<run-dir>/train/wandb_logs/`，
之后用 `uv run --no-sync wandb sync <offline-run-directory>` 上传。

原生训练配置使用 `wandb_logging` 字段，避免与旧 checkpoint 的 `wandb`→Trackio 迁移冲突。
只有全局 rank 0 建立 W&B run 并记录跨 rank 汇总的训练 loss、text/FAST/flow 子 loss、
梯度范数、学习率、吞吐等原生指标；每步是否有某个子 loss 由实际 recipe 路由决定。
run ID 会保存到训练配置和 checkpoint，在线续训通过 `resume=allow` 继续同一个 run。
project、entity、run ID 和 mode 参数使用 [W&B 官方 SDK 接口](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py)。

已有兼容环境时，可显式复用解释器；不会同步该环境或更改其 editable 安装，
脚本通过 `PYTHONPATH` 使用本仓库源码：

```bash
LEROBOT_PYTHON=/data/DualDirection-RSI/.venv-lerobot/bin/python \
  bash scripts/train_fineart_so101.sh check
```

已有环境能运行 `check` 并不保证训练依赖齐全；`smoke/train` 会检查 accelerate、
liger-kernel、wandb 和所选卡的 CUDA/BF16。检查会打印原生多卡命令；正常执行优先使用
`smoke/train`，以便由父进程先完成 FAST 拟合。

## 验收与记录

smoke 需要看到有限的 loss/梯度、text/FAST/flow 对应日志以及最后的 checkpoint。
`check` 通过只证明数据、解码、recipe 和配置检查通过，不证明权重加载、GPU 训练或收敛。
脚本生成的报告始终是训练前检查报告，不会把未运行的模型检查记为成功。

模型和 processor 保存在 `<run-dir>/train/checkpoints/<step>/pretrained_model/`，
`last` 指向最新 checkpoint。保留 processor，它包含 SO101 归一化统计和拟合后的 FAST。
各 checkpoint 同时保存原生训练状态和 W&B run ID，可用相同两卡恢复同一轮训练：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
uv run --no-sync python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module lerobot.scripts.lerobot_train \
  --config_path=/absolute/run-dir/train/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

默认不留出 episode，因此训练 loss 只能用来检查训练信号。
设置 `TRAIN_RATIO<1` 后，原生训练器按 `EVAL_FREQ` 记录留出 episode 的验证 loss。
全量 17,281 帧、双卡每卡 batch 12，每轮为 721 steps，5 轮为 3605 steps；
FAST 拟合和独立 smoke 不计入这 5 轮。留出集 loss 仍不等于机器人成功率。
原排除区间的动作会被一起模仿，补齐文字不等于确认所有示范动作都适合学习。
