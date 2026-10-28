# PI05 FSDP2 验证记录

日期：2026-10-22。实现与结果均位于当前 LeRobot 工作区，未启动真实数据集的长训练。

## 环境

- Python 3.13；PyTorch 2.11.0+cu128；Accelerate 1.14.0；Transformers 5.5.4；safetensors 0.8.0。
- CUDA 验证和基准：两张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96GB/卡。
- 使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2` 固定设备。默认 CUDA 枚举顺序可能与 `nvidia-smi` 不同，必须核对设备名称。
- 默认沙箱不暴露 GPU；CUDA 和本机 Gloo 进程通信测试在获准的本机执行环境运行。

## 正确性

| 检查 | 结果 |
| --- | --- |
| PI05 专项、既有 recipe/MEM/RTC、训练配置、policy 接口、checkpoint 回归 | 92 passed，5 skipped |
| CUDA 双卡专项 | 3 passed，3 个 Gloo 用例未选中 |
| 两进程 CPU/Gloo 专项 | 3 passed，3 个 CUDA 用例未选中 |
| Ruff 0.14.1 检查与格式、脚本 `bash -n`、`git diff --check` | 通过 |
| `openspec validate add-pi05-fsdp-training --strict` | 通过 |

CUDA 三组分别覆盖 FP32 无重算 + safetensors、FP32 重算 + safetensors/DCP、BF16 重算 + safetensors/DCP。使用缩小尺寸的真实 PI05，经过生产 Accelerate.prepare 路径，验证：

- 联合 decoder、SigLIP encoder、语言 embedding 均被分片，所有参数为 DTensor。
- FP32 loss、梯度和更新与相同全局 batch 的普通模型一致；BF16 在明确容差内比较 loss。
- 完整导出能由普通 PI05 严格加载；真实 DCP 分片经离线 converter 合并后能通过普通 from_pretrained 加载。
- 保存恢复后的参数与 Adam 状态逐位一致，继续一步更新后仍逐位一致。
- 分片模型在所有 rank 上可执行带 KV cache 的 action 推理。

另有真实单进程 DCP 回归覆盖普通 Adam 状态、未使用参数缺少状态，以及已使用参数缺少动量字段时必须报错。CPU/Gloo 用例直接调用 torch fully_shard，不替代 CUDA/Accelerate 验证。

执行命令（当前环境禁用系统 ROS pytest 插件，并使用可写缓存）：

```bash
HF_DATASETS_CACHE=/tmp/lerobot-hf-datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync pytest \
  tests/policies/pi0_pi05/test_pi05_fsdp.py \
  tests/policies/pi0_pi05/test_pi05_recipe_alignment.py \
  tests/policies/pi0_pi05/test_pi05_memory.py \
  tests/policies/pi0_pi05/test_pi05_training_time_rtc.py \
  tests/policies/pi0_pi05/test_pi05_rtc.py \
  tests/configs/test_train_config_distributed.py \
  tests/distributed/test_policy_surface.py \
  tests/distributed/test_checkpoint_wrappers.py \
  tests/common/test_checkpoint_save_resume.py -q

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_DATASETS_CACHE=/tmp/lerobot-hf-datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync pytest tests/distributed/test_pi05_fsdp_multigpu.py -q -k 'not gloo'

CUDA_VISIBLE_DEVICES='' LEROBOT_TEST_FSDP_CPU=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_DATASETS_CACHE=/tmp/lerobot-hf-datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync pytest tests/distributed/test_pi05_fsdp_multigpu.py -q -k gloo
```

环境未安装 mypy，未执行类型检查；未将此项记录为通过。

## 完整尺寸模型测量

使用随机初始化的完整 PI05、三个 224×224 相机、200 个语言 token、50 步动作、每卡 batch 6、全局 batch 12。参数与 AdamW 状态 FP32，计算 BF16，开启 policy.gradient_checkpointing；DDP 使用脚本同款 BF16 通信 hook，FSDP 使用 Accelerate 的 BF16 mixed precision policy。每种布局独立进程，预热 2 步、计时 3 步。

| 指标 | DDP | 双卡 FSDP2 |
| --- | ---: | ---: |
| 训练峰值 allocated，取各 rank 最大值 | 82.22 GiB | 32.02 GiB |
| 训练峰值 reserved，取各 rank 最大值 | 83.26 GiB | 36.36 GiB |
| 初始化峰值 allocated | 30.94 GiB | 23.18 GiB |
| 每步耗时 | 1.310 秒 | 1.120 秒 |
| 全局吞吐 | 9.16 samples/s | 10.72 samples/s |

本次训练峰值 allocated 降低 61.0%，每步耗时降低 14.5%，吞吐提高 17.0%。原始逐 rank 数值见 [ddp.json](validation/ddp.json) 和 [fsdp.json](validation/fsdp.json)。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  benchmarks/benchmark_pi05_fsdp.py --dp-shard=1 \
  --output openspec/changes/add-pi05-fsdp-training/validation/ddp.json

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync torchrun --standalone --nproc-per-node=2 \
  benchmarks/benchmark_pi05_fsdp.py --dp-shard=2 \
  --output openspec/changes/add-pi05-fsdp-training/validation/fsdp.json
```

测量边界：输入常驻 GPU，不含数据解码、增强、processor、保存与恢复；无预训练权重或训练质量评估；短样本未报告统计置信区间。峰值为 PyTorch allocator 指标，不是整卡总占用，训练峰值含预热步骤。两条路径均开启 policy 重算，但普通 PI05 保留原有外层重算，FSDP 按联合层内部重算；数据对比的是这两条实际实现，不能把速度变化全部归因于通信。初始化仍先构建完整模型，未实现 meta 或 rank-0-only 加载。

## 支持范围

首版支持全参 PI05，参数使用 float32、计算 no/bf16。LoRA/PEFT、视觉 MEM、冻结视觉或仅训练 expert、torch.compile、自定义 wrap policy 明确拒绝。尚未验证真实数据集长训、四卡 HSDP、CPU offload 或变化拓扑下 PI05 恢复。

## Ctrl+C 退出修复验证（2026-10-23）

用户确认反复出现的退出后高功耗发生于 Ctrl+C。现场曾观察到 PRO 6000 在没有 CUDA compute 进程、仅占用 19 MiB 时仍为 P1、100% 利用率、约 124/133W；各卡运行一次小型 CUDA 运算并正常退出后恢复低功耗。此现象未定位到确定的驱动根因，不能单凭清理修复认定永久解决。

代码中确认的缺口是：原先仅正常训练结束调用 `accelerator.end_training()`，中断或异常会绕过清理。现在使用统一的退出上下文，SIGINT/SIGTERM 先记录停止请求，在各 rank 相同的训练边界通过 collective 协调退出；清理不增加新的 barrier。DataLoader 子进程忽略 SIGINT，继续提供当前 batch，并保留 SIGTERM 的默认终止行为。中断不额外保存 checkpoint 或发布模型。

| 检查 | 结果 |
| --- | --- |
| 清理单测、真实 DataLoader 信号行为、训练入口与既有训练回归 | 22 passed，6 skipped（CPU 环境跳过既有 GPU 用例） |
| 双 PRO 6000、真实 torchrun 外部信号测试 | 3 passed，50.10 秒 |
| Ruff 0.14.1 检查与格式、`git diff --check` | 通过 |

CPU 覆盖正常退出、训练异常、KeyboardInterrupt、Accelerator 部分初始化失败、tracker 清理异常、重复信号、collective 期间信号到达的竞态、单 rank 请求传播及 DataLoader 子进程退出。最小 fork 信号测试有一条 Python 3.13 多线程 fork 弃用警告；实际 DataLoader 测试使用 spawn。

CUDA 三组是 DDP + 终端 SIGINT、FSDP2 + 终端 SIGINT、FSDP2 + 仅 rank 0 收到 SIGTERM。每组使用缩小尺寸的真实 PI05、BF16、梯度重算、AdamW，以及真实 spawn DataLoader 子进程。两次更新成功后才发送信号，验证两 rank 在同一步停止、退出状态非零、进程组均已销毁，日志没有未销毁进程组警告。终端信号仅发给测试创建的独立进程组。

最后一轮 CUDA 测试结束后，没有执行恢复 CUDA 运算。`nvidia-smi` 显示两张 PRO 6000 均为 P8、0% 利用率、19 MiB，功耗分别约 13.16W / 8.44W，没有 CUDA compute 进程残留。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_DATASETS_CACHE=/tmp/lerobot-hf-datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync pytest \
  tests/distributed/test_shutdown.py \
  tests/scripts/test_train_shutdown.py \
  tests/scripts/test_train_remote_dispatch.py \
  tests/scripts/test_lerobot_train_batch_preprocessing.py \
  tests/training/test_multi_gpu.py \
  tests/training/test_visual_validation.py -q

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_DATASETS_CACHE=/tmp/lerobot-hf-datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache \
uv run --no-sync pytest tests/distributed/test_shutdown_multigpu.py -q
```

CPU 的真实 DataLoader 测试也需要本机 IPC 权限，沙箱限制 socket 时不可将其超时判断为实现失败。本次 CPU/CUDA 最终结果均在允许相应本机能力的环境执行。

验证边界：未中断用户的完整尺寸真实数据集长训练，也没有证明高功耗现象只由进程组未清理导致。初始化、评估或保存阶段的信号需要等到共享停止边界；SIGKILL、失去响应的 rank/驱动及 torchrun 的强制终止超时无法由 Python 清理保证。
