## Context

训练器已经通过 Accelerate 配置 FSDP2、混合精度和 DCP。PI05 联合 attention 不调用 decoder 层的 forward，普通的按 decoder 类名分片会错过参数收集。现有 LoRA 目标正则和预训练模型依赖原参数路径。

## Goals / Non-Goals

**Goals:** 支持全参 PI05 的 FSDP2 前反向、逐层重算、训练内 loss 验证和保存恢复；导出模型必须能由普通 PI05 加载。

**Non-Goals:** 本次禁止 PEFT、视觉 MEM、自定义 wrap policy 与 torch.compile 的 FSDP 组合。禁止将缺少 GPU 的测试跳过记录为多卡验证通过。

## Decisions

1. 只在 FSDP prepare 前启用联合层结构，保留参数对象，兼容训练器已创建的优化器。每对 VLM/expert decoder 必须由一个联合层唯一注册；禁止同一模块同时注册在原路径和联合层下。
2. 原语言模型的 layers 保留不注册参数的调用视图，prefix/suffix 路径经视图进入联合层，再调用对应 decoder。联合训练必须经过 nn.Module.__call__。重算在联合层 forward 内部执行，位于 FSDP 输入精度转换之后；参数重收集由 FSDP pre-backward 钩子负责。
3. 常规图像路径按 SigLIP encoder 层分片，token embedding 按实际参数共享情况选择独立边界；其余参数由根单元负责。保持现有 attention 数学，不采用逐 Linear 分片造成大量小通信。
4. DCP 使用稳定的联合层内部键，完整权重导出映射回现有 PI05 键。普通加载同时识别原始键与 DCP 合并后的联合层键，禁止静默覆盖映射冲突。
   不均匀分片收集后的张量可能仍引用填充存储，导出前仅复制这些视图。恢复优化器时依据 DCP 元数据保留实际存在的参数状态，允许未使用参数没有 Adam 状态，同时严格检查已使用参数的动量字段。
5. 参数与优化器状态使用 FP32；支持 no/bf16 计算模式。先验证无重算，再验证现有 policy.gradient_checkpointing；不启用尚未接线的 accelerator.activation_checkpointing。
6. Ctrl+C/SIGTERM 先记录停止请求，不在未完成的 CUDA/NCCL 工作中直接抛异常。各 rank 在固定训练边界以一个标量 all-reduce 协调请求，再跳出训练；不会为了中断启动保存或发布。finally 负责 end_training 及必要的进程组销毁，清理不新增 barrier；重复信号不会在训练 rank 内打断清理。DataLoader 初始化时忽略 SIGINT，继续供应当前 batch；SIGTERM 保持默认终止语义，以便 DataLoader 强制清理。初始化失败、普通异常及直接 KeyboardInterrupt 同样进入清理。SIGKILL 或进程/驱动失去响应无法由 Python 保证恢复，torchrun 的终止超时仍然有效。

## Risks / Trade-offs

- 参数重归属可能破坏 prefix/suffix、冻结模式与 checkpoint → 使用同一参数对象、不注册的调用视图及数值/键集合测试；FSDP 初版限定全参。
- BF16 重算输入可能与首轮前向不同 → 必须将 checkpoint 放在联合层内部；禁止在外部对 FSDP 联合层再次 checkpoint。两进程 CPU 实验已复现外部重算的 dtype 元数据冲突，并以真实分片回归验证修复。
- BF16 的模块级转换与 AMP 有差异 → 加入真实双卡 no/bf16 测试，不承诺逐 bit 相同。
- 默认沙箱不可见 NVIDIA 驱动，但获准的本机测试进程可访问 CUDA。双卡专项与 CPU/Gloo 测试已经执行；硬件、命令与完整模型测量结果记录在 validation.md。
- 模型在 prepare 前仍会完整构建 → 本次不实现 meta/rank-0-only 加载，不保证初始化峰值按卡数缩小。
- 2026-10-23 用户确认高功耗发生在 Ctrl+C 后，训练器原有清理只位于正常退出路径。缺失清理是可确认代码问题，但不足以证明高功耗根因；实际退出测试需将进程退出、清理和 GPU 读数分别记录，不依靠退出后探测运算掩盖状态。

## Migration Plan

默认脚本继续使用 DDP。FSDP 由显式环境变量开启并要求关闭 LoRA。已有模型无需迁移；关闭 FSDP 即回到原训练路径。新增 DCP 恢复使用本版本的内部结构。

## Open Questions

双卡完整尺寸合成数据测量已完成，见 validation.md；真实数据读取、保存开销及长期训练质量仍需后续实验。
