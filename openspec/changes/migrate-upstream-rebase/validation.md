# 上游 rebase 迁移验证

目标上游：`ba83e4cf43737e2665be0aafcb5f513b83c7d0f3`。
本记录仅包含本次迁移的验证；历史 `add-pi05-fsdp-training/validation.md` 描述旧版本联合层实现。

## PI05 分片方案选择

- 采用上游模型结构，显式包裹 `SiglipEncoderLayer`；其余参数由根 FSDP 单元处理。
- CPU 缩小模型：前向、反向、Adam 更新、动作预测有限，重算开关对应的 FP32 loss 和梯度一致。
- 双卡原生路径：FP32/BF16 × 重算开/关，训练、完整权重导出和加载通过。
- 上游原始 optimizer restore 因未使用 `lm_head` 的 lazy Adam state 缺失失败；迁移保留严格的 lazy-state 兼容修复。
- 完整尺寸合成输入：4,143,404,816 参数，三路224图像、200 tokens、50步动作、每卡batch 6、BF16、重算开启；2步预热+3步测量，峰值 allocated 35.23984 GiB/卡，reserved 60.07031 GiB/卡，1.303秒/步。
- 完整尺寸测试使用原生14维动作 loss；历史联合层测试配置不同，未做同环境性能对照。没有运行真实数据长训练、真机或训练质量实验。

## 环境

独立临时环境：PyTorch 2.11.0+cu128、Accelerate 1.14.0、Transformers 5.22.1、huggingface-hub 1.33.0。两张 NVIDIA RTX PRO 6000 Blackwell 96GB。用户原有 `.venv` 保持原状。

## 最终集成验收

- takeover最终代码组合：推理、remote_inference、rollout、saliency、PI模型原生对照、flow/RTC、DM05 HTTP与checkpoint，共1103 passed / 20 skipped；一项上游CLI测试替身未声明新可选字段，补齐后该项单独复验通过（合计1104通过）。
- 配置与训练入口/退出：97通过；DataLoader IPC测试先被沙箱socket限制，允许本机IPC后该项通过，合计98通过。
- 图像增强/reader/recipe：112 passed / 2 skipped；原11项recipe断言保留。旧图像增强快照是Git LFS指针，未运行其快照比较。
- FSDP聚焦：17 CPU、10 CUDA类别通过；另用旧b1a75228+Transformers5.5.4真实生成的合成checkpoint验证新环境FP32/BF16双卡恢复与下一步更新严格相等。覆盖旧joint keys和旧vision container；不涉及用户训练权重。
- 最终代码双卡退出：DDP与FSDP终端SIGINT、FSDP仅rank0 SIGTERM，3/3通过；全部报告确认process group已销毁。launcher/group SIGTERM会同时终止DataLoader worker，可走异常清理而不是协调退出143，文档已明确此边界。
- 独立审查验证PEFT adapter、EMA及嵌套AcceleratedOptimizer/GradScaler实际CPU往返；保留上游missing-DCP异常契约的聚焦回归17通过。
- `uv lock --check --offline`通过（362包），本地openpi-client用相邻仓库相对路径，原`.venv`未修改。Ruff、脚本语法、DDP/FSDP真实argv的draccus解析通过。

研究分支的最终代码回归与原工作区恢复核验随后补充。详细FSDP证据与复现方式见`validation/MIGRATION_VALIDATION.md`及同目录JSON。
