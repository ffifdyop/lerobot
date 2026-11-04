## Context

迁移起点：main=fbb811fc，takeover=b1a75228，research=2030311e；目标官方 main=ba83e4cf。takeover 有18个本地提交，research 另有4个提交。原工作区有四个未提交文件。

## Goals / Non-Goals

**Goals:** 必须保留可恢复历史、用户工作和现有本地功能，接入上游新接口，留下独立验证证据。

**Non-Goals:** 禁止真机运行、长时间训练、改写历史验证结果或未经授权强推远端。

## Decisions

1. 原 main/takeover/research 建立备份引用，main 仅快进；功能提交在独立 worktree 重放。研究分支用旧 takeover tip 作为 `rebase --onto` 的排除边界。
2. 上游原生 `dm05` 保持不变，本地 HTTP 客户端命名为 `dm05_remote`，使用上游动态 processor 工厂。禁止两个实现竞争同一注册名。
3. 上游 `lerobot.inference`、EnvTransition、ActionProvenance 和 remote context 是迁移基底。本地 view-mask 和 saliency 必须接入新版调用和结果接收边界；禁止恢复已删除的旧引擎并让两套接口混用。
4. 采用纯上游 PI05 结构，显式设置 `accelerator.fsdp.wrap_modules=["SiglipEncoderLayer"]`，由根 FSDP 单元承担其余参数，移除本地联合层运行时。上游 decoder 层的 `forward` 被联合 attention 绕过，不将它们作为分片边界。纯上游四组精度/重算短回归及完整尺寸双卡短测通过；后者每卡 batch 6 的峰值 allocated 为35.24 GiB，满足当前两张96GB卡的目标。保留必要 checkpoint 格式兼容与 lazy Adam 恢复，不保留联合层结构改造。
5. 本地 checkpoint lazy-state 与上游 optimizer unwrap、PEFT、EMA 增量按功能整合；退出清理独立于 PI05 分片实现保留。dtype 按上游 torch.dtype 接口处理。
6. 依赖以官方约束为基底，保留所需 openpi-client 和本地默认 extras，最终统一重新生成锁文件。测试用独立临时环境，不直接升级用户当前虚拟环境。

## Risks / Trade-offs

- 多个历史补丁依赖旧接口 → 以最终行为回归验收，不将 Git 自动合并成功视为正确。
- 名称迁移影响已保存 DM05 JSON → 更新受版本控制的配置，并为用户未提交配置保留备份后做明确的必要字段迁移。
- 新旧 checkpoint 参数名称不同 → 保留导出与加载映射，验证继续一步更新一致。
- 临时环境复用旧基础包 → 明确记录实际版本和测试边界，锁文件检查单独执行。

## Migration Plan

备份与 main 快进 → 纯上游短回归 → takeover rebase及兼容迁移 → 聚焦验收 → research rebase及回归 → 恢复用户工作 → 核验分支与文件。保留旧引用供回退；仅在完成验收后推进原功能分支。

## Validation boundaries

显存结论来自随机初始化完整尺寸 PI05 的合成驻留输入短测，不包括数据 I/O、checkpoint 保存或训练质量。新旧环境的 loss 配置不同，不能据此宣称相对旧联合层的吞吐优劣。原用户虚拟环境不升级，验证环境与依赖锁单独记录。
