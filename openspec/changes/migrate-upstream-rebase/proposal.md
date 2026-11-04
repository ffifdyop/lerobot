## Why

本地 takeover 与 research 分支基于较旧上游，直接重放会遇到推理接口、DM05 命名和 FSDP 配置冲突。需要将本地 main 同步到官方 main，再以可回退的 rebase 保留本地功能并接入上游更新。

## What Changes

- 建立原 main、takeover、research 的备份引用，在独立 worktree 重放本地提交，保存并恢复用户未提交文件。
- **BREAKING**：HTTP DM05 客户端注册名改为 `dm05_remote`，`dm05` 使用上游原生模型；提供配置与脚本迁移说明。
- 将 saliency、view mask 与 EE6D 接入上游 inference/EnvTransition 契约，保持远程推理及策略行为。
- 采用上游 FSDP 精度与梯度累积配置，验证纯上游 PI05 路径；仅保留仍有必要的专用适配及 checkpoint/退出修复。
- 同步依赖锁、训练脚本和验收记录；研究分支只重放旧 takeover tip 之后的提交。

## Capabilities

### New Capabilities

- `upstream-rebase-compatibility`：分支迁移、现有功能兼容及验证边界。

### Modified Capabilities

无；DM05 两种实现的名称分离及迁移约束由本次兼容性规格定义，并同步现有客户端说明。

## Impact

Git 分支历史、依赖清单、PI05/FSDP、DM05 客户端、inference/rollout、saliency、训练脚本、测试与 OpenSpec。原工作区未提交文件必须可恢复。

## Non-goals

不执行真机操作、真实数据集长训或科学性能结论；不因 Git 无冲突或旧测试记录宣称新版本验证通过；不自动发布或强推远端历史。
