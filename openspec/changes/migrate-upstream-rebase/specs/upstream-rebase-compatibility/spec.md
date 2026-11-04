## ADDED Requirements

### Requirement: 分支迁移必须可恢复
系统 MUST 保留原分支引用和用户未提交内容，并保持重放后的功能分支包含目标上游历史。

#### Scenario: 迁移与用户工作共存
- **WHEN** 在独立 worktree 完成 rebase
- **THEN** 原提交可由备份引用访问，原未提交内容有校验和备份，研究提交仅重放一次

### Requirement: DM05 实现具有独立身份
系统 MUST 同时支持上游原生 `dm05` 和 HTTP 客户端 `dm05_remote`，且不得混用模型配置和处理器。

#### Scenario: 两种策略配置解析
- **WHEN** 分别构建 dm05 与 dm05_remote 配置并经工厂解析
- **THEN** 两者得到各自实现，远程客户端保持原 HTTP 协议，旧部署配置具有明确迁移说明

### Requirement: 新推理接口保持本地控制和解释语义
系统 MUST 在上游 EnvTransition 与推理结果接收机制中保持 view-mask 版本隔离、saliency 对齐与记录语义。

#### Scenario: 遮罩改变期间返回旧动作
- **WHEN** 遮罩更新前启动的推理在更新后完成
- **THEN** 旧动作和解释结果均不可作为新版本有效输出，且原始相机数据不被覆盖

#### Scenario: 同步和RTC路径
- **WHEN** 本地策略使用同步或RTC推理并启用saliency
- **THEN** 解释使用真正被接受的动作、噪声、continuation和视角上下文，strategy仍收到上游定义的transition

### Requirement: PI05 分片迁移必须有当前版本证据
系统 MUST 验证选定PI05分片路径在新配置下的训练与恢复，并明确未验证组合。

#### Scenario: BF16重算和恢复
- **WHEN** 小模型执行双卡BF16重算、保存恢复并继续更新
- **THEN** loss/梯度有限、状态恢复符合容差与预期，普通模型可加载导出权重；历史通过结果不得作为本次证据

#### Scenario: 上游候选替代联合层
- **WHEN** 决定删除本地联合层
- **THEN** 必须先证明替代路径正确且满足目标显存要求，否则保留最小适配并说明原因
