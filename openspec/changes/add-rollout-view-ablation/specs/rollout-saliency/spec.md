## ADDED Requirements

### Requirement: 视角消融下的 saliency 必须复现实际 masked batch

启用 View Ablation 时，saliency snapshot SHALL 在 View Mask processor 之后捕获，并 MUST 携带生成该 batch 的 `view_mask_revision` 与 masked canonical views。explainer MUST 回放策略实际收到的黑图 batch；被 mask 视角的 gradient/attention 结果即使非零也必须保留，禁止人为清零、隐藏或解释为 logical missing-view attention mask。

#### Scenario: 黑图视角产生非零敏感度

- **WHEN** masked 视角的实际黑图输入产生非零 gradient × input 或 attention 质量
- **THEN** saliency map 保留模型真实值，overlay 标记该视角为 MASKED，控制回路与原始相机显示不受影响

#### Scenario: 回放 batch 与执行 batch 一致

- **WHEN** revision N 的 policy action 生成 saliency snapshot
- **THEN** snapshot 中的图像 tensor、mask 集合、revision、noise 与 chunk 参数均对应 revision N 实际执行 batch

### Requirement: saliency 必须丢弃过期 View Mask revision

Mask Transition 后，saliency manager/recorder MUST 在回放和发布前检查 snapshot revision；任何不等于最新可执行 revision 的 snapshot MUST 被丢弃。APPLYING 与 FAILED 期间禁止为未执行动作发布新 saliency。该丢弃属于正常 provenance 过滤，禁止计入 saliency 连续失败降级计数，也禁止影响 policy 控制。

#### Scenario: recorder queue 中的旧 snapshot 被丢弃

- **WHEN** revision N snapshot 已排入异步 recorder，但回放前 View Mask 已切换为 revision N+1
- **THEN** revision N snapshot 不生成 overlay、视频帧或 Rerun saliency 数据，且不增加错误计数

#### Scenario: APPLYING 不展示旧动作解释

- **WHEN** Mask Application State 为 APPLYING 且系统正在保持旧 revision 的最后动作
- **THEN** saliency UI 不把旧 snapshot 显示为新 revision 的解释，并明确显示 APPLYING 状态
