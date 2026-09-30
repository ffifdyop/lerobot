## 1. 配置与控制模型

- [ ] 1.1 增加 takeover opt-in 配置、默认键位、Fault 阈值和启动期 capability 校验，验证关闭 takeover 时现有配置与数据 schema 不变
- [ ] 1.2 定义 `POLICY/HUMAN/RESUMING/PAUSED/FAULT` 阶段、输入事件和优先级仲裁，并覆盖 hold、latch、pause、stop 竞争场景
- [ ] 1.3 引入结构化 dispatch result，区分 policy proposal、selected action、robot command 与 robot 返回的 executed action

## 2. 主从臂同步与硬仲裁

- [ ] 2.1 实现兼容单臂和双臂前缀的 LeaderFollowerSync，以 follower 实际位置驱动 teleoperator feedback，且不修改 PID
- [ ] 2.2 接入双臂 torque 生命周期，保证接管时先阻止 policy dispatch、再释放 torque，恢复时重新进入 policy follow
- [ ] 2.3 增加同步层测试，验证左右 feature 路由、每 tick feedback、proposal 不驱动 leader，以及任一侧失败触发共享 Fault

## 3. RTC Shadow 与恢复隔离

- [ ] 3.1 为 RTC 增加 execute/shadow 生命周期、epoch 失效和可执行 queue 清理，证明 shadow proposal 永不进入 dispatch
- [ ] 3.2 实现 latest-only shadow observation/proposal 输出，并向 saliency sink 提交带 shadow mode 的同批次快照
- [ ] 3.3 实现 `RESUMING` 的 processor/interpolator reset、last executed action hold、fresh chunk 等待和平滑恢复
- [ ] 3.4 增加 RTC 并发测试，覆盖在途旧 chunk、shadow 请求积压、模式往返和 fresh resume

## 4. DAgger Takeover 控制循环

- [ ] 4.1 将 observation processing、相机 telemetry 和状态 telemetry 提升到所有阶段共享的逐 tick 路径，移除阶段切换中的阻塞式显示冻结
- [ ] 4.2 把事件仲裁、policy/human 单一 action 选择和 dispatch result 接入 DAgger continuous recording 路径
- [ ] 4.3 实现可替换的键盘事件后端和 listener teardown，默认支持 Shift hold、Tab latch、Space pause 与 Esc stop
- [ ] 4.4 保留 takeover 关闭、corrections-only 和既有 DAgger 状态转换的回归行为并补齐测试

## 5. 数据集与事件时间线

- [ ] 5.1 扩展 takeover 数据集 features，加入 policy action、有效位、control source 和独立 robot-space executed action schema
- [ ] 5.2 更新 Policy/Human 帧写入，确保 Human 帧对齐同观测 shadow proposal，proposal 缺失时使用有效位而非伪造零动作
- [ ] 5.3 为 Paused、Resuming 和 Fault 增加事件 sidecar/Rerun 时间线，禁止将 hold 帧写为训练示范
- [ ] 5.4 增加数据契约测试，覆盖 robot 裁剪返回值、policy/human 来源、不同 action schema 和非动作阶段过滤

## 6. Saliency 与可视化连续性

- [ ] 6.1 为 saliency snapshot、recorder 和 overlay 增加 queued-policy/shadow mode；Policy 保留队列 a_d，Shadow 独立提交并显式选择 proposal a₀，禁止伪造队列接受元数据；保留层/步筛选、预算和完整梯度算法
- [ ] 6.2 在 Rerun/Foxglove 首帧注册 phase、source、intervention、左右 tracking error、fault code 和 saliency mode 状态路径
- [ ] 6.3 保持阶段切换期间 manager/recorder/visualization 生命周期不重启，并保证 saliency 失败只降级旁路
- [ ] 6.4 增加跨阶段相机刷新、latest-only shadow saliency、模式标识和显示故障隔离测试

## 7. Fault 分级与双臂共享域

- [ ] 7.1 实现可配置的 recoverable/fatal Fault 分类、连续周期计数和锁存状态，禁止自动恢复
- [ ] 7.2 实现 recoverable Fault 的人工确认/重新同步，以及 fatal Fault 的失败 episode 保存和控制 teardown
- [ ] 7.3 聚合左右 feedback、torque 和通信结果，验证部分成功时两侧均停止且诊断显示继续
- [ ] 7.4 增加 Fault 场景测试，覆盖 tracking error、关键状态缺失、非有限动作、通信异常和 teardown 后迟到事件

## 8. CLI、文档与整体验证

- [ ] 8.1 更新 `rollout.sh` 的 DAgger、双主臂、takeover、数据记录和 shadow saliency 参数示例
- [ ] 8.2 更新用户文档，说明控制键位、阶段/来源、Shadow Saliency 语义、数据字段、Fault 恢复和研究原型边界
- [ ] 8.3 运行 rollout、interactive rollout、saliency、Rerun/Foxglove 和数据集相关测试，并执行受影响文件的 pre-commit 检查
- [ ] 8.4 按 Policy→Human→Paused→Human→Resuming→Policy 和 recoverable/fatal Fault 清单完成低速双臂上机验收，确认相机与 saliency 全程持续显示
