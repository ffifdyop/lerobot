# Tasks: add-openpi-client-rollout

## 1. 脚本骨架与 CLI 配置

- [x] 1.1 新增 `src/lerobot/scripts/lerobot_openpi_rollout.py`：draccus 配置 dataclass（server host/port、fps、duration、task prompt、相机键映射、bi_so_follower 机器人配置），并注册 CLI 入口
- [x] 1.2 在 `pyproject.toml [project.scripts]` 注册 `lerobot-openpi-rollout`；`openpi-client` 保持惰性导入（未安装时输出含安装命令的错误提示并退出）

## 2. Server 通信与观测组织

- [x] 2.1 实现 server 探活：连接机器人前调用 server 元数据接口，失败时给出含 deploy.sh/host/port 提示的错误并以非零状态退出
- [x] 2.2 实现观测组织：从 `robot.get_observation()` 按相机映射取三路 uint8 (H,W,C) 图像，按左臂电机顺序+右臂电机顺序拼接 12 维 float32 state，附 `prompt`，组装为 server 约定的键

## 3. 动作执行与控制循环

- [x] 3.1 实现动作队列：缓存 server 返回的 action chunk，逐周期取一帧，前 6 维发左臂、后 6 维发右臂，耗尽后先请求新 chunk 再继续
- [x] 3.2 实现固定 FPS（默认 30）的 deadline 式主循环，支持 duration 上限；try/finally 保证任何退出路径（正常/Ctrl-C/异常）都调用 `robot.disconnect()` 且中断时立即停止发动作

## 4. 启动脚本与验证

- [x] 4.1 新增 `rollout_openpi.sh`：复用 `rollout.sh` 的串口检查/赋权与相机配置风格，增加 server host/port 参数，调用 `lerobot-openpi-rollout`
- [ ] 4.2 实机验证：openpi server（deploy.sh）+ client 闭环运行，核对观测键/关节顺序/动作拆分正确，确认 Ctrl-C 与异常路径均能安全断开机器人
