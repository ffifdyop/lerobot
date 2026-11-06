#!/usr/bin/env bash
# FineART-VLA 双臂 SO101 rollout；硬件设置参考 ~/Dev/Manipulation/lerobot/{rollout,teleoperate}.sh。
# bash rollout.sh check：只解析配置；bash rollout.sh：加载模型，等待 /start。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode=run
case "${1:-}" in
    check|run) mode="$1"; shift ;;
    insert|pull) TASK_MODE="$1"; shift ;;
    -h|--help|help)
        cat <<'HELP'
用法：bash rollout.sh [check|run|insert|pull] [lerobot-rollout 参数...]
  check  检查 checkpoint 和配置，不加载权重、不连接硬件。
  run    加载模型并连接双臂，相机就绪后等待 /start（默认）。
  insert 插入任务（默认）；pull 拔出并放回桌面。

示例：
  bash rollout.sh check
  bash rollout.sh
  bash rollout.sh pull
  GPU_IDS=2 ROLLOUT_POLICY_PATH=/path/to/pretrained_model bash rollout.sh
  bash rollout.sh --inference.type=sync --duration=60

启动后输入 /start：先生成首个 subtask，再执行动作。
待机时可用 /autosteer <完整目标> 换任务；/reset 后再次 /start 会重新规划。
人工接管默认启用，沿用 teleoperate.sh 的 bimanual_leader 标定 ID 和两条主臂固定串口：
  bash rollout.sh check
  配置检查后去掉 check 启动；终端空输入行短按空格接管，再按空格恢复，无需回车。
  LEADER_ID、LEFT_LEADER_PORT、RIGHT_LEADER_PORT 可用环境变量覆盖；TELEOP_TAKEOVER=0 关闭接管。
  保留 /teleop 和 /resume；命令中的空格不触发切换，Ctrl-U 清空未完成输入。
HELP
        exit 0
        ;;
    ""|--*) ;;
    *) echo "未知模式：$1；使用 bash rollout.sh --help 查看用法。" >&2; exit 2 ;;
esac

# ===== 常用参数：可编辑默认值，也可用同名环境变量覆盖 =====
POLICY_PATH="${ROLLOUT_POLICY_PATH:-${POLICY_PATH:-$repo_root/outputs/fineart_so101/train_20261105T054916785617Z/train/checkpoints/last/pretrained_model}}"
# POLICY_PATH="${ROLLOUT_POLICY_PATH:-${POLICY_PATH:-/data/.cache/checkpoints/pi05_so101_bimanual_insert-2_20261106_110706/001690/pretrained_model}}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042915-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D046378-if00}"
# 与 teleoperate.sh 共用标定 ID；分别读取 bimanual_leader_left/right 的现有标定。
LEADER_ID="${LEADER_ID:-bimanual_leader}"
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E089335-if00}"
RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D044325-if00}"
LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA:-/dev/video0}"
RIGHT_WRIST_CAMERA="${RIGHT_WRIST_CAMERA:-/dev/video8}"
TOP_CAMERA="${TOP_CAMERA:-/dev/video6}"
# 沿用参考脚本的采集设置；训练数据为 640×480，可按实际采集条件覆盖宽高。
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-360}"         # 每次 /start 的运行秒数；0 为不限时
AUTOSTEER_INTERVAL_S="${AUTOSTEER_INTERVAL_S:-3}"
# 整体任务进入高层规划；首个 subtask 由模型生成，不预置人工低层指令。
TASK_MODE="${TASK_MODE:-insert}"
case "$TASK_MODE" in
    insert) default_goal="Fully insert the black-handled screwdriver's metal shaft into the white sleeve's top opening." ;;
    pull) default_goal="Pull the black-handled screwdriver completely out of the white sleeve and place it back on the table." ;;
    *) echo "TASK_MODE 必须为 insert 或 pull" >&2; exit 2 ;;
esac
ROLLOUT_GOAL="${ROLLOUT_GOAL:-$default_goal}"
TELEOP_TAKEOVER="${TELEOP_TAKEOVER:-1}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/outputs/fineart_so101/uv_cache}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-2,3}}"
# 本地推理模型使用第一张可见 GPU，不会自动分摊到两张卡。

cd "$repo_root"
# 不同步依赖，避免改变已经用于训练的环境。
uv_cmd=(uv run --no-sync --project "$repo_root")
if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
    uv_cmd=(uv run --no-project --python "$LEROBOT_PYTHON")
fi

rollout_args=(
    --strategy.type=base
    --policy.path="$POLICY_PATH"
    --device=cuda
    --robot.type=bi_so_follower
    --robot.id="$ROBOT_ID"
    --robot.left_arm_config.port="$LEFT_FOLLOWER_PORT"
    --robot.right_arm_config.port="$RIGHT_FOLLOWER_PORT"
    --robot.left_arm_config.use_degrees=false
    --robot.right_arm_config.use_degrees=false
    --robot.left_arm_config.cameras="{wrist: {type: opencv, index_or_path: '$LEFT_WRIST_CAMERA', width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS, fourcc: MJPG}}"
    --robot.right_arm_config.cameras="{wrist: {type: opencv, index_or_path: '$RIGHT_WRIST_CAMERA', width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS, fourcc: MJPG}}"
    --robot.cameras="{top: {type: opencv, index_or_path: '$TOP_CAMERA', width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS, fourcc: MJPG}}"
    # BiSO 自动给两侧 wrist 加前缀，得到 top/left_wrist/right_wrist。
    --rename_map='{}'
    --fps="$FPS"
    --task=""
    --autosteer_goal="$ROLLOUT_GOAL"
    --duration="$DURATION"
    --interactive=true
    --autosteer_interval_s="$AUTOSTEER_INTERVAL_S"
    --inference.type=rtc
    # RTCConfig 的 execution_horizon 默认为 10，与参考脚本一致。
    # 不固定传入 RTC 专用字段，以便追加 --inference.type=sync 时仍可解析。
)

if [[ "$TELEOP_TAKEOVER" == 1 ]]; then
    rollout_args+=(
        --takeover.enabled=true
        --teleop.type=bi_so_leader
        --teleop.id="$LEADER_ID"
        --teleop.left_arm_config.port="$LEFT_LEADER_PORT"
        --teleop.right_arm_config.port="$RIGHT_LEADER_PORT"
        --teleop.left_arm_config.use_degrees=false
        --teleop.right_arm_config.use_degrees=false
    )
elif [[ "$TELEOP_TAKEOVER" != 0 ]]; then
    echo 'TELEOP_TAKEOVER 必须为 0 或 1' >&2
    exit 2
fi

if [[ "$mode" == check ]]; then
    exec "${uv_cmd[@]}" python - "${rollout_args[@]}" "$@" <<'PY'
from pathlib import Path

from lerobot.configs import parser
from lerobot.rollout.configs import RolloutConfig
# Register the same camera, robot and policy choices as the real CLI.
import lerobot.scripts.lerobot_rollout  # noqa: F401


@parser.wrap()
def check(cfg: RolloutConfig) -> None:
    if cfg.policy is None or cfg.policy.type != "fineart_vla":
        raise ValueError("此脚本需要 FineART-VLA checkpoint")
    checkpoint = Path(cfg.policy.pretrained_path).expanduser().resolve()
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    print("配置检查通过；未加载权重或连接硬件。")
    print(f"Checkpoint: {checkpoint}")
    print(f"Robot: {cfg.robot.type}, ID: {cfg.robot.id}")
    print(f"Followers: {cfg.robot.left_arm_config.port}, {cfg.robot.right_arm_config.port}")
    print(f"use_degrees: left={cfg.robot.left_arm_config.use_degrees}, right={cfg.robot.right_arm_config.use_degrees}")
    print(f"Inference: {cfg.inference.type}, device: {cfg.device}, interactive: {cfg.interactive}")
    print(f"Startup goal: {cfg.autosteer_goal}")
    print(f"Teleop takeover: {cfg.takeover}")
    if cfg.takeover.enabled:
        print(f"Leader: {cfg.teleop.type}, ID: {cfg.teleop.id}")
        print(f"Leader config: {cfg.teleop}")
        print("仅配置验证；未验证串口、标定、主从臂误差或实际跟随频率。")
    if cfg.inference.type == "rtc":
        print(f"RTC execution_horizon: {cfg.inference.rtc.execution_horizon}")
    print(f"Input features: {cfg.policy.input_features}")
    print(f"Top camera: {cfg.robot.cameras}")
    print(f"Left camera: {cfg.robot.left_arm_config.cameras}")
    print(f"Right camera: {cfg.robot.right_arm_config.cameras}")


check()
PY
fi

printf 'FineART checkpoint: %s\n可见 GPU: %s\n' "$POLICY_PATH" "$CUDA_VISIBLE_DEVICES"
cat <<'COMMANDS'
进入交互终端后输入 /start：先生成首个 subtask，再执行策略动作。
COMMANDS
printf '完整目标：%s\n' "$ROLLOUT_GOAL"
echo '待机时用 /autosteer <完整目标> 切换任务，再 /start。'
echo '/reset 停止并回到启动姿态；/stop 结束会话（默认返回启动姿态）。'
if [[ "$TELEOP_TAKEOVER" == 1 ]]; then
    echo '已启用主臂跟随。聚焦终端，空输入行短按空格接管，等待 Human control active 再拖动；再次空格恢复。'
    echo '空格无需回车，两次短按间隔至少 0.75 秒；恢复后松开主臂等待对齐和规划。也可用 /teleop、/resume。'
fi
# 串口权限由当前账户提供，不自动执行 sudo chmod。
# 额外 CLI 参数放在最后，可覆盖上面的默认设置。
exec "${uv_cmd[@]}" lerobot-rollout "${rollout_args[@]}" "$@"
