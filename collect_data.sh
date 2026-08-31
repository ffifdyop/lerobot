#!/usr/bin/env bash
# SO-101 双臂遥操作数据采集脚本
# 支持多相机配置（OpenCV / RealSense）
#
# 使用方法:
#   1. 根据实际情况修改下方的串口路径和相机配置
#   2. chmod +x collect_data.sh && ./collect_data.sh

set -e

# 自动激活虚拟环境（lerobot-record 在 .venv/bin/ 中）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ==================== 配置区（请根据实际情况修改）====================


# --- 从臂串口 ---
# 左从臂串口
LEFT_FOLLOWER_PORT="/dev/ttyACM2"
# 右从臂串口
RIGHT_FOLLOWER_PORT="/dev/ttyACM3"

# --- 主臂串口 ---
# 左主臂串口
LEFT_LEADER_PORT="/dev/ttyACM0"
# 右主臂串口
RIGHT_LEADER_PORT="/dev/ttyACM1"

# --- 数据集配置 ---
# 数据集仓库ID（建议格式: {用户名}/{数据集名}）
DATASET_REPO_ID="ffifdyop/so101_bimanual_folding-cloth-720p-garment2"
# 任务描述
TASK_DESC="fold the garment on the table"
# 采集 episode 数量
NUM_EPISODES=30
# 每个 episode 时长（秒）
EPISODE_TIME_S=180
# 每次重置时间（秒）
RESET_TIME_S=120
# 帧率
# 单 USB2 控制器 + 三路相机同步解码 + Rerun 实时显示，30Hz 会掉到 ~20Hz。
# 折叠是慢动作任务，15fps 足够训练；若仍卡且不需要看画面，可再把
# --display_data 改成 false 关闭 Rerun 推流。
FPS=15
# 单个视频分片最大体积（MB）。多个 episode 的视频会拼进同一个 MP4，
# 直到累计大小超过此值才开新文件；单个 episode 超过该值时该集会独占一个（超限的）文件。
# 留空则使用默认值 200MB。
VIDEO_FILES_SIZE_IN_MB=1
# 是否推送到 HuggingFace Hub (true/false)
PUSH_TO_HUB=false
# 是否在已有数据集上续录 (true/false)
RESUME=false

# --- 相机配置 ---
# 当前 setup: 3 个相机
#   - left_wrist:  左臂 wrist 相机
#   - right_wrist: 右臂 wrist 相机
#   - top:         固定全局 top 相机（挂在左臂下）
#
# 支持两种相机类型:
#   - opencv: 使用 index_or_path (整数，如 0, 1, 2...)
#   - intelrealsense: 使用 serial_number_or_name
#
# 注意：左右臂的相机名称不要重复，否则后加载的会覆盖先加载的
#
# 已绑定固定路径（udev规则），插拔后不变
# 应用规则: sudo cp 99-so101-cameras.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules
#
# backend: 200 = Cv2Backends.V4L2。必须显式指定，否则 OpenCV 默认用 ANY/CAP_FFMPEG，
# 在 Linux 上会忽略 MJPG fourcc、只能拿到 640x480，导致 _validate_width_and_height 报错。
# 注意：三路 1080p@30 MJPG 在单 USB2 控制器(480M)上会饱和，导致录制循环掉到
# 16Hz 甚至 3Hz、丢帧、机械臂抖动。本机仅有一个 xHCI 控制器，拓扑无法分摊，
# 故默认降到 720p。如需 1080p，把下面的 width/height 改回 1920/1080（建议同时
# 把 FPS 降到 15）。
LEFT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video6, width: 1280, height: 720, fps: 30, fourcc: "MJPG", backend: 200},
  top:   {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: "MJPG", backend: 200}
}'

RIGHT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video8, width: 1280, height: 720, fps: 30, fourcc: "MJPG", backend: 200}
}'

# 示例配置（RealSense摄像头，取消下方注释使用）:
# LEFT_CAMERAS='{
#   left_wrist: {type: intelrealsense, serial_number_or_name: "123456789012", width: 640, height: 480, fps: 30},
#   top:        {type: intelrealsense, serial_number_or_name: "111111111111", width: 640, height: 480, fps: 30}
# }'
# RIGHT_CAMERAS='{
#   right_wrist: {type: intelrealsense, serial_number_or_name: "098765432109", width: 640, height: 480, fps: 30}
# }'

# ==================== 辅助函数 ====================

check_port() {
    local port=$1
    local name=$2
    if [ ! -e "$port" ]; then
        echo "[警告] $name 串口不存在: $port"
        echo "       可用串口列表:"
        ls -la /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "       (无可用串口)"
        echo ""
        echo "       请修改脚本中的串口路径，或使用以下命令查找正确的串口："
        echo "       ls -la /dev/serial/by-path/   # 按USB位置固定"
        echo "       ls -la /dev/serial/by-id/     # 按设备ID固定"
        exit 1
    fi
}

check_camera() {
    echo "[信息] 检测可用摄像头..."
    # OpenCV摄像头
    ls -la /dev/video* 2>/dev/null || true
    # RealSense摄像头
    if command -v rs-enumerate-devices &>/dev/null; then
        echo "[信息] RealSense 设备列表:"
        rs-enumerate-devices | grep -E "Serial Number|Device Name" || echo "       (未检测到RealSense设备)"
    fi
}

# ==================== 主流程 ====================

echo "========================================"
echo "SO-101 双臂遥操作数据采集"
echo "========================================"

# 检查串口
echo ""
echo "[步骤 1/4] 检查串口..."
check_port "$LEFT_FOLLOWER_PORT" "左从臂"
check_port "$RIGHT_FOLLOWER_PORT" "右从臂"
check_port "$LEFT_LEADER_PORT" "左主臂"
check_port "$RIGHT_LEADER_PORT" "右主臂"

# 赋权
echo ""
echo "[步骤 2/4] 设置串口权限..."
sudo chmod 666 "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" "$LEFT_LEADER_PORT" "$RIGHT_LEADER_PORT" 2>/dev/null || {
    echo "[警告] 无法修改串口权限，尝试使用当前权限继续..."
}

# 检测摄像头
echo ""
echo "[步骤 3/4] 检测摄像头..."
check_camera

# 确认配置
echo ""
echo "[步骤 4/4] 采集配置确认:"
echo "  左从臂端口: $LEFT_FOLLOWER_PORT"
echo "  右从臂端口: $RIGHT_FOLLOWER_PORT"
echo "  左主臂端口: $LEFT_LEADER_PORT"
echo "  右主臂端口: $RIGHT_LEADER_PORT"
echo "  数据集ID:   $DATASET_REPO_ID"
echo "  任务描述:   $TASK_DESC"
echo "  Episode数:  $NUM_EPISODES"
echo "  每段时长:   ${EPISODE_TIME_S}s"
echo "  帧率:       ${FPS}fps"
echo "  视频分片:   ${VIDEO_FILES_SIZE_IN_MB:-默认200}MB"
echo "  续录模式:   $RESUME"
echo ""
read -p "配置正确吗? 按 Enter 开始采集，或 Ctrl+C 退出..."

# 开始采集
echo ""
echo "[开始采集] 启动 lerobot-record..."
echo ""

lerobot-record \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port="$LEFT_FOLLOWER_PORT" \
    --robot.right_arm_config.port="$RIGHT_FOLLOWER_PORT" \
    --robot.id=bimanual_follower \
    --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
    --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
    --teleop.type=bi_so_leader \
    --teleop.left_arm_config.port="$LEFT_LEADER_PORT" \
    --teleop.right_arm_config.port="$RIGHT_LEADER_PORT" \
    --teleop.id=bimanual_leader \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.single_task="$TASK_DESC" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.fps="$FPS" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --dataset.video_files_size_in_mb="$VIDEO_FILES_SIZE_IN_MB" \
    --resume="$RESUME" \
    --display_data=true \
    --play_sounds=true

echo ""
echo "[完成] 数据采集结束"
