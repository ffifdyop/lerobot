#!/bin/bash
# Rollout on bimanual SO-101 with policy inference served by the openpi
# websocket server (start it first with: bash ~/Dev/Manipulation/openpi/deploy.sh)
set -e

source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# openpi policy server 地址
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"

# 左从臂串口
LEFT_FOLLOWER_PORT="/dev/ttyACM2"
# 右从臂串口
RIGHT_FOLLOWER_PORT="/dev/ttyACM3"

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

# 检查串口（rollout 不需要主臂）
echo ""
echo "[步骤 1/3] 检查串口..."
check_port "$LEFT_FOLLOWER_PORT" "左从臂"
check_port "$RIGHT_FOLLOWER_PORT" "右从臂"

# 赋权
echo ""
echo "[步骤 2/3] 设置串口权限..."
sudo chmod 666 "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" 2>/dev/null || {
    echo "[警告] 无法修改串口权限，尝试使用当前权限继续..."
}

LEFT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video0, width: 1920, height: 1080, fps: 30,fourcc: "MJPG"},
  top:   {type: opencv, index_or_path: /dev/video6, width: 1920, height: 1080, fps: 30,fourcc: "MJPG"}
}'

RIGHT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video8, width: 1920, height: 1080, fps: 30,fourcc: "MJPG"}
}'

echo ""
echo "[步骤 3/3] 连接 openpi server ($SERVER_HOST:$SERVER_PORT) 并开始 rollout..."
echo ""

lerobot-openpi-rollout \
  --host="$SERVER_HOST" \
  --port="$SERVER_PORT" \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port="$LEFT_FOLLOWER_PORT" \
  --robot.right_arm_config.port="$RIGHT_FOLLOWER_PORT" \
  --robot.left_arm_config.use_degrees=false \
  --robot.right_arm_config.use_degrees=false \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --task="fold the garment on the table" \
  --fps=30 \
  --duration=120
