source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 左从臂串口
LEFT_FOLLOWER_PORT="/dev/ttyACM2"
# 右从臂串口
RIGHT_FOLLOWER_PORT="/dev/ttyACM3"

# --- 主臂串口 ---
# 左主臂串口
LEFT_LEADER_PORT="/dev/ttyACM0"
# 右主臂串口
RIGHT_LEADER_PORT="/dev/ttyACM1"

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



LEFT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 30,fourcc: "MJPG"},
  top:   {type: opencv, index_or_path: /dev/video6, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

RIGHT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video8, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

# Map robot camera names (left_top/left_wrist/right_wrist) to the names the
# policy was trained on (camera1/camera2/camera3). Same map saved in the
# checkpoint's policy_preprocessor.json, but rollout does not auto-load it.
RENAME_MAP='{"observation.images.left_top":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
# RENAME_MAP='{"observation.images.left_top":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}'
POLICY_PATH=models/train/smolvla/smolvla-real_720p_ft_20260830_142257/checkpoints/last/pretrained_model
# bPOLICY_PATH=models/train/pi05/pi05-real_ft_20260701_202015/checkpoints/040000/pretrained_model

lerobot-rollout \
  --strategy.type=base \
  --policy.path=$POLICY_PATH \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --rename_map="$RENAME_MAP" \
  --task="fold the garment on the table" \
  --duration=120

