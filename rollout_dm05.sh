source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 客户端不载权重（推理在 DM05 服务端），无需占本机 GPU

# DM05 推理服务地址（服务端启动：bash /data/opendm/script/deploy.sh）
SERVER_HOST="127.0.0.1"
SERVER_PORT="7891"
# 部署配置目录：连接信息 + 编码参数（execution_horizon/use_history/camera_keys）
# 注意该目录在 .gitignore 的 models/ 下，为本机部署产物
POLICY_PATH="models/opendm/deploy/dm05_folding"

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

# 检查串口
echo ""
echo "[步骤 1/4] 检查串口..."
check_port "$LEFT_FOLLOWER_PORT" "左从臂"
check_port "$RIGHT_FOLLOWER_PORT" "右从臂"

# 检查 DM05 服务可达（策略加载阶段也会探测，这里提前给出更友好的提示）
if ! python -c "import socket; socket.create_connection(('$SERVER_HOST', $SERVER_PORT), timeout=5)" 2>/dev/null; then
    echo "[错误] DM05 推理服务不可达: $SERVER_HOST:$SERVER_PORT"
    echo "       请先启动服务: bash /data/opendm/script/deploy.sh"
    exit 1
fi
echo "[步骤 2/4] DM05 服务可达: $SERVER_HOST:$SERVER_PORT"

# 赋权
echo "[步骤 3/4] 设置串口权限..."
sudo chmod 666 "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" 2>/dev/null || {
    echo "[警告] 无法修改串口权限，尝试使用当前权限继续..."
}

LEFT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

RIGHT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video8, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

# 顶层相机挂在 robot 顶层（观测键为 top，与 DM05 camera_keys 槽位 "1" 对应）；
# 左右腕相机观测键为 left_wrist/right_wrist（槽位 "2"/"3"）。
# 相机键名与 config.json 的 camera_keys 一致，无需 --rename_map。
TOP_CAMERAS='{
  top: {type: opencv, index_or_path: /dev/video6, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

echo "[步骤 4/4] 启动 rollout..."

# DM05 当前不支持 saliency；这里只显示原始相机、机器人状态和动作曲线。
lerobot-rollout \
  --strategy.type=base \
  --policy.path=$POLICY_PATH \
  --inference.type=sync \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.left_arm_config.use_degrees=false \
  --robot.right_arm_config.use_degrees=false \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --robot.cameras="$TOP_CAMERAS" \
  --task="fold the garment on the table" \
  --duration=360 \
  --display_data=true \
  --display_mode=rerun
