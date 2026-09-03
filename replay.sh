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


lerobot-replay \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=/dev/ttyACM2 \
    --robot.right_arm_config.port=/dev/ttyACM3 \
    --robot.left_arm_config.use_degrees=false \
    --robot.right_arm_config.use_degrees=false \
    --robot.id=bimanual_follower \
    --dataset.repo_id=ffifdyop/so101_bimanual_folding-cloth-720p \
    --dataset.episode=51 \
    --dataset.fps=30
