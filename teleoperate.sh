# --- 串口：一律用 /dev/serial/by-id/，不要写 /dev/ttyACM<N> ---
# ttyACM 编号按「接入顺序」分配且断开后会被复用，拔插一次就可能重排；
# 一旦左右臂串口互换，lerobot 会拿另一条臂的标定文件去校验电机，
# is_calibrated 判定失败 → 弹出重建标定提示（按回车或输 c 都会写坏 EPROM/文件）。
# by-id 按适配器序列号固定，与插拔顺序无关。
# 若下面任一路径不存在：ls -la /dev/serial/by-id/
# 左从臂串口
LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042915-if00"
# 右从臂串口
RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D046378-if00"

# --- 主臂串口 ---
# 左主臂串口
LEFT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E089335-if00"
# 右主臂串口
RIGHT_LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D044325-if00"

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

lerobot-teleoperate \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port="$LEFT_FOLLOWER_PORT" \
    --robot.right_arm_config.port="$RIGHT_FOLLOWER_PORT" \
    --robot.left_arm_config.use_degrees=false \
    --robot.right_arm_config.use_degrees=false \
    --robot.id=bimanual_follower \
    --teleop.type=bi_so_leader \
    --teleop.left_arm_config.port="$LEFT_LEADER_PORT" \
    --teleop.right_arm_config.port="$RIGHT_LEADER_PORT" \
    --teleop.left_arm_config.use_degrees=false \
    --teleop.right_arm_config.use_degrees=false \
    --teleop.id=bimanual_leader