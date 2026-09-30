#!/bin/bash
# ee6d checkpoint rollout（引擎版）。用法: ./rollout_ee6d.sh
# 等价于 rollout.sh,但走 examples/so101_bimanual_joints_to_ee6d/rollout_engine_ee6d.py,
# 在脚本内部挂载双臂 FK/IK 处理器(ee6d <-> 关节)。
#
# 所有相机、串口、checkpoint 路径都在本文件配置,py 文件只从环境变量读取。

set -e

source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2

# ===== 用户配置区 =====

# checkpoint 路径(ee6d action 训练的 SmolVLA)
export ROLLOUT_CHECKPOINT="models/lehome/train/smolvla/smolvla-real_720p_ft_ee6d_20260914_191954/checkpoints/080000/pretrained_model"

# 任务 prompt(需与训练时一致)
export ROLLOUT_TASK="fold the garment on the table"

# 串口：一律用 /dev/serial/by-id/，不要写 /dev/ttyACM<N>（编号按接入顺序分配，
# 拔插一次就可能左右互换 → lerobot 拿错标定文件校验电机 → 弹出重建标定提示，
# 而按回车/输 c 都会写坏 EPROM 或标定文件）。by-id 按序列号固定。
# 若路径不存在：ls -la /dev/serial/by-id/
export LEFT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042915-if00"
export RIGHT_FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D046378-if00"

# 相机
export LEFT_WRIST_CAM="/dev/video0"
export RIGHT_WRIST_CAM="/dev/video8"
export TOP_CAM="/dev/video6"

# rename map: 机器人相机 -> policy 训练时的相机名(需与 checkpoint 的 preprocessor 一致)
export POLICY_CAM_TOP="observation.images.camera1"
export POLICY_CAM_LEFT_WRIST="observation.images.camera2"
export POLICY_CAM_RIGHT_WRIST="observation.images.camera3"

# 是否开启 rerun 可视化(true/false)
export ROLLOUT_DISPLAY_DATA="true"

# =====================

check_port() {
    local port=$1
    local name=$2
    if [ ! -e "$port" ]; then
        echo "[警告] $name 串口不存在: $port"
        echo "       可用串口列表:"
        ls -la /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "       (无可用串口)"
        exit 1
    fi
}

echo ""
echo "[步骤 1/2] 检查串口..."
check_port "$LEFT_FOLLOWER_PORT" "左从臂"
check_port "$RIGHT_FOLLOWER_PORT" "右从臂"

echo ""
echo "[步骤 2/2] 设置串口权限..."
sudo chmod 666 "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" 2>/dev/null || {
    echo "[警告] 无法修改串口权限,尝试使用当前权限继续..."
}

python examples/so101_bimanual_joints_to_ee6d/rollout_engine_ee6d.py
