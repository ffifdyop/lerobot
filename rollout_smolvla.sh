source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3

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



LEFT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

RIGHT_CAMERAS='{
  wrist: {type: opencv, index_or_path: /dev/video8, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

# 顶层相机：key 不加 left_/right_ 前缀，机器人观测键即为 top/left_wrist/right_wrist，
# 与数据集（so101_bimanual_folding-cloth-720p）的键名完全一致
TOP_CAMERAS='{
  top: {type: opencv, index_or_path: /dev/video6, width: 1280, height: 720, fps: 30,fourcc: "MJPG"}
}'

# 与所选 checkpoint 的 train_config.json / policy_preprocessor.json 里的 rename_map 逐字一致。
# 注意：必须显式传——CLI 的 map 无条件覆盖 checkpoint policy_preprocessor.json 里保存的
# rename step（传 {} 等于清空），不存在"自动加载 checkpoint 里的 map"。
# 传错 = 把观测键改成 policy 不认识的键名 → pi05/smolvla 报
# "All image features are missing from the batch"（batch 里只剩 rename 后的 camera1/2/3）。

# 当前 checkpoint = pi05（train_config.json 里 rename_map 为空，input_features 已是
# observation.images.{top,left_wrist,right_wrist}，与机器人观测键天然一致）→ 不要 rename。
# RENAME_MAP='{}'

# 以下 SmolVLA checkpoint 训练时用的是 camera1/2/3，需启用对应 map：
RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
# RENAME_MAP='{"observation.images.top":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}'
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_freeze_20260920_174448/checkpoints/006000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_no-wrist_20260912_205349/checkpoints/040000/pretrained_model
POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_enhanced_20260913_144858/checkpoints/040000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_20260902_105132/checkpoints/080000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_20260830_142257/checkpoints/020000/pretrained_model
# POLICY_PATH=models/train/pi05/pi05-real_ft_20260701_202015/checkpoints/040000/pretrained_model

# POLICY_PATH=models/lerobot/pi05_so101_bimanual_folding_cloth_20260921_224353/checkpoints/040000/pretrained_model

lerobot-rollout \
  --strategy.type=base \
  --policy.path=$POLICY_PATH \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --robot.left_arm_config.use_degrees=false \
  --robot.right_arm_config.use_degrees=false \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras="$LEFT_CAMERAS" \
  --robot.right_arm_config.cameras="$RIGHT_CAMERAS" \
  --robot.cameras="$TOP_CAMERAS" \
  --rename_map="$RENAME_MAP" \
  --task="fold the garment on the table" \
  --duration=360 \
  --saliency.enabled=true \
  --saliency.method=attention \
  --saliency.video_fps=1 \
  --display_data=true --display_mode=rerun \
  --view_mask.enabled=true \
  # --view_mask.initial_views='["top"]' \

  # --- saliency 叠加热力图（上面四行已启用）---
  # 前提：第 5 行 CUDA_VISIBLE_DEVICES 是多卡掩码（explainer 副本跑第二张卡）。
  # method=attention（当前）：动作 token a₀ 的交叉注意力落在哪些 patch 上 ——
  #   每相机 8×8 Patch 注意力图（块状+网格线，原始份额、跨相机可比，不重归一化），
  #   帧顶注意力预算条（相机/语言/state 五份，总和=1）+ rerun 曲线 saliency/budget/*。
  # 切回像素敏感度图（∂‖a₀‖²/∂x）：--saliency.method=grad_x_input
  # 产物：outputs/saliency/<时间戳>/{camera1,camera2,camera3}.mp4
  #   （快照每个 chunk 一帧 ≈1.3 帧/s，video_fps=1 让播放节奏≈真实节奏）
  # 色标：默认全程统一尺度（norm_scope=run：跨相机、跨时刻可比）；
  #   想恢复逐 chunk 自适应对比度（局部对比最强，但"平静时刻"也会被拉红）：
  #   --saliency.norm_scope=chunk

