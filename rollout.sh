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
RENAME_MAP='{}'

# 以下 SmolVLA checkpoint 训练时用的是 camera1/2/3，需启用对应 map：
# RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
# RENAME_MAP='{"observation.images.top":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}'
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_freeze_20260920_174448/checkpoints/006000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_no-wrist_20260912_205349/checkpoints/040000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_enhanced_20260913_144858/checkpoints/040000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_20260902_105132/checkpoints/080000/pretrained_model
# POLICY_PATH=models/lehome/train/smolvla/smolvla-real_720p_ft_20260830_142257/checkpoints/020000/pretrained_model
# POLICY_PATH=models/train/pi05/pi05-real_ft_20260701_202015/checkpoints/040000/pretrained_model

POLICY_PATH=models/lerobot/pi05_so101_bimanual_folding_cloth_20260921_224353/checkpoints/040000/pretrained_model

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

  # --- 当前 pi05 checkpoint 的 saliency 用法（参数已在上方启用）---
  # CUDA_VISIBLE_DEVICES=2,3 时，默认控制模型用第一张可见 GPU，解释器副本用第二张。
  # method=attention：解释 RTC 接受的 chunk 中实际保留的首个动作 a_d；
  #   d 是队列裁去的前缀步数，不一定为 0，也不表示该动作最终已发送到电机。
  #   pi05 action expert 的动作 token 对图像 patch 的注意力形成每相机 16×16 图；
  #   当前 pi05 checkpoint：action expert 共 18 层（可选层号 0–17），去噪 10 轮（0–9）；
  #   默认聚合全部 18×10 个层/轮次组合，保留原始注意力份额，不只在相机内归一化。
  #   若切换到上方列出的 SmolVLA checkpoint：VLM/expert 共 16 层、去噪 10 轮；
  #   attention saliency 只捕获 8 个 cross-attention 层（1,3,5,…,15），默认聚合 8×10 个组合。
  #   SmolVLA 图像 patch 网格为 8×8，预算分类为各相机、language、state。
  #   当前 pi05 帧顶预算条及 Rerun saliency/budget/* 曲线包含各相机、prompt（文本与离散状态）和 action。
  #   如只看 pi05 的部分层/轮次，在 --saliency.method=attention 后加入（编号均从 0 开始）：
  #     --saliency.attention_layers='[0,4,8]' \
  #     --saliency.attention_steps='[0,5,9]' \
  #   SmolVLA 层选择示例为 '[1,5,9]'，不能选偶数 self-attention 层；
  #   只设置一个选择器时另一个维度仍全选。
  # 切换像素敏感度图：--saliency.method=grad_x_input，计算 a_d 的 ∂‖a_d‖²/∂图像；
  #   grad_x_input 不接受 attention_layers/attention_steps 参数。
  # 当前 pi05 产物：outputs/saliency/<时间戳>/{top,left_wrist,right_wrist}.mp4；
  #   video_fps=1 是视频播放帧率，旁路可能丢弃已接受 chunk 的诊断帧。
  # 色标：默认全程统一尺度（norm_scope=run：跨相机、跨时刻可比）；
  #   想恢复逐 chunk 自适应对比度（局部对比最强，但"平静时刻"也会被拉红）：
  #   --saliency.norm_scope=chunk
