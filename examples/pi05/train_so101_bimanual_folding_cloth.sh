#!/usr/bin/env bash

# Fine-tune pi0.5 on the local ffifdyop/so101_bimanual_folding-cloth-720p dataset.
#
# The defaults launch two CUDA processes (one per GPU) with Distributed Data
# Parallel. BATCH_SIZE is per GPU, so the default global batch size is 12.
# Override any setting through environment variables, for example:
#
#   NUM_GPUS=2 BATCH_SIZE=6 STEPS=10000 \
#     ./examples/pi05/train_so101_bimanual_folding_cloth.sh
#
# Extra command-line arguments are forwarded to lerobot-train. To train the
# entire model on a large-memory GPU, use:
#
#   TRAIN_EXPERT_ONLY=false FREEZE_VISION_ENCODER=false BATCH_SIZE=8 \
#     ./examples/pi05/train_so101_bimanual_folding_cloth.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_REPO_ID="${DATASET_REPO_ID:-ffifdyop/so101_bimanual_folding-cloth-720p}"
DATASET_ROOT="${DATASET_ROOT:-/home/gsp/.cache/huggingface/lerobot/ffifdyop/so101_bimanual_folding-cloth-720p}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/lerobot/pi05_so101_bimanual_folding_cloth_${RUN_TIMESTAMP}}"
JOB_NAME="${JOB_NAME:-pi05_so101_bimanual_folding_cloth}"

BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
STEPS="${STEPS:-40000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
SEED="${SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-0.03125}"
EVAL_STEPS="${EVAL_STEPS:-500}"
# 0 keeps every frame from the selected episodes, so validation remains episode-complete.
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
OPENPI_AUGMENTATION="${OPENPI_AUGMENTATION:-true}"
OPENPI_AUG_CROP_SCALE="${OPENPI_AUG_CROP_SCALE:-0.95}"
OPENPI_AUG_ROTATION_DEGREES="${OPENPI_AUG_ROTATION_DEGREES:-5.0}"
OPENPI_AUG_COLOR_PROBABILITY="${OPENPI_AUG_COLOR_PROBABILITY:-0.5}"
OPENPI_AUG_BRIGHTNESS="${OPENPI_AUG_BRIGHTNESS:-0.3}"
OPENPI_AUG_CONTRAST="${OPENPI_AUG_CONTRAST:-0.4}"
OPENPI_AUG_HUE="${OPENPI_AUG_HUE:-0.1}"
# ``bf16`` compresses only gradient all-reduces; parameters and AdamW state remain FP32.
DDP_COMM_HOOK="${DDP_COMM_HOOK:-bf16}"
DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-25}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# This recipe uses model/tokenizer files already present in the local Hugging Face cache.
# Force offline mode so a malformed shell proxy cannot trigger a Hub metadata request.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Dataset not found at: ${DATASET_ROOT}" >&2
    echo "Set DATASET_ROOT to the local LeRobot dataset directory." >&2
    exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate

# Resolve a cached Hub repo id to its concrete snapshot directory. Passing a local directory to
# lerobot-train prevents the processor loader from performing even a Hub metadata HEAD request.
if [[ ! -e "${PRETRAINED_PATH}" ]]; then
    PRETRAINED_PATH="$(
        python -c '
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], local_files_only=True))
' "${PRETRAINED_PATH}"
    )"
fi
echo "HF offline mode: HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "Pretrained snapshot: ${PRETRAINED_PATH}"
echo "Distributed training: ${NUM_GPUS} GPUs, per-GPU batch=${BATCH_SIZE}, global batch=$((NUM_GPUS * BATCH_SIZE))"

if [[ "${OPENPI_AUGMENTATION}" != "true" && "${OPENPI_AUGMENTATION}" != "false" ]]; then
    echo "OPENPI_AUGMENTATION must be true or false, got: ${OPENPI_AUGMENTATION}" >&2
    exit 1
fi

# Match OpenPI's JAX training precision: retain trainable parameters, gradients, and
# optimizer state in FP32, while AMP runs eligible forward/backward ops in BF16.
torchrun --standalone --nproc-per-node="${NUM_GPUS}" "$(command -v lerobot-train)" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.return_uint8=true \
    --dataset.eval_split="${EVAL_SPLIT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --dataset.image_transforms.enable="${OPENPI_AUGMENTATION}" \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs="{\"openpi\":{\"type\":\"OpenPIImageAugmentation\",\"kwargs\":{\"crop_scale\":${OPENPI_AUG_CROP_SCALE},\"rotation_degrees\":${OPENPI_AUG_ROTATION_DEGREES},\"color_probability\":${OPENPI_AUG_COLOR_PROBABILITY},\"brightness\":${OPENPI_AUG_BRIGHTNESS},\"contrast\":${OPENPI_AUG_CONTRAST},\"hue\":${OPENPI_AUG_HUE}}}}" \
    --policy.train_on_padded_action_dims=true \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.empty_cameras=0 \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
    --policy.gradient_checkpointing=true \
    --policy.dtype=float32 \
    --accelerator.mixed_precision=bf16 \
    --accelerator.ddp.comm_hook="${DDP_COMM_HOOK}" \
    --accelerator.ddp.bucket_cap_mb="${DDP_BUCKET_CAP_MB}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.optimizer_lr=4e-5 \
    --policy.optimizer_betas='[0.9,0.95]' \
    --policy.optimizer_eps=1e-8 \
    --policy.optimizer_weight_decay=0.01 \
    --policy.optimizer_grad_clip_norm=1.0 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=40000 \
    --policy.scheduler_decay_lr=4e-6 \
    --ema.enable=false \
    --ema.decay=0.999 \
    --ema.use_for_eval=false \
    --wandb.enable=true \
    --wandb.project=lehome-pi05 \
    --wandb.entity=2943872339-zhejiang-university \
    --wandb.disable_artifact=true \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --prefetch_factor=2 \
    --steps="${STEPS}" \
    --eval_steps="${EVAL_STEPS}" \
    --max_eval_samples="${MAX_EVAL_SAMPLES}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq=25 \
    --env_eval_freq=0 \
    --seed="${SEED}" \
    "$@"
