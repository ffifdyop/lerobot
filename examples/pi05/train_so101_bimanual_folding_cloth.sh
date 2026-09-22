#!/usr/bin/env bash

# Fine-tune pi0.5 on the local ffifdyop/so101_bimanual_folding-cloth-720p dataset.
#
# The defaults launch two CUDA processes (one per GPU) with Distributed Data
# Parallel. BATCH_SIZE is per GPU, so the default global batch size is 24.
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

BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-15000}"
SEED="${SEED:-1000}"
# Complete episodes used for validation. These are spread across the dataset rather than
# taking a contiguous tail. Set EVAL_EPISODES=random to sample a reproducible random holdout.
EVAL_EPISODES="${EVAL_EPISODES:-[90,91,92,93,94,95]}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-6}"
EVAL_SEED="${EVAL_SEED:-1000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
# 0 keeps every frame from the selected episodes, so validation remains episode-complete.
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
# ``bf16`` compresses only gradient all-reduces; parameters and AdamW state remain FP32.
DDP_COMM_HOOK="${DDP_COMM_HOOK:-none}"
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

# LeRobot's native eval_split holds out the last episodes in dataset.episodes. Reorder that
# allowlist so the explicitly selected complete episodes are at the end, then derive the exact
# split fraction. This dataset has one task; fail clearly rather than mis-split a multi-task set.
EPISODE_SPLIT_OUTPUT="$(
    python -c '
import json
import random
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
eval_spec = sys.argv[2]
total_episodes = int(info["total_episodes"])
total_tasks = int(info.get("total_tasks", 1))

if total_tasks != 1:
    raise SystemExit(f"Expected a single-task dataset, found {total_tasks} tasks")
if eval_spec.lower() == "random":
    eval_count = int(sys.argv[3])
    eval_seed = int(sys.argv[4])
    if not 0 < eval_count < total_episodes:
        raise SystemExit(f"EVAL_NUM_EPISODES must be in [1, {total_episodes - 1}]")
    eval_episodes = sorted(random.Random(eval_seed).sample(range(total_episodes), eval_count))
else:
    eval_episodes = json.loads(eval_spec)
if not isinstance(eval_episodes, list) or not eval_episodes:
    raise SystemExit("EVAL_EPISODES must be a non-empty JSON list or random")
if any(type(ep) is not int for ep in eval_episodes):
    raise SystemExit("Every EVAL_EPISODES entry must be an integer")
if len(eval_episodes) != len(set(eval_episodes)):
    raise SystemExit("EVAL_EPISODES contains duplicate indices")
invalid = [ep for ep in eval_episodes if not 0 <= ep < total_episodes]
if invalid:
    raise SystemExit(f"EVAL_EPISODES contains out-of-range indices: {invalid}")
if len(eval_episodes) >= total_episodes:
    raise SystemExit("EVAL_EPISODES must leave at least one training episode")

eval_set = set(eval_episodes)
ordered_episodes = [ep for ep in range(total_episodes) if ep not in eval_set] + eval_episodes
print(json.dumps(ordered_episodes, separators=(",", ":")))
print(repr(len(eval_episodes) / total_episodes))
print(json.dumps(eval_episodes, separators=(",", ":")))
' "${DATASET_ROOT}/meta/info.json" "${EVAL_EPISODES}" "${EVAL_NUM_EPISODES}" "${EVAL_SEED}"
)"
mapfile -t EPISODE_SPLIT <<< "${EPISODE_SPLIT_OUTPUT}"
DATASET_EPISODES="${EPISODE_SPLIT[0]}"
EVAL_SPLIT="${EPISODE_SPLIT[1]}"
SELECTED_EVAL_EPISODES="${EPISODE_SPLIT[2]}"
echo "Validation episodes: ${SELECTED_EVAL_EPISODES}"

# Match OpenPI's JAX training precision: retain trainable parameters, gradients, and
# optimizer state in FP32, while AMP runs eligible forward/backward ops in BF16.
torchrun --standalone --nproc-per-node="${NUM_GPUS}" "$(command -v lerobot-train)" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.return_uint8=true \
    --dataset.episodes="${DATASET_EPISODES}" \
    --dataset.eval_split="${EVAL_SPLIT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
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
    --policy.optimizer_lr=1e-5 \
    --policy.scheduler_warmup_steps=350 \
    --policy.scheduler_decay_steps=10000 \
    --policy.scheduler_decay_lr=2.5e-6 \
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
