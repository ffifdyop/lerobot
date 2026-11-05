#!/usr/bin/env bash
# Configurable FineART-VLA recipe. The default `check` mode does not launch training.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/outputs/fineart_so101/uv_cache}"
mode="${1:-check}"
if (($#)); then shift; fi

if [[ "$mode" == setup ]]; then
    cd "$repo_root"
    exec uv sync --locked --extra training --extra fineart_vla --extra fineart_vla_kernels --extra wandb "$@"
fi

# ===== 常用训练参数：可直接修改默认值，也可用同名环境变量覆盖 =====
BATCH_SIZE="${BATCH_SIZE:-24}"          # 每卡 batch；双卡全局 batch = BATCH_SIZE * 2
LR="${LR:-4e-5}"                     # 初始/峰值学习率
MIN_LR="${MIN_LR:-4e-6}"                 # 留空：LR / 10
EPOCHS="${EPOCHS:-5}"                # 按实际训练集帧数自动换算 steps
STEPS="${STEPS:-2000}"                   # 留空按 EPOCHS；非空时覆盖 EPOCHS
TRAIN_RATIO="${TRAIN_RATIO:-1.0}"    # 1.0 全量训练；0.8 按任务留出约 20% episode 验证
NUM_WORKERS="${NUM_WORKERS:-4}"      # 每卡 DataLoader workers
SEED="${SEED:-1000}"                # 训练随机种子；原生数据划分按每个任务末尾留出
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}" # 梯度检查点；false 关闭以减少重计算
# 临时覆盖：GRADIENT_CHECKPOINTING=false bash scripts/train_fineart_so101.sh train
SAVE_FREQ="${SAVE_FREQ:-1000}"       # checkpoint 周期；0 只保存最后一次
LOG_FREQ="${LOG_FREQ:-20}"
EVAL_FREQ="${EVAL_FREQ:-1000}"       # 验证 loss 周期；TRAIN_RATIO=1 时自动关闭
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}" # 0 验证全部留出帧
# smoke 默认每卡 batch=1、workers=0、warmup=2、每步日志；命令行仍可覆盖。
if [[ "$mode" == smoke ]]; then
    BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
    NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
    WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-2}"
    LOG_FREQ=1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$repo_root/outputs/fineart_so101/datasets_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$repo_root/outputs/fineart_so101/modules_cache}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,2}}"
export WANDB_PROJECT="${WANDB_PROJECT:-fineart-so101}"
# Same W&B team as the PI05 folding recipe, with this project's own project/run names.
export WANDB_ENTITY="${WANDB_ENTITY:-2943872339-zhejiang-university}"
export WANDB_MODE="${WANDB_MODE:-online}"

training_args=(
    --batch-size "$BATCH_SIZE" --lr "$LR" --epochs "$EPOCHS"
    --train-ratio "$TRAIN_RATIO" --num-workers "$NUM_WORKERS" --seed "$SEED"
    --warmup-steps "$WARMUP_STEPS" --save-freq "$SAVE_FREQ" --log-freq "$LOG_FREQ"
    --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
    --eval-freq "$EVAL_FREQ" --max-eval-samples "$MAX_EVAL_SAMPLES"
)
if [[ -n "$STEPS" ]]; then training_args+=(--steps "$STEPS"); fi
if [[ -n "$MIN_LR" ]]; then training_args+=(--min-lr "$MIN_LR"); fi

if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
    # Use an existing environment without syncing or changing its editable installation.
    exec uv run --no-project --python "$LEROBOT_PYTHON" \
        python "$repo_root/examples/training/train_fineart_so101.py" "$mode" "${training_args[@]}" "$@"
fi

exec uv run --no-sync --project "$repo_root" \
    python "$repo_root/examples/training/train_fineart_so101.py" "$mode" "${training_args[@]}" "$@"
