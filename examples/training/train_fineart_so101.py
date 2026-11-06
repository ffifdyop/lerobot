# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Prepare or launch a configurable SO101 FineART recipe through the native trainer.

Prefer the companion scripts/train_fineart_so101.sh launcher. `check` inspects data
and serializes a native config without loading weights, fitting FAST, or training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import draccus
import torch
from huggingface_hub import snapshot_download

from lerobot.configs import FeatureType
from lerobot.configs.default import DatasetConfig, PeftConfig, WandBLoggingConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

if TYPE_CHECKING or (_datasets_available and _av_available):
    import pyarrow.parquet as pq

    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.datasets.language_render import active_at, render_sample
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.recipe import TrainingRecipe
    from lerobot.policies.fineart_vla.fit_fast_tokenizer import resolve_fast_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path("/data/DataCaption/Flora/outputs/so101_bimanual_insert_2/subtask/dataset")
DEFAULT_CHECKPOINT = "lerobot/FineART-VLA-subtask-midtrained"
DEFAULT_WANDB_ENTITY = "2943872339-zhejiang-university"
EXPERT_LORA_TARGETS = r".*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)"
VLM_LORA_TARGETS = (
    r"model\.paligemma_with_expert\.paligemma\.model\.language_model\..*\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def parse_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "train"), nargs="?", default="check")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default="ffifdyop/so101_bimanual_insert-2")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Local snapshot or HF model ID.")
    parser.add_argument(
        "--run-dir", type=Path, help="Artifacts go here; native training uses its train/ child."
    )
    parser.add_argument("--steps", type=int, help="Override --epochs. Default: smoke=20; otherwise auto.")
    parser.add_argument("--epochs", type=float, default=5.0, help="Training epochs when --steps is unset.")
    parser.add_argument(
        "--train-ratio", type=float, default=1.0, help="Episode train fraction; the remainder is eval."
    )
    parser.add_argument("--batch-size", type=int, help="Per GPU: smoke=1; check/train=4.")
    parser.add_argument("--num-workers", type=int, help="Default: smoke=0; check/train=4.")
    parser.add_argument(
        "--max-lr", "--peak-lr", "--lr", "--learning-rate", dest="lr", type=float, default=1e-5
    )
    parser.add_argument(
        "--final-lr", "--min-lr", dest="min_lr", type=float, help="Endpoint; default: max LR / 10."
    )
    parser.add_argument("--warmup-steps", type=int, help="Default: smoke=2; otherwise 100.")
    parser.add_argument("--save-freq", type=int, default=1000, help="0 saves only the final checkpoint.")
    parser.add_argument("--log-freq", type=int, help="Default: smoke=1; otherwise 20.")
    parser.add_argument("--eval-freq", type=int, default=1000, help="0 disables held-out loss evaluation.")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 evaluates all held-out frames.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--distributed-backend", choices=("fsdp", "ddp"), default="fsdp")
    parser.add_argument("--dp-shard", type=int, help="GPUs per shard group. Default: fsdp=2, ddp=1.")
    parser.add_argument("--dp-replicate", type=int, help="Number of replicas/groups. Default: fsdp=1, ddp=2.")
    parser.add_argument("--fsdp-reshard-after-forward", type=parse_bool, default=True)
    parser.add_argument("--fsdp-cpu-offload", type=parse_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--use-lora", type=parse_bool, default=False)
    parser.add_argument("--lora-r", "--lora-rank", dest="lora_r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--freeze-vlm", type=parse_bool, default=False)
    parser.add_argument("--freeze-vision-encoder", type=parse_bool, default=False)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "fineart-so101"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"), default=os.environ.get("WANDB_MODE", "online")
    )
    args = parser.parse_args(argv)
    args.batch_size = args.batch_size if args.batch_size is not None else (1 if args.mode == "smoke" else 4)
    args.num_workers = (
        args.num_workers if args.num_workers is not None else (0 if args.mode == "smoke" else 4)
    )
    args.min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    args.warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else (2 if args.mode == "smoke" else 100)
    )
    args.log_freq = args.log_freq if args.log_freq is not None else (1 if args.mode == "smoke" else 20)
    if args.dp_shard is None:
        args.dp_shard = 2 if args.distributed_backend == "fsdp" else 1
    if args.dp_replicate is None:
        args.dp_replicate = 1 if args.distributed_backend == "fsdp" else 2
    if args.dp_shard < 1 or args.dp_replicate < 1:
        parser.error("dp-shard and dp-replicate must be positive")
    if args.distributed_backend == "fsdp" and args.dp_shard < 2:
        parser.error("fsdp requires dp-shard >= 2; use ddp with dp-shard=1 for replicas only")
    if args.distributed_backend == "ddp" and args.dp_shard != 1:
        parser.error("ddp requires dp-shard=1; use fsdp for parameter sharding")
    args.num_gpus = args.dp_shard * args.dp_replicate
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.batch_size < 1 or args.num_workers < 0 or not 0 < args.lr < float("inf"):
        parser.error("batch-size/lr must be positive and finite; num-workers must be nonnegative")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("epochs must be positive and finite")
    if not 0 < args.train_ratio <= 1:
        parser.error("train-ratio must be in (0, 1]")
    if not 0 <= args.min_lr <= args.lr:
        parser.error("min-lr must be finite and in [0, lr]")
    if args.lora_r < 1 or args.lora_alpha < 1:
        parser.error("lora-r and lora-alpha must be positive")
    if min(args.warmup_steps, args.save_freq, args.eval_freq, args.max_eval_samples) < 0 or args.log_freq < 1:
        parser.error("warmup/save/eval/sample limits must be nonnegative; log-freq must be positive")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.run_dir = args.run_dir or REPO_ROOT / "outputs" / "fineart_so101" / f"{args.mode}_{stamp}"
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def training_schedule(args: argparse.Namespace, train_frames: int) -> tuple[int, int]:
    """Mirror native data sharding; steps count frames, not action-chunk length."""
    if train_frames < 1:
        raise ValueError("The training split has no frames")
    global_batch = args.batch_size * args.num_gpus
    steps_per_epoch = (train_frames + global_batch - 1) // global_batch
    steps = args.steps
    if steps is None:
        steps = 20 if args.mode == "smoke" else math.ceil(args.epochs * steps_per_epoch)
    return steps, steps_per_epoch


def visible_gpu_ids(args: argparse.Namespace) -> list[str]:
    """Require exactly one distinct visible GPU per worker in the requested topology."""
    devices = [device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "0,2").split(",")]
    if not all(devices) or len(devices) != args.num_gpus or len(set(devices)) != args.num_gpus:
        raise ValueError(
            f"GPU_IDS (or CUDA_VISIBLE_DEVICES) must list {args.num_gpus} distinct GPUs: "
            f"dp_shard={args.dp_shard} * dp_replicate={args.dp_replicate}; got {devices}"
        )
    return devices


def configure_distributed(cfg: TrainPipelineConfig, args: argparse.Namespace) -> None:
    """Select native FSDP2 or DDP, keeping joint-decoder parameters in the root FSDP unit."""
    cfg.accelerator.mixed_precision = "bf16"
    cfg.parallelism.dp_replicate = args.dp_replicate
    cfg.parallelism.dp_shard = args.dp_shard
    cfg.parallelism.resolve(args.num_gpus)
    if args.distributed_backend == "fsdp":
        # Joint layers read Gemma decoder weights without calling decoder.forward().
        # Only SigLIP encoder layers have independent FSDP units; the decoder, LM head,
        # and expert stay together in the root unit so their parameters are gathered.
        cfg.accelerator.fsdp.wrap_modules = ["SiglipEncoderLayer"]
        cfg.accelerator.fsdp.reshard_after_forward = args.fsdp_reshard_after_forward
        cfg.accelerator.fsdp.cpu_offload = args.fsdp_cpu_offload
    else:
        cfg.accelerator.ddp.find_unused_parameters = True


def configure_finetuning(policy: FineARTVLAConfig, args: argparse.Namespace) -> PeftConfig | None:
    """Apply explicit trainability controls without leaving inactive high-level-only batches."""
    policy.use_peft = args.use_lora
    policy.gradient_checkpointing = args.gradient_checkpointing
    policy.train_expert_only = args.freeze_vlm
    policy.freeze_vision_encoder = args.freeze_vision_encoder or args.freeze_vlm
    policy.text_loss_weight = 0.0 if args.freeze_vlm else 1.0
    policy.fast_action_loss_weight = 0.0 if args.freeze_vlm else 1.0
    policy.enable_fast_action_loss = not args.freeze_vlm
    policy.auto_fit_fast_tokenizer = not args.freeze_vlm
    if args.freeze_vlm:
        if policy.recipe is None or "low_level_execution" not in policy.recipe.get("blend", {}):
            raise ValueError("Frozen VLM mode needs the low_level_execution subtask recipe")
        low_level = copy.deepcopy(policy.recipe["blend"]["low_level_execution"])
        low_level["weight"] = 1.0
        policy.recipe = {"blend": {"low_level_execution": low_level}}
    if not args.use_lora:
        return None
    # PI05's stock defaults target only the expert. Include language adapters for text/FAST
    # supervision, but exclude vision and LM-head projections that use direct weight access.
    targets = EXPERT_LORA_TARGETS
    if not args.freeze_vlm:
        targets = f"{targets}|{VLM_LORA_TARGETS}"
    return PeftConfig(
        method_type="LORA", r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=f"({targets})"
    )


def checkpoint_directory(source: str) -> Path:
    local = Path(source).expanduser()
    if local.is_dir():
        checkpoint = local.resolve()
    else:
        offline = os.environ.get("HF_HUB_OFFLINE", "1").lower() not in {"0", "false"}
        checkpoint = Path(snapshot_download(source, local_files_only=offline)).resolve()
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint / name}")
    return checkpoint


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_dataset(dataset: LeRobotDataset, recipe: TrainingRecipe) -> dict[str, Any]:
    """Check real language columns, independently of the curation readiness flags."""
    counts: Counter[str] = Counter(frames=0, blank_subtasks=0, blank_tasks=0)
    examples: list[dict[str, Any]] = []
    tasks = dataset.meta.tasks
    if tasks is None:
        raise ValueError("Dataset has no task metadata")
    task_map = {int(index): str(task) for task, index in tasks["task_index"].items()}
    columns = ["index", "episode_index", "frame_index", "timestamp", "task_index", "language_persistent"]
    for parquet in sorted((dataset.root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                counts["frames"] += 1
                label = active_at(row["timestamp"], persistent=row["language_persistent"], style="subtask")
                text = (label or {}).get("content") or ""
                if not isinstance(text, str) or not text.strip():
                    counts["blank_subtasks"] += 1
                    if len(examples) < 5:
                        examples.append({key: row[key] for key in ("episode_index", "frame_index")})
                if not task_map.get(row["task_index"], "").strip():
                    counts["blank_tasks"] += 1
    if counts["frames"] != len(dataset) or counts["blank_subtasks"] or counts["blank_tasks"]:
        raise ValueError(f"Language coverage failed: {dict(counts)}; blank examples={examples}")

    video_files = list((dataset.root / "videos").rglob("*.mp4"))
    broken = [str(path) for path in video_files if not path.is_file()]
    if not video_files or broken:
        raise FileNotFoundError(f"Missing video files or broken links: {broken[:5]}")
    decoded = 0
    rendered_examples = []
    for episode in dataset.meta.episodes:
        start, stop = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        for index in sorted({start, stop - 1}):
            sample = dataset[index]
            if tuple(sample["action"].shape) != (50, 12):
                raise ValueError(f"Expected SO101 [50,12] action chunk; got {sample['action'].shape}")
            for camera in dataset.meta.camera_keys:
                if sample[camera].ndim != 3 or sample[camera].shape[0] != 3:
                    raise ValueError(f"Invalid camera tensor: {camera}: {sample[camera].shape}")
                decoded += 1
            if len(rendered_examples) < 2:
                rendered = {}
                for name, component in (recipe.blend or {}).items():
                    output = render_sample(
                        recipe=component,
                        persistent=sample["language_persistent"],
                        events=sample.get("language_events", []),
                        t=float(sample["timestamp"]),
                        sample_idx=int(sample["index"]),
                        task=sample["task"],
                    )
                    if output is None:
                        raise ValueError(f"Recipe {name} did not render at frame {index}")
                    rendered[name] = output
                rendered_examples.append({"index": index, "recipes": rendered})

    markers = dataset.root / "meta" / "curation" / "frame_annotations.parquet"
    if markers.exists():
        table = pq.read_table(markers, columns=["exclude_confirmed"])
        counts["curation_excluded_frames_used"] = sum(bool(value) for value in table.column(0).to_pylist())
    return {
        **counts,
        "episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "camera_keys": dataset.meta.camera_keys,
        "decoded_camera_frames": decoded,
        "video_files": len(video_files),
        "interval_filter_applied": False,
        "readiness_flags_used_as_gate": False,
        "recipe_examples": rendered_examples,
    }


def build_config(
    args: argparse.Namespace, checkpoint: Path
) -> tuple[TrainPipelineConfig, LeRobotDataset, LeRobotDataset, LeRobotDataset | None]:
    recipe_path = REPO_ROOT / "src/lerobot/configs/recipes/subtask.yaml"
    # Load the checkpoint's own config (including fast_skip_tokens), then adapt its features.
    policy = FineARTVLAConfig.from_pretrained(
        checkpoint,
        cli_overrides=["--device=cpu", f"--recipe_path={recipe_path}"],
    )
    if policy.chunk_size != 50:
        raise ValueError("This SO101 baseline expects checkpoint.chunk_size=50")
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
        delta_timestamps={"action": [i / 30 for i in range(50)]},
    )
    if dataset.fps != 30 or not dataset.meta.has_language_columns:
        raise ValueError("Expected a 30 FPS LeRobot dataset with language columns")
    features = dataset_to_policy_features(dataset.features)
    if features["action"].shape != (12,) or features["observation.state"].shape != (12,):
        raise ValueError("This baseline expects a bimanual SO101 with 12 state/action dimensions")
    cameras = ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist")
    if set(dataset.meta.camera_keys) != set(cameras):
        raise ValueError(f"Expected top/left_wrist/right_wrist cameras; got {dataset.meta.camera_keys}")
    # Preserve the base model's semantic view order: overhead, left wrist, right wrist.
    policy.input_features = {key: features[key] for key in ("observation.state", *cameras)}
    policy.output_features = {
        key: value for key, value in features.items() if value.type is FeatureType.ACTION
    }
    if policy.max_state_dim < 12 or policy.max_action_dim < 12:
        raise ValueError("Checkpoint padding dimensions are too small for SO101")
    policy.action_feature_names = dataset.features["action"]["names"]
    policy.validate_features()
    policy.device = "cuda"
    policy.dtype = torch.bfloat16
    policy.push_to_hub = False
    policy.repo_id = None
    policy.use_amp = False
    policy.use_relative_actions = False
    policy.compile_model = False
    policy.use_compiled_text_ce = False
    policy.use_compiled_vision = False
    policy.use_flashrt_adarms = False
    policy.use_flex_attention = False
    policy.use_manual_attention = False
    policy.flow_num_repeats = 1
    policy.knowledge_insulation = True
    policy.flow_loss_weight = 10.0
    peft = configure_finetuning(policy, args)
    # The released checkpoint's Hub tokenizer name need not exist locally: use its actual artifacts.
    bundled_fast = checkpoint / "action_tokenizer"
    if bundled_fast.is_dir():
        policy.action_tokenizer_name = str(bundled_fast)
    policy.fast_tokenizer_fit_samples = 1024
    policy.fast_tokenizer_validation_samples = 256
    policy.fast_tokenizer_cache_dir = str(REPO_ROOT / "outputs/fineart_so101/fast_tokenizers")
    policy.tokenizer_max_length = max(policy.tokenizer_max_length, 256)
    policy.optimizer_lr = args.lr

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id,
            root=str(args.dataset_root),
            video_backend="pyav",
            use_imagenet_stats=False,
            eval_split=1.0 - args.train_ratio,
        ),
        policy=policy,
        output_dir=args.run_dir / "train",
        job_name=f"fineart_so101_full_{args.mode}",
        steps=args.steps or 1,  # Resolved from actual training frames below, before validation.
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        env_eval_freq=0,
        eval_steps=args.eval_freq if args.train_ratio < 1.0 else 0,
        max_eval_samples=args.max_eval_samples,
        peft=peft,
    )
    cfg.dataset.image_transforms.enable = False
    if cfg.dataset.eval_split:
        train_dataset, eval_dataset = make_train_eval_datasets(cfg)
    else:
        train_dataset, eval_dataset = dataset, None
    args.steps, args.steps_per_epoch = training_schedule(args, len(train_dataset))
    cfg.steps = args.steps
    policy.scheduler_decay_lr = args.min_lr
    policy.scheduler_decay_steps = args.steps
    policy.scheduler_warmup_steps = min(args.warmup_steps, args.steps - 1)
    configure_distributed(cfg, args)
    cfg.trackio.enable = False
    cfg.wandb_logging = WandBLoggingConfig(
        enable=True,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_name or f"fineart-so101-{args.run_dir.name}",
        run_id=uuid.uuid4().hex[:12],
        mode=args.wandb_mode,
        disable_artifact=True,
    )
    cfg.validate()
    return cfg, dataset, train_dataset, eval_dataset


def prepare_fast_tokenizer(cfg: TrainPipelineConfig, train_dataset: LeRobotDataset) -> None:
    """Prepare once in the parent, restricting fitted actions and cache identity to training episodes."""
    if not cfg.policy.enable_fast_action_loss:
        cfg.policy.auto_fit_fast_tokenizer = False
        return
    cfg.policy.action_tokenizer_name = resolve_fast_tokenizer(
        cfg.policy,
        cfg.dataset.repo_id,
        train_dataset.root,
        train_dataset.meta.stats,
        episodes=train_dataset.episodes,
    )
    cfg.policy.auto_fit_fast_tokenizer = False


def main() -> None:
    args = parse_args()
    for package, extra in (("datasets", "training"), ("av", "training"), ("transformers", "fineart_vla")):
        require_package(package, extra=extra)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This parent launcher creates workers; do not wrap it in torchrun/accelerate launch")
    visible_devices = visible_gpu_ids(args)
    checkpoint = checkpoint_directory(args.checkpoint)
    cfg, dataset, train_dataset, eval_dataset = build_config(args, checkpoint)
    print("Checking all subtask labels and decoding episode boundary frames...", flush=True)
    audit = inspect_dataset(dataset, TrainingRecipe.from_dict(cfg.policy.recipe))
    git_revision = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    audit.update(
        checkpoint=str(checkpoint),
        checkpoint_config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        source_data_sha256={
            str(path.relative_to(dataset.root)): file_sha256(path)
            for path in sorted((dataset.root / "data").rglob("*.parquet"))
        },
        git_revision=git_revision,
        git_worktree_dirty=bool(
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
        ),
        launcher_config_generator_sha256=file_sha256(Path(__file__)),
        dataset_metadata_sha256={
            name: file_sha256(dataset.root / "meta" / name) for name in ("info.json", "stats.json")
        },
        camera_input_order=list(cfg.policy.image_features),
        python=sys.executable,
        fast_skip_tokens=cfg.policy.fast_skip_tokens,
        distributed_backend="FSDP2" if cfg.parallelism.is_sharded else "DDP",
        dp_replicate=cfg.parallelism.dp_replicate,
        dp_shard=cfg.parallelism.dp_shard,
        fsdp_wrap_modules=cfg.accelerator.fsdp.wrap_modules if cfg.parallelism.is_sharded else None,
        fsdp_reshard_after_forward=(
            cfg.accelerator.fsdp.reshard_after_forward if cfg.parallelism.is_sharded else None
        ),
        fsdp_cpu_offload=cfg.accelerator.fsdp.cpu_offload if cfg.parallelism.is_sharded else None,
        checkpoint_format=cfg.checkpoint_format.value,
        num_gpus=args.num_gpus,
        visible_gpu_ids=visible_devices,
        batch_size_per_gpu=args.batch_size,
        global_batch_size=args.batch_size * args.num_gpus,
        requested_train_ratio=args.train_ratio,
        train_frames=len(train_dataset),
        eval_frames=len(eval_dataset) if eval_dataset is not None else 0,
        train_episode_indices=train_dataset.episodes or list(range(dataset.num_episodes)),
        eval_episode_indices=eval_dataset.episodes if eval_dataset is not None else [],
        actual_train_episode_ratio=train_dataset.num_episodes / dataset.num_episodes,
        actual_train_frame_ratio=len(train_dataset) / len(dataset),
        split_strategy="native per-task episode tail holdout",
        normalization_stats_source="source dataset meta/stats.json (native behavior)",
        requested_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        training_steps=args.steps,
        planned_epochs=args.steps / args.steps_per_epoch,
        approximate_frame_passes=args.steps * args.batch_size * args.num_gpus / len(train_dataset),
        learning_rate=cfg.policy.optimizer_lr,
        min_learning_rate=cfg.policy.scheduler_decay_lr,
        max_learning_rate=cfg.policy.optimizer_lr,
        final_learning_rate=cfg.policy.scheduler_decay_lr,
        warmup_steps=cfg.policy.scheduler_warmup_steps,
        gradient_checkpointing=cfg.policy.gradient_checkpointing,
        use_lora=cfg.policy.use_peft,
        lora_config=vars(cfg.peft) if cfg.peft is not None else None,
        freeze_vlm=cfg.policy.train_expert_only,
        freeze_vision_encoder=cfg.policy.freeze_vision_encoder,
        text_loss_weight=cfg.policy.text_loss_weight,
        fast_action_loss_enabled=cfg.policy.enable_fast_action_loss,
        fast_action_loss_weight=cfg.policy.fast_action_loss_weight,
        flow_loss_weight=cfg.policy.flow_loss_weight,
        save_freq=cfg.save_freq,
        eval_freq=cfg.eval_steps,
        max_eval_samples=cfg.max_eval_samples,
        logger_backend="wandb",
        wandb_project=cfg.wandb_logging.project,
        wandb_entity=cfg.wandb_logging.entity,
        job_name=cfg.job_name,
        wandb_run_name=cfg.wandb_logging.run_name,
        wandb_mode=cfg.wandb_logging.mode,
        wandb_disable_artifact=cfg.wandb_logging.disable_artifact,
        wandb_initialized=False,
        checks_not_performed=[
            "model weight loading",
            "FAST fitting",
            "GPU forward/backward",
            "robot evaluation",
        ],
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "train_config.json"
    with config_path.open("w") as stream, draccus.config_type("json"):
        draccus.dump(cfg, stream, indent=2)
    (args.run_dir / "preflight.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.num_gpus}",
        "--module",
        "lerobot.scripts.lerobot_train",
        f"--config_path={config_path}",
    ]
    print(json.dumps({key: value for key, value in audit.items() if key != "recipe_examples"}, indent=2))
    print(f"Config: {config_path}\nNative command: {shlex.join(command)}", flush=True)
    if args.mode == "check":
        print("CHECK ONLY: no weights loaded, no FAST tokenizer fitted, no training started.")
        return
    require_package("accelerate", extra="training")
    require_package("liger-kernel", extra="fineart_vla_kernels", import_name="liger_kernel")
    require_package("wandb", extra="wandb")
    if cfg.peft is not None:
        require_package("peft", extra="peft")
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.num_gpus:
        raise RuntimeError(f"Expected {args.num_gpus} visible CUDA GPUs; inspect GPU_IDS and driver access")
    for index in range(args.num_gpus):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"Visible GPU {index} does not support BF16")
    # Native processor construction runs on both ranks. Fit once in the parent to avoid
    # concurrent writes to the same FAST cache; workers only load the completed artifacts.
    prepare_fast_tokenizer(cfg, train_dataset)
    with config_path.open("w") as stream, draccus.config_type("json"):
        draccus.dump(cfg, stream, indent=2)
    audit["fast_tokenizer_prepared_in_parent"] = cfg.policy.enable_fast_action_loss
    audit["fast_tokenizer_path"] = cfg.policy.action_tokenizer_name
    if cfg.policy.enable_fast_action_loss:
        audit["checks_not_performed"].remove("FAST fitting")
    (args.run_dir / "preflight.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
