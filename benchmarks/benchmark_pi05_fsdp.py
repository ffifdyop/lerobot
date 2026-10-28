# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Compare full-size PI05 DDP/FSDP memory and compute throughput on synthetic batches.

No model download, dataset access, W&B run, or training checkpoint is created.
Run each layout in a fresh torchrun process; use the same visible GPUs and batch.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from lerobot.configs.accelerator import DDPCommunicationHook
from lerobot.configs.default import DatasetConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.distributed import finalize_sharded_policy, make_accelerator, set_fsdp_wrap_modules
from lerobot.distributed.parallel_dims import ParallelDims
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-shard", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6, help="per GPU")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.batch_size < 1:
        parser.error("steps, warmup and batch-size must be positive")
    cameras = [f"observation.images.camera_{index}" for index in range(3)]
    policy_config = PI05Config(
        device="cuda",
        dtype="float32",
        gradient_checkpointing=True,
        train_on_padded_action_dims=True,
        push_to_hub=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for key in cameras},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="benchmark/synthetic"),
        policy=policy_config,
        parallelism=ParallelismConfig(dp_shard=args.dp_shard),
        batch_size=args.batch_size,
    )
    cfg.accelerator.mixed_precision = "bf16"
    cfg.accelerator.ddp.comm_hook = DDPCommunicationHook.BF16
    cfg._validate_distributed()
    accelerator = make_accelerator(cfg)
    print(f"rank {accelerator.process_index}: building PI05 on {torch.cuda.get_device_name()}", flush=True)
    torch.manual_seed(42)
    policy = PI05Policy(policy_config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=4e-5, betas=(0.9, 0.95), weight_decay=0.01)
    set_fsdp_wrap_modules(accelerator, policy)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    finalize_sharded_policy(
        policy, ParallelDims.from_config(cfg.parallelism, accelerator.num_processes, "cuda")
    )
    if accelerator.is_main_process:
        print("Model prepared; starting measured training recipe", flush=True)
    init_peak = torch.cuda.max_memory_allocated()
    device = accelerator.device
    batch = {
        OBS_STATE: torch.zeros(args.batch_size, 14, device=device),
        **{key: torch.rand(args.batch_size, 3, 224, 224, device=device) for key in cameras},
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (args.batch_size, 200), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(args.batch_size, 200, dtype=torch.bool, device=device),
        ACTION: torch.randn(args.batch_size, 50, 14, device=device),
    }
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    elapsed = 0.0
    for step in range(args.warmup + args.steps):
        accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with accelerator.autocast():
            loss = policy(batch)[0]
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        if step >= args.warmup:
            elapsed += time.perf_counter() - start
        if accelerator.is_main_process:
            print(f"step {step + 1}/{args.warmup + args.steps}, loss={loss.item():.5f}", flush=True)
    metrics = (
        accelerator.gather(
            torch.tensor(
                [
                    torch.cuda.max_memory_allocated() / 2**30,
                    torch.cuda.max_memory_reserved() / 2**30,
                    init_peak / 2**30,
                    elapsed / args.steps,
                ],
                device=device,
            ).unsqueeze(0)
        )
        .cpu()
        .tolist()
    )
    if accelerator.is_main_process:
        seconds = max(row[3] for row in metrics)
        result = {
            "layout": "fsdp" if cfg.parallelism.is_sharded else "ddp",
            "dp_shard": args.dp_shard,
            "world_size": accelerator.num_processes,
            "per_gpu_batch": args.batch_size,
            "global_batch": args.batch_size * accelerator.num_processes,
            "cameras": 3,
            "image_size": 224,
            "language_tokens": 200,
            "action_horizon": 50,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "precision": "bf16",
            "gradient_checkpointing": True,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "peak_allocated_gib_per_rank": [row[0] for row in metrics],
            "peak_reserved_gib_per_rank": [row[1] for row in metrics],
            "init_peak_allocated_gib_per_rank": [row[2] for row in metrics],
            "seconds_per_step": seconds,
            "samples_per_second": args.batch_size * accelerator.num_processes / seconds,
            "scope": "random full-size PI05, synthetic resident batches; excludes dataset I/O and checkpointing",
        }
        payload = json.dumps(result, indent=2)
        print(payload)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n")
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
