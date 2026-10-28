# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Small real PI05 workload for external torchrun signal tests; no dataset or Hub access."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from lerobot.configs.default import DatasetConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed import finalize_sharded_policy, make_accelerator, set_fsdp_wrap_modules
from lerobot.distributed.parallel_dims import ParallelDims
from lerobot.distributed.shutdown import TrainingInterrupted, TrainingShutdown, init_training_data_worker
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from tests.fixtures.pi05_fsdp import pi05_batch, tiny_pi05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("ddp", "fsdp"), required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/synthetic"),
        policy=PI05Config(device="cuda", push_to_hub=False),
        parallelism=ParallelismConfig(dp_shard=2 if args.layout == "fsdp" else 1),
    )
    cfg.accelerator.mixed_precision = "bf16"
    cfg._validate_distributed()
    step = 0
    try:
        with TrainingShutdown() as shutdown:
            accelerator = make_accelerator(cfg)
            shutdown.accelerator = accelerator
            shutdown.raise_if_requested(accelerator.device)
            torch.manual_seed(42)
            policy = tiny_pi05(checkpointing=True, device="cuda")
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
            set_fsdp_wrap_modules(accelerator, policy)
            policy, optimizer = accelerator.prepare(policy, optimizer)
            finalize_sharded_policy(policy, ParallelDims.from_config(cfg.parallelism, 2, "cuda"))
            batch = pi05_batch(seed=42 + rank, device=accelerator.device)
            # Real child data workers share the trainer's process group and receive
            # torchrun's forwarded SIGINT too. Serving must survive that signal.
            loader = torch.utils.data.DataLoader(
                torch.arange(20000),
                batch_size=2,
                num_workers=1,
                multiprocessing_context="spawn",
                worker_init_fn=init_training_data_worker,
            )
            batches = iter(loader)
            shutdown.raise_if_requested(accelerator.device)
            for _ in range(10000):
                next(batches)
                with accelerator.autocast():
                    loss = policy(batch)[0]
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step == 2:
                    # This proves initialization, forward, backward and optimizer
                    # updates succeeded before the parent injects a signal.
                    (args.output / f"ready-{rank}.json").write_text(json.dumps({"pid": os.getpid()}))
                shutdown.raise_if_requested(accelerator.device)
            raise AssertionError("Parent did not interrupt the workload")
    except TrainingInterrupted as exc:
        (args.output / f"stopped-{rank}.json").write_text(
            json.dumps(
                {
                    "signum": exc.signum,
                    "step": step,
                    "group_initialized": dist.is_initialized(),
                }
            )
        )
        raise SystemExit(128 + exc.signum) from None


if __name__ == "__main__":
    main()
