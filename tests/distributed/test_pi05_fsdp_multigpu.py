# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""PI05 FSDP2 regression: CUDA/Accelerate plus opt-in CPU/Gloo collectives.

CPU/Gloo validates model and checkpoint collectives on torch >= 2.11, but uses
torch's fully_shard directly; it does not substitute for CUDA/Accelerate.prepare.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from lerobot.common.train_utils import resume_after_prepare, save_checkpoint
from lerobot.configs.default import DatasetConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
from lerobot.distributed import finalize_sharded_policy, set_fsdp_wrap_modules
from lerobot.distributed.checkpoint import dcp_to_safetensors, full_model_state_dict
from lerobot.distributed.parallel_dims import ParallelDims
from lerobot.policies.pi05.fsdp import canonical_pi05_state_dict
from tests.distributed.test_multigpu_training import (
    _assert_tree_equal,
    _gather_full,
    _init_worker_env,
    _spawn,
)
from tests.fixtures.pi05_fsdp import deterministic_flow, pi05_batch, tiny_pi05


def _prepare(accelerator, cfg, *, checkpointing, device_type="cuda"):
    policy = tiny_pi05(checkpointing=checkpointing, device=accelerator.device.type)
    deterministic_flow(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    set_fsdp_wrap_modules(accelerator, policy)
    if device_type == "cuda":
        policy, optimizer = accelerator.prepare(policy, optimizer)
    else:
        # Accelerate's CPU distributed mode is DDP, so test torch FSDP directly.
        # The CUDA branch above covers production optimizer rebinding.
        kwargs = {
            "mesh": init_device_mesh("cpu", (2,)),
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16 if cfg.accelerator.mixed_precision == "bf16" else None,
                reduce_dtype=torch.float32,
            ),
        }
        core = policy.model.paligemma_with_expert
        for layer in core.joint_layers:
            fully_shard(layer, **kwargs)
        for layer in core.paligemma.model.vision_tower.vision_model.encoder.layers:
            fully_shard(layer, **kwargs)
        fully_shard(policy.get_input_embeddings(), **kwargs)
        fully_shard(policy, **kwargs)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    dims = ParallelDims.from_config(cfg.parallelism, 2, device_type)
    finalize_sharded_policy(policy, dims)
    assert isinstance(policy, FSDPModule)
    assert isinstance(policy.get_input_embeddings(), FSDPModule)
    core = policy.model.paligemma_with_expert
    assert all(isinstance(layer, FSDPModule) for layer in core.joint_layers)
    assert all(
        isinstance(layer, FSDPModule)
        for layer in core.paligemma.model.vision_tower.vision_model.encoder.layers
    )
    assert all(isinstance(param, DTensor) for param in policy.parameters())
    return policy, optimizer


def _step(accelerator, policy, optimizer, batch):
    policy.train()
    with accelerator.autocast():
        loss = policy(batch)[0]
    accelerator.backward(loss)
    # FSDP2 uses torch's DTensor-aware clipper. Accelerate 1.14's dispatcher
    # compares parameter lists with tensor equality when two models are prepared
    # on one Accelerator (only this continuation comparison does that).
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    assert torch.isfinite(loss)
    return loss.detach()


def _worker(rank, world_size, port, directory, precision, checkpointing, fmt_value, device_type="cuda"):
    _init_worker_env(rank, world_size, port)
    torch.set_num_threads(1)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/synthetic"),
        parallelism=ParallelismConfig(dp_shard=world_size),
        checkpoint_format=CheckpointFormat(fmt_value),
        batch_size=2,
    )
    cfg.parallelism.resolve(world_size)
    cfg.accelerator.mixed_precision = precision
    if device_type == "cuda":
        accelerator = cfg.accelerator.build(cfg.parallelism)
    else:
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True, mixed_precision=precision)
        # The checkpoint helpers consume the plugin's format settings only.
        accelerator.state.fsdp_plugin = cfg.accelerator.fsdp.build_plugin()
    torch.manual_seed(7)
    reference = tiny_pi05(checkpointing=checkpointing, device=accelerator.device.type)
    deterministic_flow(reference)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    torch.manual_seed(7)
    policy, optimizer = _prepare(accelerator, cfg, checkpointing=checkpointing, device_type=device_type)
    cfg.policy = policy.config

    for step in range(2):
        global_batch = pi05_batch(batch_size=4, seed=20 + step, device=accelerator.device)
        local_batch = {key: value[2 * rank : 2 * rank + 2] for key, value in global_batch.items()}
        reference.train()
        with accelerator.autocast():
            ref_loss = reference(global_batch)[0]
        ref_loss.backward()
        with accelerator.autocast():
            loss = policy(local_batch)[0]
        accelerator.backward(loss)
        mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
        if precision == "no":
            torch.testing.assert_close(mean_loss, ref_loss.detach(), atol=1e-5, rtol=1e-4)
            grads = canonical_pi05_state_dict(
                {
                    name: None if param.grad is None else param.grad.full_tensor()
                    for name, param in policy.named_parameters()
                }
            )
            for name, param in reference.named_parameters():
                torch.testing.assert_close(grads[name], param.grad, atol=1e-5, rtol=2e-3)
        else:
            # Module-level BF16 and AMP differ in casting placement; check useful
            # numerical agreement, while FP32 above checks gradients tightly.
            torch.testing.assert_close(mean_loss.float(), ref_loss.detach().float(), atol=1e-2, rtol=3e-2)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        accelerator.clip_grad_norm_(policy.parameters(), 1.0)
        reference_optimizer.step()
        optimizer.step()
        reference_optimizer.zero_grad()
        optimizer.zero_grad()

    full_state = full_model_state_dict(policy)
    if rank == 0 and precision == "no":
        torch.testing.assert_close(
            canonical_pi05_state_dict(full_state),
            {key: value.cpu() for key, value in reference.state_dict().items()},
            atol=2e-5,
            rtol=2e-3,
        )
    checkpoint_dir = Path(directory) / "checkpoint"
    save_checkpoint(checkpoint_dir, 2, cfg, policy, optimizer, accelerator=accelerator)
    expected_model, expected_optimizer = _gather_full(policy, optimizer)
    accelerator.wait_for_everyone()

    fresh, fresh_optimizer = _prepare(accelerator, cfg, checkpointing=checkpointing, device_type=device_type)
    if cfg.checkpoint_format is CheckpointFormat.SAFETENSORS:
        # Production loads full weights before prepare; use the DCP state-dict API
        # here so the second prepared model can reproduce that same state.
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        plain = tiny_pi05()
        plain.load_state_dict(load_file(checkpoint_dir / "pretrained_model/model.safetensors"), strict=True)
        plain.prepare_for_fsdp()
        set_model_state_dict(fresh, plain.state_dict(), options=StateDictOptions(full_state_dict=True))
    cfg.checkpoint_path = checkpoint_dir
    resume_after_prepare(cfg, accelerator, fresh, fresh_optimizer, None)
    actual_model, actual_optimizer = _gather_full(fresh, fresh_optimizer)
    if rank == 0:
        _assert_tree_equal(expected_model, actual_model, "model")
        _assert_tree_equal(expected_optimizer, actual_optimizer, "optimizer")
        exported = load_file(checkpoint_dir / "pretrained_model/model.safetensors")
        plain = tiny_pi05()
        plain.load_state_dict(exported, strict=True)
        torch.testing.assert_close(exported, canonical_pi05_state_dict(expected_model))
        assert torch.isfinite(plain.predict_action_chunk(pi05_batch())).all()
    accelerator.wait_for_everyone()

    batch = pi05_batch(seed=40 + rank, device=accelerator.device)
    before = _step(accelerator, policy, optimizer, batch)
    after = _step(accelerator, fresh, fresh_optimizer, batch)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    before_model, before_optimizer = _gather_full(policy, optimizer)
    after_model, after_optimizer = _gather_full(fresh, fresh_optimizer)
    if rank == 0:
        _assert_tree_equal(before_model, after_model, "continued model")
        _assert_tree_equal(before_optimizer, after_optimizer, "continued optimizer")
    # Exercise registered non-forward entry points on ALL ranks too.
    with accelerator.autocast():
        assert torch.isfinite(fresh.predict_action_chunk(batch)).all()
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize(
    "precision,checkpointing,fmt",
    [
        ("no", False, "safetensors"),
        ("no", True, "safetensors_dcp"),
        ("bf16", True, "safetensors_dcp"),
    ],
)
def test_pi05_fsdp_training_export_and_resume(tmp_path, precision, checkpointing, fmt):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16-capable GPUs")
    _spawn(2, _worker, str(tmp_path), precision, checkpointing, fmt)
    if CheckpointFormat(fmt).wants_dcp:
        # Exercise the offline converter and ordinary from_pretrained loader after
        # all workers exit: merging retains runtime names, unlike direct export.
        pretrained = tmp_path / "checkpoint/pretrained_model"
        expected = load_file(pretrained / "model.safetensors")
        dcp_to_safetensors(pretrained / "pytorch_model_fsdp_0", pretrained)
        loaded = tiny_pi05(pretrained_path=pretrained)
        torch.testing.assert_close(loaded.state_dict(), expected)


@pytest.mark.skipif(
    os.environ.get("LEROBOT_TEST_FSDP_CPU") != "1"
    or Version(torch.__version__.split("+")[0]) < Version("2.11"),
    reason="opt-in Gloo test: set LEROBOT_TEST_FSDP_CPU=1 with torch >= 2.11",
)
@pytest.mark.parametrize(
    "precision,checkpointing,fmt",
    [
        ("no", False, "safetensors"),
        ("no", True, "safetensors_dcp"),
        ("bf16", True, "safetensors_dcp"),
    ],
)
def test_pi05_fsdp_gloo_training_export_and_resume(tmp_path, precision, checkpointing, fmt):
    _spawn(2, _worker, str(tmp_path), precision, checkpointing, fmt, "cpu")
