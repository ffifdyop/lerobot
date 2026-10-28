# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""CPU contracts for PI05's runtime FSDP tree; CUDA collectives live in distributed tests."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("transformers")

from lerobot.configs.accelerator import FSDPConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed.factory import set_fsdp_wrap_modules
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.fsdp import canonical_pi05_state_dict
from tests.fixtures.pi05_fsdp import deterministic_flow, pi05_batch, tiny_pi05


def _prepare(policy):
    plugin = FSDPConfig().build_plugin()
    set_fsdp_wrap_modules(SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin)), policy)
    return plugin


@pytest.fixture(autouse=True)
def _single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_joint_tree_matches_loss_gradients_and_optimizer_update(checkpointing):
    torch.manual_seed(1)
    reference = tiny_pi05(checkpointing=checkpointing)
    candidate = tiny_pi05(checkpointing=checkpointing)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    original_ids = {id(param) for param in candidate.parameters()}
    original_names = set(candidate.state_dict())
    _prepare(candidate)
    _prepare(candidate)  # prepare is idempotent
    assert original_ids == {id(param) for param in candidate.parameters()}
    assert original_ids == {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert len(list(candidate.named_parameters(remove_duplicate=False))) == len(original_ids)
    assert set(canonical_pi05_state_dict(candidate.state_dict())) == original_names

    core = candidate.model.paligemma_with_expert
    calls = [0] * len(core.joint_layers)
    handles = []
    for index, layer in enumerate(core.joint_layers):

        def hook(module, args, index=index):
            calls[index] += 1

        handles.append(layer.register_forward_pre_hook(hook))

    reference.train()
    candidate.train()
    deterministic_flow(reference)
    deterministic_flow(candidate)
    batch = pi05_batch()
    loss_ref = reference(batch)[0]
    loss = candidate(batch)[0]
    torch.testing.assert_close(loss, loss_ref, atol=1e-7, rtol=1e-6)
    loss_ref.backward()
    loss.backward()
    # Checkpoint runs inside the joint/FSDP forward boundary, after input casting.
    assert all(count == 1 for count in calls)

    grads = canonical_pi05_state_dict({name: param.grad for name, param in candidate.named_parameters()})
    for name, param in reference.named_parameters():
        torch.testing.assert_close(grads[name], param.grad, atol=1e-6, rtol=1e-5)
    torch.optim.AdamW(reference.parameters(), lr=1e-3).step()
    optimizer.step()
    torch.testing.assert_close(canonical_pi05_state_dict(candidate.state_dict()), reference.state_dict())
    for handle in handles:
        handle.remove()


def test_cached_inference_enters_joint_units_and_matches_plain_model():
    torch.manual_seed(2)
    reference = tiny_pi05()
    candidate = tiny_pi05()
    candidate.load_state_dict(reference.state_dict())
    _prepare(candidate)
    branches = []
    for layer in candidate.model.paligemma_with_expert.joint_layers:
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: branches.append(kwargs.get("branch")), with_kwargs=True
        )
    batch = pi05_batch()
    torch.manual_seed(3)
    expected = reference.predict_action_chunk(batch)
    torch.manual_seed(3)
    actual = candidate.predict_action_chunk(batch)
    torch.testing.assert_close(actual, expected)
    assert branches.count("vlm") == 2
    assert branches.count("expert") == 4  # two layers, two denoising steps


def test_export_and_merged_dcp_keys_load_in_plain_pi05(tmp_path):
    model = tiny_pi05()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    _prepare(model)
    model.save_pretrained(tmp_path)
    exported = load_file(tmp_path / "model.safetensors")
    torch.testing.assert_close(exported, expected)
    plain = tiny_pi05()
    plain.load_state_dict(plain._prepare_pretrained_state_dict(exported), strict=True)
    plain.load_state_dict(plain._prepare_pretrained_state_dict(model.state_dict()), strict=True)
    torch.testing.assert_close(plain.state_dict(), expected)
    # Exercise the actual entry point, including its legacy lm_head key handling.
    loaded = tiny_pi05(pretrained_path=tmp_path)
    torch.testing.assert_close(loaded.state_dict(), expected)


def test_export_trims_padded_fsdp_storage():
    model = tiny_pi05()
    padded = torch.randn(2, 1, 32)
    probe = padded[:1]
    assert probe.is_contiguous()
    exported = model._prepare_state_dict_for_save({"probe": probe})["probe"]
    torch.testing.assert_close(exported, probe)
    assert exported.untyped_storage().nbytes() == exported.numel() * exported.element_size()


def test_checkpoint_key_collision_is_rejected():
    tensor = torch.zeros(1)
    with pytest.raises(ValueError, match="Duplicate PI05"):
        canonical_pi05_state_dict(
            {
                "model.paligemma_with_expert.joint_layers.0.vlm.mlp.weight": tensor,
                "model.paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.weight": tensor,
            }
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("compile_model", True, "compile_model"),
        ("use_visual_memory", True, "visual memory"),
        ("freeze_vision_encoder", True, "full fine-tuning"),
        ("train_expert_only", True, "full fine-tuning"),
        ("dtype", "bfloat16", "dtype=float32"),
    ],
)
def test_unsupported_fsdp_configs_fail_before_allocation(field, value, message):
    policy = PI05Config(**{field: value})
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/dummy"), policy=policy)
    cfg._validate_distributed()  # these combinations remain legal for non-sharded runs
    cfg.parallelism = ParallelismConfig(dp_shard=2)
    with pytest.raises(ValueError, match=message):
        cfg._validate_distributed()


@pytest.mark.parametrize("field,value", [("wrap_modules", ["Linear"]), ("min_num_params", 1)])
def test_custom_wrap_policy_rejected(field, value):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/dummy"),
        policy=PI05Config(),
        parallelism=ParallelismConfig(dp_shard=2),
    )
    setattr(cfg.accelerator.fsdp, field, value)
    with pytest.raises(ValueError, match="policy-defined"):
        cfg._validate_distributed()


def test_non_sharded_prepare_leaves_model_and_lora_paths_unchanged():
    model = tiny_pi05()
    names = tuple(model.state_dict())
    set_fsdp_wrap_modules(SimpleNamespace(state=SimpleNamespace(fsdp_plugin=None)), model)
    assert tuple(model.state_dict()) == names
    assert not hasattr(model.model.paligemma_with_expert, "joint_layers")
