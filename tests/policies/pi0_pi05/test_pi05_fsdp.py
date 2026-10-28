"""PI05's upstream model with an explicit coarse FSDP wrapping recipe."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.distributed import set_fsdp_wrap_modules
from tests.distributed.test_pi05_checkpoint_migration import _legacy_key
from tests.fixtures.pi05_fsdp import deterministic_flow, pi05_batch, tiny_pi05


def test_pi05_requires_explicit_wrap_units():
    policy = tiny_pi05()
    plugin = SimpleNamespace(min_num_params=None, transformer_cls_names_to_wrap=None)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    with pytest.raises(ValueError, match="declares no FSDP wrap units"):
        set_fsdp_wrap_modules(accelerator, policy)
    plugin.transformer_cls_names_to_wrap = ["SiglipEncoderLayer"]
    set_fsdp_wrap_modules(accelerator, policy)
    assert plugin.transformer_cls_names_to_wrap == ["SiglipEncoderLayer"]
    assert not hasattr(policy, "prepare_for_fsdp")
    assert not hasattr(policy.model.paligemma_with_expert, "joint_layers")


def test_upstream_pi05_checkpointed_gradients_match_plain():
    torch.set_num_threads(1)
    outputs = []
    for checkpointing in (False, True):
        torch.manual_seed(7)
        policy = tiny_pi05(checkpointing=checkpointing)
        deterministic_flow(policy)
        loss = policy(pi05_batch())[0]
        loss.backward()
        outputs.append((loss.detach(), {name: p.grad for name, p in policy.named_parameters()}))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("joint_names", [False, True])
def test_historical_pi05_safetensors_load_without_parameter_loss(tmp_path, joint_names):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    policy = tiny_pi05()
    weights = {}
    for name, tensor in policy.state_dict().items():
        saved_name = (
            _legacy_key(name)
            if joint_names
            else name.replace("paligemma.model.vision_tower.", "paligemma.model.vision_tower.vision_model.")
        )
        weights[saved_name] = tensor
    save_file(weights, tmp_path / "model.safetensors")
    loaded = tiny_pi05(pretrained_path=tmp_path)
    torch.testing.assert_close(loaded.state_dict(), policy.state_dict(), rtol=0, atol=0)
    assert torch.isfinite(loaded.predict_action_chunk(pi05_batch())).all()
