"""Strict migration of old PI05 keys and intentionally lazy optimizer state."""

import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from lerobot.distributed.checkpoint import load_sharded_model, load_sharded_optimizer
from lerobot.utils.checkpoint_compat import canonical_pi05_key, canonical_pi05_state_dict
from tests.fixtures.pi05_fsdp import deterministic_flow, pi05_batch, tiny_pi05


def _accelerator():
    return SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(state_dict_config=None)),
        wait_for_everyone=lambda: None,
    )


def _legacy_key(name):
    for owner, branch in (("paligemma.model.language_model", "vlm"), ("gemma_expert.model", "expert")):
        name = re.sub(
            rf"(.*paligemma_with_expert\.){re.escape(owner)}\.layers\.(\d+)\.(.+)",
            rf"\1joint_layers.\2.{branch}.\3",
            name,
        )
    return name.replace("paligemma.model.vision_tower.", "paligemma.model.vision_tower.vision_model.")


def _step(model, optimizer):
    model(pi05_batch())[0].backward()
    optimizer.step()
    optimizer.zero_grad()


def test_pi05_alias_collisions_fail():
    old = "model.paligemma_with_expert.joint_layers.0.vlm.self_attn.q_proj.weight"
    current = canonical_pi05_key(old)
    assert (
        current
        == "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
    )
    with pytest.raises(ValueError, match="Duplicate PI05 checkpoint key"):
        canonical_pi05_state_dict({old: torch.zeros(1), current: torch.ones(1)})


@pytest.mark.parametrize("unused", [False, True])
def test_optimizer_lazy_state_round_trip(tmp_path, unused):
    model = torch.nn.ModuleDict({"used": torch.nn.Linear(2, 2), "other": torch.nn.Linear(2, 2)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    value = model["used"](torch.ones(2, 2))
    if not unused:
        value = model["other"](value)
    value.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved = get_optimizer_state_dict(model, optimizer)
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path / "optimizer_0")
    fresh = torch.optim.AdamW(model.parameters(), lr=1.0)
    load_sharded_optimizer(_accelerator(), fresh, model, tmp_path)
    torch.testing.assert_close(fresh.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
    assert len(fresh.state) == (2 if unused else 4)


def test_optimizer_missing_moment_is_not_treated_as_lazy(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved = get_optimizer_state_dict(model, optimizer)
    del saved["state"]["weight"]["exp_avg"]
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path / "optimizer_0")
    with pytest.raises(dcp.CheckpointException, match="Missing key.*exp_avg"):
        load_sharded_optimizer(_accelerator(), torch.optim.AdamW(model.parameters()), model, tmp_path)


def test_optimizer_unknown_parameter_is_not_dropped(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved = get_optimizer_state_dict(model, optimizer)
    saved["state"]["unknown"] = saved["state"]["weight"]
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path / "optimizer_0")
    with pytest.raises(ValueError, match="unknown parameter states"):
        load_sharded_optimizer(_accelerator(), torch.optim.AdamW(model.parameters()), model, tmp_path)


def test_optimizer_parameter_group_mismatch_is_rejected(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved = get_optimizer_state_dict(model, optimizer)
    saved["param_groups"][0]["params"].append("unknown")
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path / "optimizer_0")
    with pytest.raises(ValueError, match="parameter groups do not match"):
        load_sharded_optimizer(_accelerator(), torch.optim.AdamW(model.parameters()), model, tmp_path)


def test_legacy_pi05_model_optimizer_and_next_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    reference = tiny_pi05(checkpointing=True)
    deterministic_flow(reference)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    _step(reference, optimizer)
    legacy_weights = {_legacy_key(name): tensor for name, tensor in reference.state_dict().items()}
    legacy_optimizer = get_optimizer_state_dict(reference, optimizer)
    legacy_optimizer["state"] = {
        _legacy_key(name): state for name, state in legacy_optimizer["state"].items()
    }
    for group in legacy_optimizer["param_groups"]:
        group["params"] = [_legacy_key(name) for name in group["params"]]
    dcp.save({"model": legacy_weights}, checkpoint_id=tmp_path / "pytorch_model_fsdp_0")
    dcp.save({"optimizer": legacy_optimizer}, checkpoint_id=tmp_path / "optimizer_0")

    fresh = tiny_pi05(checkpointing=True)
    deterministic_flow(fresh)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1.0)
    load_sharded_model(_accelerator(), fresh, tmp_path)
    load_sharded_optimizer(_accelerator(), fresh_optimizer, fresh, tmp_path)
    torch.testing.assert_close(fresh.state_dict(), reference.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(fresh_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
    _step(reference, optimizer)
    _step(fresh, fresh_optimizer)
    torch.testing.assert_close(fresh.state_dict(), reference.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(fresh_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)


def test_legacy_model_missing_parameter_fails(tmp_path):
    model = tiny_pi05()
    legacy_weights = {_legacy_key(name): tensor for name, tensor in model.state_dict().items()}
    del legacy_weights["model.action_in_proj.weight"]
    dcp.save({"model": legacy_weights}, checkpoint_id=tmp_path / "pytorch_model_fsdp_0")
    with pytest.raises(ValueError, match="model keys differ.*action_in_proj"):
        load_sharded_model(_accelerator(), model, tmp_path)
