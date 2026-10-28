#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The DCP wrappers must hand accelerate exact shard directories.

accelerate 1.14 resolves the load directory with a substring check ("optimizer" /
"pytorch_model_fsdp" in the path -> use as-is) while the save side joins the shard name
unconditionally, so a run path like `--job_name=optimizer_sweep` would save to
`training_state/optimizer_0/` but load from `training_state/` itself. Passing the exact
shard dir makes the containment check deterministically a no-op.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")


def fake_accelerator() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(state_dict_config=None)),
        wait_for_everyone=lambda: None,
    )


# A parent path that trips both of accelerate's substring checks at once.
POISONED_PARENT = Path("/outputs/train/optimizer_sweep_pytorch_model_fsdp_repro/training_state")


def test_load_sharded_optimizer_passes_exact_shard_dir(monkeypatch):
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint.state_dict as dcp_state

    from lerobot.distributed.checkpoint import load_sharded_optimizer

    seen = {}
    metadata = SimpleNamespace(
        planner_data={"optimizer.state.used.step": ("optimizer", "state", "used", "step")}
    )
    reader = SimpleNamespace(read_metadata=lambda: metadata)

    def make_reader(path):
        seen["path"] = path
        return reader

    monkeypatch.setattr(dcp, "FileSystemReader", make_reader)
    monkeypatch.setattr(
        dcp_state,
        "get_optimizer_state_dict",
        lambda *args, **kwargs: {
            "state": {"used": {"step": 0, "exp_avg": 0}, "unused": {"step": 0}},
            "param_groups": [],
        },
    )
    monkeypatch.setattr(dcp, "load", lambda state, **kwargs: seen.update(state=state))
    monkeypatch.setattr(dcp_state, "set_optimizer_state_dict", lambda *args, **kwargs: None)
    load_sharded_optimizer(fake_accelerator(), optimizer=object(), model=object(), input_dir=POISONED_PARENT)
    assert seen["path"] == str(POISONED_PARENT / "optimizer_0")
    assert isinstance(seen["path"], str)  # str, never Path (accelerate does string checks)
    assert seen["state"]["optimizer"]["state"] == {"used": {"step": 0, "exp_avg": 0}}


def test_load_sharded_model_passes_exact_shard_dir(monkeypatch):
    import accelerate.utils

    from lerobot.distributed.checkpoint import load_sharded_model

    seen = {}
    monkeypatch.setattr(
        accelerate.utils,
        "load_fsdp_model",
        lambda plugin, accelerator, model, input_dir: seen.update(path=input_dir),
    )
    load_sharded_model(fake_accelerator(), model=object(), input_dir=POISONED_PARENT)
    assert seen["path"] == str(POISONED_PARENT / "pytorch_model_fsdp_0")
    assert isinstance(seen["path"], str)


@pytest.mark.parametrize("unused_parameter", [False, True])
def test_optimizer_dcp_round_trip_preserves_lazy_state(tmp_path, unused_parameter):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    from lerobot.distributed.checkpoint import load_sharded_optimizer

    model = torch.nn.ModuleDict({"used": torch.nn.Linear(2, 2), "other": torch.nn.Linear(2, 2)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    output = model["used"](torch.ones(2, 2))
    if not unused_parameter:
        output = model["other"](output)
    output.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved = get_optimizer_state_dict(model, optimizer)
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path / "optimizer_0")

    fresh = torch.optim.AdamW(model.parameters(), lr=1.0)
    load_sharded_optimizer(fake_accelerator(), fresh, model, tmp_path)
    torch.testing.assert_close(fresh.state_dict(), optimizer.state_dict())
    assert len(fresh.state) == (2 if unused_parameter else 4)


def test_optimizer_dcp_rejects_missing_moment(tmp_path):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    from lerobot.distributed.checkpoint import load_sharded_optimizer

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved = get_optimizer_state_dict(model, optimizer)
    del saved["state"]["weight"]["exp_avg"]
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path / "optimizer_0")
    fresh = torch.optim.AdamW(model.parameters())
    with pytest.raises(dcp.CheckpointException, match="Missing key.*exp_avg"):
        load_sharded_optimizer(fake_accelerator(), fresh, model, tmp_path)
