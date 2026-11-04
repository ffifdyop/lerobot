from pathlib import Path
import argparse

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from tests.fixtures.pi05_fsdp import deterministic_flow, pi05_batch, tiny_pi05

torch.set_num_threads(1)
parser = argparse.ArgumentParser(description="Generate a synthetic old PI05 checkpoint with the b1a75228 source and transformers 5.5.4 environment.")
parser.add_argument("--output", type=Path, default=Path("legacy_fixture"))
args = parser.parse_args()
torch.manual_seed(27)
model = tiny_pi05(checkpointing=True)
deterministic_flow(model)
model.prepare_for_fsdp()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model(pi05_batch())[0].backward()
optimizer.step()
optimizer.zero_grad()
directory = args.output
directory.mkdir(exist_ok=True)
weights = model.state_dict()
state = get_optimizer_state_dict(model, optimizer)
dcp.save({"model": weights}, checkpoint_id=directory / "pytorch_model_fsdp_0")
dcp.save({"optimizer": state}, checkpoint_id=directory / "optimizer_0")
save_file(weights, directory / "legacy_joint.safetensors")
save_file(model._prepare_state_dict_for_save(weights), directory / "legacy_export.safetensors")
torch.save(state, directory / "optimizer_expected.pt")
print(f"Saved {len(weights)} historical model keys and {len(state['state'])} initialized optimizer parameters")
