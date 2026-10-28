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
"""Sharding-aware checkpoint primitives.

Two artifact channels with distinct owners:

- the **distributable** ``model.safetensors``: produced by ``PreTrainedPolicy.save_pretrained``
  through :func:`full_model_state_dict` — a collective full gather when the model is sharded;
- the **resume** channel (sharded runs): torch DCP directories written through accelerate
  (``pytorch_model_fsdp_0/`` and ``optimizer_0/``, names imported from accelerate constants).
  Model loading uses accelerate; optimizer loading uses torch DCP directly to preserve
  lazy optimizer state for unused parameters. Both reshard on load across topology changes.

Every function that touches sharded state is a collective and must run on ALL ranks.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator


def is_sharded_module(module: nn.Module) -> bool:
    """True when `fully_shard` owns this module's parameters (FSDP2's in-place class swap).

    Args:
        module (nn.Module): The module to inspect (a torch.compile wrapper is looked through
            via `_orig_mod`).

    Returns:
        bool: True when the module (or its compiled `_orig_mod`) is an `FSDPModule`.
    """
    from torch.distributed.fsdp import FSDPModule

    if isinstance(module, FSDPModule):
        return True
    # torch.compile wraps the sharded module; mirror accelerate's `_orig_mod` check.
    orig_mod = getattr(module, "_orig_mod", None)
    return orig_mod is not None and isinstance(orig_mod, FSDPModule)


def full_model_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """The module's full (unsharded) state dict, however its parameters are laid out.

    Sharded modules gather through torch's DCP state-dict API: a COLLECTIVE that must run on
    every rank; with ``cpu_offload=True`` the full dict materializes on the main rank only and
    every other rank receives a literal ``{}`` (runtime-verified — a
    rank-0-gated call deadlocks). Plain modules return ``module.state_dict()`` on every rank.

    Args:
        module (nn.Module): The (possibly sharded) module to read the state dict from.

    Returns:
        dict[str, torch.Tensor]: The full state dict — on the main rank only (``{}``
            elsewhere) when the module is sharded, on every rank otherwise.
    """
    if not is_sharded_module(module):
        return module.state_dict()

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def _fsdp_plugin(accelerator: "Accelerator") -> object:
    """The accelerator's FSDP plugin, required by every DCP save/load helper below.

    Args:
        accelerator (Accelerator): The accelerator that prepared the sharded model.

    Returns:
        object: The FSDP plugin held by `accelerator.state`.

    Raises:
        RuntimeError: If the accelerator was not configured with an FSDP plugin.
    """
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None:
        raise RuntimeError("Sharded checkpointing requires an FSDP-prepared Accelerator.")
    return plugin


def save_sharded_model(accelerator: "Accelerator", model: nn.Module, output_dir: Path) -> None:
    """Write the DCP model shards (`pytorch_model_fsdp_0/`). Collective: call on all ranks.

    Args:
        accelerator (Accelerator): The accelerator that prepared the sharded model.
        model (nn.Module): The prepared (sharded) model to save.
        output_dir (Path): The directory the shard subdirectory is created in.
    """
    from accelerate.utils import save_fsdp_model

    # accelerate 1.14's DCP helpers do string containment checks on the path:
    # always hand them str, never Path.
    save_fsdp_model(_fsdp_plugin(accelerator), accelerator, model, str(output_dir))


def load_sharded_model(accelerator: "Accelerator", model: nn.Module, input_dir: Path) -> None:
    """Load DCP model shards into the prepared (sharded) model. Collective: call on all ranks.

    Args:
        accelerator (Accelerator): The accelerator that prepared the sharded model.
        model (nn.Module): The prepared (sharded) model to load into.
        input_dir (Path): The directory containing the `pytorch_model_fsdp_0/` shard
            subdirectory.
    """
    from accelerate.utils import load_fsdp_model
    from accelerate.utils.constants import FSDP_MODEL_NAME

    # Pass the exact shard directory: accelerate's load resolves it with a substring check
    # ("pytorch_model_fsdp" in the path -> use as-is), which misfires on run paths that happen
    # to contain the marker; the exact dir makes the check deterministic.
    load_fsdp_model(_fsdp_plugin(accelerator), accelerator, model, str(input_dir / f"{FSDP_MODEL_NAME}_0"))


def save_sharded_optimizer(
    accelerator: "Accelerator", optimizer: torch.optim.Optimizer, model: nn.Module, output_dir: Path
) -> None:
    """Write the DCP optimizer shards (`optimizer_0/`). Collective: call on all ranks.

    Args:
        accelerator (Accelerator): The accelerator that prepared the model and optimizer.
        optimizer (torch.optim.Optimizer): The prepared optimizer to save the state from.
        model (nn.Module): The prepared (sharded) model the optimizer state is keyed by.
        output_dir (Path): The directory the shard subdirectory is created in.
    """
    from accelerate.utils import save_fsdp_optimizer

    save_fsdp_optimizer(_fsdp_plugin(accelerator), accelerator, optimizer, model, str(output_dir))


def load_sharded_optimizer(
    accelerator: "Accelerator", optimizer: torch.optim.Optimizer, model: nn.Module, input_dir: Path
) -> None:
    """Load DCP optimizer shards into the prepared optimizer. Collective: call on all ranks.

    Must run AFTER ``accelerator.prepare()``: FSDP2's prepare rebinds the optimizer's param
    groups to sharded DTensors but never migrates ``optimizer.state`` — the resharding load is
    the only correct way to restore it.

    Args:
        accelerator (Accelerator): The accelerator that prepared the model and optimizer.
        optimizer (torch.optim.Optimizer): The prepared optimizer to restore the state into.
        model (nn.Module): The prepared (sharded) model the optimizer state is keyed by.
        input_dir (Path): The directory containing the `optimizer_0/` shard subdirectory.
    """
    from accelerate.utils.constants import OPTIMIZER_NAME
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    plugin = _fsdp_plugin(accelerator)
    accelerator.wait_for_everyone()
    reader = FileSystemReader(str(input_dir / f"{OPTIMIZER_NAME}_0"))
    metadata = reader.read_metadata()
    if metadata.planner_data is None:
        raise ValueError("Optimizer DCP checkpoint lacks the nested state metadata required for loading.")
    # Adam initializes state lazily: PI05's unused LM heads (among other parameters)
    # never acquire moments. get_optimizer_state_dict on a fresh optimizer instead
    # initializes ALL parameters. Accelerate's stock loader then requires nonexistent
    # moment keys. Use the saved parameter-state membership, retaining strict loading
    # of every field for parameters that DO have state (a missing moment still fails).
    saved_parameters = {
        path[2]
        for path in metadata.planner_data.values()
        if len(path) >= 3 and tuple(path[:2]) == ("optimizer", "state")
    }
    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=getattr(getattr(plugin, "state_dict_config", None), "offload_to_cpu", False),
        # Missing WHOLE parameter states are intentional lazy Adam state. DCP's
        # load planner below remains strict about every requested tensor/field.
        strict=False,
    )
    state_dict = get_optimizer_state_dict(model, optimizer, options=options)
    state_dict["state"] = {
        name: state for name, state in state_dict["state"].items() if name in saved_parameters
    }
    load({"optimizer": state_dict}, storage_reader=reader)
    set_optimizer_state_dict(model, optimizer, state_dict, options=options)


def dcp_to_safetensors(dcp_dir: Path, output_dir: Path, *, delete_dcp: bool = False) -> Path:
    """Merge a DCP shard directory into a single `model.safetensors` (offline, single process).

    Thin wrapper over `accelerate.utils.merge_fsdp_weights`, which loads the shards without a
    process group, writes safetensors directly, and — when asked — removes the merged shard
    directory itself, only on the main process and only once the merge has succeeded.

    Args:
        dcp_dir (Path): The DCP shard directory to merge (e.g. `.../pytorch_model_fsdp_0`).
        output_dir (Path): The directory the merged `model.safetensors` is written into.
        delete_dcp (bool): Whether to remove the shard directory once it has been merged.
            Defaults to False.

    Returns:
        Path: The written `model.safetensors` file's path.
    """
    from accelerate.utils import merge_fsdp_weights

    merge_fsdp_weights(
        str(dcp_dir), str(output_dir), safe_serialization=True, remove_checkpoint_dir=delete_dcp
    )
    return output_dir / "model.safetensors"
