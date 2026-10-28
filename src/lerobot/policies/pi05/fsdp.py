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

"""Runtime-only PI05 FSDP boundaries and portable weight names.

The ordinary model tree stays unchanged until ``prepare_for_fsdp``. DCP uses the
prepared tree; exported weights use the original PI05 names. No parameter is copied.
"""

import re
from collections.abc import Callable

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class PI05JointDecoderLayer(nn.Module):
    """Own both experts at the boundary where their joint attention is computed."""

    def __init__(self, vlm: nn.Module, expert: nn.Module, compute_joint: Callable):
        super().__init__()
        self.vlm = vlm
        self.expert = expert
        self.compute_joint = compute_joint

    def forward(self, *args, branch: str | None = None, checkpointing: bool = False, **kwargs):
        if branch is not None:
            # KV-cache prefix/suffix paths must enter the SAME FSDP boundary as training.
            return getattr(self, branch)(*args, **kwargs)
        if checkpointing and self.training:
            # Checkpoint INSIDE the FSDP boundary: the arguments have already been
            # cast by MixedPrecisionPolicy. An outer checkpoint of this FSDP module
            # would replay the original FP32 inputs while the backward-state hooks
            # skip the forward cast, causing BF16 recomputation metadata mismatches.
            # FSDP's pre-backward hook gathers parameters before this recomputation.
            return checkpoint(
                self.compute_joint,
                *args,
                layers=(self.vlm, self.expert),
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return self.compute_joint(*args, layers=(self.vlm, self.expert), **kwargs)


class _DecoderLayerView(nn.Module):
    """Parameter-free view used by PiGemmaModel's existing cached inference loop."""

    def __init__(self, joint: PI05JointDecoderLayer, branch: str):
        super().__init__()
        # Registering joint here would give its parameters a second owner.
        object.__setattr__(self, "_joint", joint)
        self._branch = branch

    def forward(self, *args, **kwargs):
        return self._joint(*args, branch=self._branch, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            joint = object.__getattribute__(self, "_joint")
            return getattr(getattr(joint, self._branch), name)


def prepare_joint_layers(model: nn.Module, compute_joint: Callable) -> None:
    """Reparent decoder parameters once, preserving optimizer parameter identities."""
    if hasattr(model, "joint_layers"):
        return
    vlm = model.paligemma.model.language_model
    expert = model.gemma_expert.model
    joint_layers = nn.ModuleList(
        PI05JointDecoderLayer(left, right, compute_joint)
        for left, right in zip(vlm.layers, expert.layers, strict=True)
    )
    del vlm.layers
    del expert.layers
    model.joint_layers = joint_layers
    # Keep these outside the module registry; .to(), .train() and state_dict() all
    # visit the real decoders through joint_layers only.
    object.__setattr__(vlm, "layers", tuple(_DecoderLayerView(layer, "vlm") for layer in joint_layers))
    object.__setattr__(expert, "layers", tuple(_DecoderLayerView(layer, "expert") for layer in joint_layers))


_JOINT_KEY = re.compile(r"^(.*paligemma_with_expert\.)joint_layers\.(\d+)\.(vlm|expert)\.(.+)$")


def canonical_pi05_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map runtime/DCP decoder keys to the portable PI05 checkpoint schema."""
    result = {}
    for name, value in state_dict.items():
        if match := _JOINT_KEY.match(name):
            prefix, index, branch, suffix = match.groups()
            owner = "paligemma.model.language_model" if branch == "vlm" else "gemma_expert.model"
            name = f"{prefix}{owner}.layers.{index}.{suffix}"
        if name in result:
            raise ValueError(f"Duplicate PI05 checkpoint key after FSDP conversion: {name}")
        result[name] = value
    return result
