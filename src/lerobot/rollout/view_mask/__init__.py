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

"""View ablation ("black-view mask") for ``lerobot-rollout``.

Opt-in runtime capability that applies a spatial mask to selected canonical policy
views while leaving camera capture, dataset frames and telemetry on the original
pixels.  The default operator replaces the whole image with zeros.  See
``openspec/changes/add-rollout-view-ablation`` and CONTEXT.md (View Ablation /
Black-view Mask / Mask Transition).

The package is import-light on purpose: ``config`` and the plan resolver are pure
stdlib, so ``RolloutConfig.__post_init__`` can validate the flag combination
without paying for torch.
"""

from .config import ViewMaskConfig, ViewMaskPlan, build_view_mask_plan, validate_view_mask
from .controller import (
    MaskApplicationState,
    ViewMaskController,
    ViewMaskSnapshot,
    ViewMaskTransitionSink,
)
from .events import ViewMaskEventRecorder
from .operator import FullFrameViewMask, ViewMaskOperator

__all__ = [
    "MaskApplicationState",
    "FullFrameViewMask",
    "ViewMaskConfig",
    "ViewMaskController",
    "ViewMaskEventRecorder",
    "ViewMaskPlan",
    "ViewMaskSnapshot",
    "ViewMaskTransitionSink",
    "ViewMaskOperator",
    "build_view_mask_plan",
    "validate_view_mask",
]
