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

"""Spatial mask boundary used by the rollout preprocessor.

The controller and inference engines carry an immutable operator without knowing
how it changes pixels.  Today the only implementation blacks out whole views;
future rectangle, polygon, or bitmap implementations can share the same snapshot
and processor path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ViewMaskOperator(Protocol):
    """Immutable transformation applied to one policy observation.

    Implementations must return the input mapping unchanged when no target is
    present.  When applying a mask they must not mutate the mapping or tensors
    supplied by the caller.
    """

    @property
    def masked_views(self) -> frozenset[str]:
        """Canonical views whose pixels are affected by this operator."""

    def apply(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return a policy observation with this spatial mask applied."""

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable description recorded for this revision."""


@dataclass(frozen=True)
class FullFrameViewMask:
    """Black out every pixel of each selected canonical view."""

    masked_views: frozenset[str]

    def apply(self, observation: dict[str, Any]) -> dict[str, Any]:
        import torch

        targets = [view for view in self.masked_views if view in observation]
        if not targets:
            return observation

        processed = dict(observation)
        for view in targets:
            value = processed[view]
            if isinstance(value, torch.Tensor):
                processed[view] = torch.zeros_like(value)
        return processed

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "full_frame",
            "views": sorted(self.masked_views),
        }
