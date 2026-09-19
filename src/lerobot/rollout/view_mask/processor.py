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

"""Runtime processor step that blacks out masked views for the policy only.

Injected into the *checkpoint preprocessor* right after the observation rename step
and before any normalization or policy-specific image transform, so:

* the dataset, Rerun/Foxglove and camera diagnostics keep reading the original pixels —
  the robot-side observation processor never sees this step; and
* local and remote policies see the same black images they would get from a
  normally-preprocessed all-zero frame.

The mask never lives on the step itself: the inference engine stamps a per-call
snapshot on the batch (see :mod:`lerobot.rollout.view_mask.keys`) and this step
applies it and then strips it, so concurrent or replayed inferences cannot cross
masks and no policy ever sees the marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.utils.constants import RUNTIME_CONTEXT

from .keys import VIEW_MASK_SNAPSHOT


@ProcessorStepRegistry.register(name="view_mask_processor")
@dataclass
class ViewMaskProcessorStep(ObservationProcessorStep):
    """Apply the snapshot's spatial mask operator to canonical policy views.

    Order is preserved and no key is removed: policies with a fixed visual schema
    (ACT, Diffusion, VQBeT, DM05 and friends) keep every camera slot, so the mask
    cannot shift a later view into an earlier slot.  The default operator produces
    a full black image; other operators may affect only part of the tensor.  This is
    *not* a logical missing-view protocol — the model still receives an image token.

    A batch is masked only when the rollout runtime has stamped a snapshot into
    its runtime context; an unstamped batch passes through unchanged.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Apply the mask, then strip the reserved snapshot markers from the transition.

        The strip happens here, not in a later step, because this is the last point
        that legitimately needs the snapshot: everything downstream (normalization,
        device moves, the policy itself) must see an ordinary batch.
        """
        transformed = super().__call__(transition)
        complementary = transformed.get(TransitionKey.COMPLEMENTARY_DATA)
        if isinstance(complementary, dict):
            runtime_context = complementary.get(RUNTIME_CONTEXT)
            if isinstance(runtime_context, dict):
                runtime_context.pop(VIEW_MASK_SNAPSHOT, None)
                if not runtime_context:
                    complementary.pop(RUNTIME_CONTEXT, None)
        return transformed

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Delegate the active spatial mask to the immutable snapshot operator.

        The operator contract requires a derived mapping/tensor for every change,
        so the caller's original observation (the same dict the dataset and
        telemetry read) is never mutated.
        """
        complementary = (self._current_transition or {}).get(TransitionKey.COMPLEMENTARY_DATA) or {}
        runtime_context = complementary.get(RUNTIME_CONTEXT)
        if not isinstance(runtime_context, dict):
            return observation
        snapshot = runtime_context.get(VIEW_MASK_SNAPSHOT)
        if snapshot is None:
            return observation
        return snapshot.operator.apply(observation)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Feature description is unchanged: the mask preserves every key and shape.

        That is the whole point of pixel blacking — the schema the policy declares
        stays exactly what the dataset and the checkpoint agree on.
        """
        return features
