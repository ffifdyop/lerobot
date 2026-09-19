# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from ..view_mask.inference import ViewMaskInferenceMixin, stamp_view_mask
from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


# TODO(Steven): support relative-action policies.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so cached chunk
# actions popped on later ticks get reanchored to the *current* robot state and
# absolute targets drift through the chunk.  Relative-action policies are
# rejected at context-build time today; RTC postprocesses the whole chunk and
# is unaffected.
#
# Candidate fix: drive the policy via ``predict_action_chunk`` and serve a
# local FIFO of postprocessed actions.  Eliminates drift by construction and
# saves per-tick pre/post work, but bypasses ``select_action`` — needs
# fallbacks for SAC (raises), ACT temporal ensembling (ensembler lives in
# ``select_action``), and Diffusion-family (obs-history queues populated as a
# side effect of ``select_action``).


class SyncInferenceEngine(ViewMaskInferenceMixin, InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        # The policy was just reset, so a pending task change has nothing stale to flush.
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task, task_changed = self._take_task()
        # Captured before the policy call: a mask change landing mid-inference must
        # not relabel this action, and its result must be discarded below.
        mask = self._view_mask.snapshot() if self._view_mask is not None else None
        with torch.inference_mode(), autocast_ctx:
            if task_changed:
                # Chunking policies queue actions computed under the previous instruction,
                # so drop them and let the new one take effect on this tick.  Narrower
                # than ``policy.reset``: observation history and other episode state stay.
                logger.info("Task changed to '%s' — dropping precomputed actions", task)
                self._policy.drop_queued_actions()
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            if mask is not None:
                observation = stamp_view_mask(observation, mask)
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        if mask is not None and not self._view_mask.is_executable(mask.revision):
            # The mask changed while the policy was running (or the mask latched
            # FAILED): this action describes a scene the operator has moved on from.
            # Dropping it happens before the interpolator or the robot ever see it.
            logger.info(
                "Discarding action computed under stale view mask revision %d (current %d)",
                mask.revision,
                self._view_mask.revision,
            )
            return None

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        # ``task`` is the pre-inference snapshot: a /subtask landing mid-inference must
        # not relabel this action.
        self._set_dispatched_task(task)
        if mask is not None:
            self._view_mask.set_dispatched(mask)
            # First executable action for the newest revision: the transition is done.
            self._view_mask.mark_active(mask)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    def reset_for_mask_transition(self) -> None:
        """View Mask Inference Flush.

        Sync has no queue to drain — it recomputes from the observation every tick —
        but chunking policies keep an internal action queue from ``select_action``,
        and the processors may carry history (Diffusion-family observation queues).
        Clearing the same state ``reset`` clears, minus the task edge, is what makes
        the next returned action fresh for the new mask.
        """
        logger.info("Sync inference: flushing policy and processors for the new view mask")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """Inference runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # Live task, read without consuming the task-changed edge (the action path needs it).
        task = self.task
        # Text queries share the action path's mask: asking the policy a question about a
        # blacked-out view must show it the same black view the actions were based on.
        mask = self._view_mask.snapshot() if self._view_mask is not None else None
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            if mask is not None:
                observation = stamp_view_mask(observation, mask)
            observation = self._mark_query(observation, query)
            # Reusing the action path's preprocessor is safe only while its steps are
            # stateless per call; the one that is not (an enabled RelativeActionsProcessorStep)
            # is rejected for this backend at context-build time.
            observation = self._preprocessor(observation)
            # No str() coercion: _service_query validates the return value.
            return self._policy.generate_text(observation)
