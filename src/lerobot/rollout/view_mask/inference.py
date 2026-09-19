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

"""View Mask glue for inference backends.

This module keeps mask-specific provenance and transition flushing out of the
generic :class:`InferenceEngine` interface.  Only backends used by rollout with
the capability enabled mix this class in; ordinary inference backends remain
unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.utils.constants import RUNTIME_CONTEXT

from .keys import VIEW_MASK_SNAPSHOT

logger = logging.getLogger(__name__)


def stamp_view_mask(batch: dict, snapshot: Any) -> dict:
    """Attach one immutable mask snapshot to a policy batch in place."""
    context = dict(batch.get(RUNTIME_CONTEXT) or {})
    context[VIEW_MASK_SNAPSHOT] = snapshot
    batch[RUNTIME_CONTEXT] = context
    return batch


class ViewMaskInferenceMixin:
    """Optional mask provenance/flush behavior for sync and RTC engines."""

    def __init__(self, *args, **kwargs) -> None:
        self._view_mask = None
        super().__init__(*args, **kwargs)

    def attach_view_mask(self, runtime) -> None:
        """Bind the runtime and receive transition callbacks."""
        self._view_mask = runtime
        if runtime is not None:
            runtime.controller.set_sink(self)

    @property
    def view_mask(self):
        return self._view_mask

    @property
    def dispatched_view_mask(self):
        """Snapshot of the most recently returned action, when enabled."""
        if self._view_mask is None:
            return None
        return self._view_mask.dispatched

    def on_mask_transition(self, transition) -> None:
        """Invalidate backend state computed under the previous revision."""
        logger.info(
            "View mask transition %d -> %d: flushing inference state",
            transition.old_revision,
            transition.new_revision,
        )
        self.reset_for_mask_transition()

    def _autosteer_queries_allowed(self) -> bool:
        """Pause automatic planning while a new mask is still APPLYING."""
        return self._view_mask is None or not self._view_mask.blocked

    def reset_for_mask_transition(self) -> None:
        """Default transition invalidation for a backend without a queue."""
        self.reset()
