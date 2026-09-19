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

"""Reserved batch keys for the per-inference View Mask snapshot.

The inference engine stamps these on the observation batch it feeds the checkpoint
preprocessor; :class:`~lerobot.rollout.view_mask.processor.ViewMaskProcessorStep`
consumes and removes them.

They are stored inside the generic runtime-context complementary field purely so they
survive the batch→transition grouping and reach the step — which then strips them, so
unlike the text-query keys nothing downstream (normalization, device moves, the policy
itself) ever sees them.

The names are not batch keys and therefore cannot collide with an observation or a
camera configured by an operator.
"""

from __future__ import annotations

# These names are local to the View Mask capability.  The shared processor layer
# carries them inside the generic ``RUNTIME_CONTEXT`` container instead of knowing
# about each rollout feature's vocabulary.
VIEW_MASK_SNAPSHOT = "view_mask_snapshot"

__all__ = ["VIEW_MASK_SNAPSHOT"]
