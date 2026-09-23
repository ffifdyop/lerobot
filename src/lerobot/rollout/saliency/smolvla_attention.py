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
"""SmolVLA attention Explainer: Patch Attention Map + Attention Budget Bar (see CONTEXT.md).

Replays the accepted chunk (same batch, same injected noise, same chunk kwargs — the same
contract as the grad_x_input explainer) under ``torch.no_grad`` and taps the eager attention
path via ``SmolVLMWithExpertModel.attention_capture``. Only ``kind == "cross"`` events are
consumed: pure cross-attention (odd layers, suffix queries over prefix keys only) exists
by construction only during denoising, so no query-count filtering is needed.

Semantics (openspec change add-attention-saliency, design D3-D6):
- a_d query only (the suffix query corresponding to the first action retained by RTC);
- per event: mean over heads; then mean over all cross events (odd layers × denoise steps);
- raw shares, never renormalized over image tokens (a model attending to language goes dark);
- per-camera 8×8 patch grids sliced from contiguous 64-column prefix segments;
- five-way attention budget: per-camera image mass + language + state (sums to 1).
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .attention import AttentionAccumulator
from .explainer import Explainer, register_explainer

# SmolVLA letterboxes to 512×512; the SmolVLM connector (patch 16, pixel_shuffle scale 4)
# turns each camera into an 8×8 token grid. This resolution is part of the Patch Attention
# Map's contract — a checkpoint that changes it must fail fast, not silently misalign.
PATCH_GRID = 8
TOKENS_PER_CAMERA = PATCH_GRID * PATCH_GRID  # 64


@register_explainer("smolvla", method="attention")
class SmolVLAAttentionExplainer(Explainer):
    """a_d's cross-attention share over image patch tokens, per camera, plus the budget split."""

    map_kind: str = "patch"

    def __init__(self, policy) -> None:
        super().__init__(policy)
        self._vlm = policy.model.vlm_with_expert
        self._accumulator = AttentionAccumulator()
        # Per-instance tap on THIS model copy only; the control-loop model's attribute stays None.
        self._vlm.attention_capture = self._on_attention
        # Last five-way budget: {cam_key: mass, ..., "language": mass, "state": mass}. Read by
        # the recorder right after compute_saliency returns (same thread, no race).
        self.last_budget: dict[str, float] | None = None

    def close(self) -> None:
        """Detach the tap (manager teardown). Idempotent."""
        self._vlm.attention_capture = None

    def _on_attention(self, layer_idx: int, probs: torch.Tensor, kind: str) -> None:
        if kind != "cross":
            return
        step = getattr(self._vlm, "attention_step", None)
        # Select the action retained at the queue's actual target index; mean over heads; float32; never
        # retain the full probability matrix.
        row_index = getattr(self, "_query_index", 0)
        if probs.ndim != 4 or not 0 <= row_index < probs.shape[2]:
            raise RuntimeError("SmolVLA attention query layout mismatch")
        row = probs[0, :, row_index, :].detach().float().mean(dim=0)
        self._accumulator.add(layer_idx, step, row)

    def compute_saliency(
        self,
        batch: dict,
        noise: torch.Tensor,
        chunk_kwargs: dict,
    ) -> dict[str, np.ndarray]:
        """Replay one chunk under no_grad, capturing cross-attention.

        Returns {camera_key: (8, 8) float32 patch map}; also sets ``self.last_budget``.
        Mirrors the grad explainer's snapshot contract (defensive state clone, no caller
        mutation); raises on missing noise or prefix-layout mismatch (recorder degrades).
        """
        policy = self.policy
        if noise is None:
            raise ValueError(
                "Saliency replay requires the injected noise tensor (snapshot came from an engine "
                "without noise injection); replay of the accepted chunk is impossible"
            )
        policy.eval()
        device = next(policy.parameters()).device

        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # Defensive clone: a pi_aloha decode can modify state in place; don't corrupt the caller's snapshot.
        if OBS_STATE in batch:
            batch[OBS_STATE] = batch[OBS_STATE].clone()
        noise = noise.to(device)
        chunk_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in (chunk_kwargs or {}).items()
        }

        self._query_index = int((chunk_kwargs or {}).get("target_action_index", 0))
        self._accumulator = AttentionAccumulator(
            getattr(self, "attention_layers", None), getattr(self, "attention_steps", None)
        )
        self.last_budget = None
        with torch.no_grad():
            batch = policy._prepare_batch(batch)
            policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
            for k in batch:
                if k in policy._queues and k != ACTION:
                    batch[k] = torch.stack(list(policy._queues[k]), dim=1)

            img_keys = [k for k in policy.config.image_features if k in batch]
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

            try:
                policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state, noise=noise, **chunk_kwargs
                )
            finally:
                self._vlm.attention_step = None

        num_steps = policy.model._attention_num_steps
        vlm = policy.model.vlm_with_expert
        num_layers = vlm.num_vlm_layers
        multiple = num_layers // vlm.num_expert_layers
        self_every = vlm.self_attn_every_n_layers
        valid_layers = {
            layer
            for layer in range(num_layers)
            if layer % multiple == 0 and not (self_every > 0 and layer % self_every == 0)
        }
        mean_row = self._accumulator.mean(valid_layers, num_steps).cpu().numpy()

        # Column attribution: prepared image slots (real cameras followed by any configured
        # empty-camera placeholders) are contiguous 64-token segments in prefix order, then
        # tokenizer_max_length language columns and one state column.  Missing cameras are
        # never mapped onto a neighboring view; their masked slot remains budget-only.
        n_image_slots = len(images)
        lang_len = policy.config.tokenizer_max_length
        expected = TOKENS_PER_CAMERA * n_image_slots + lang_len + 1
        if mean_row.shape[0] != expected:
            raise RuntimeError(
                f"Prefix layout mismatch: attention has {mean_row.shape[0]} keys, expected "
                f"{expected} (= {TOKENS_PER_CAMERA}×{n_image_slots} image slots + {lang_len} language + 1 "
                "state). The checkpoint's prefix layout changed (pad_language_to != "
                "'max_length'? add_image_special_tokens? prefix_length?) — refusing to guess."
            )

        maps: dict[str, np.ndarray] = {}
        budget: dict[str, float] = {}
        for i, key in enumerate(img_keys):
            seg = mean_row[i * TOKENS_PER_CAMERA : (i + 1) * TOKENS_PER_CAMERA]
            maps[key] = seg.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
            budget[key] = float(seg.sum())
        lang_end = TOKENS_PER_CAMERA * n_image_slots + lang_len
        budget["language"] = float(mean_row[TOKENS_PER_CAMERA * n_image_slots : lang_end].sum())
        budget["state"] = float(mean_row[-1])
        self.last_budget = budget
        return maps
