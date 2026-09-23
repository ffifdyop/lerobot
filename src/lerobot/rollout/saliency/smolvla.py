# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SmolVLA gradient×input replay of the first action retained by RTC (a_d).

The sidecar reuses the accepted chunk's batch, noise and generation conditions;
it never sends the replayed actions to the robot. Parameters stay frozen and only
image inputs are differentiated. Replay equivalence is tested with path-specific
tolerances, not assumed bitwise across devices. Queued does not mean executed.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .explainer import Explainer, register_explainer


@register_explainer("smolvla")
class SmolVLAExplainer(Explainer):
    """Gradient×input saliency for SmolVLA."""

    def compute_saliency(
        self,
        batch: dict,
        noise: torch.Tensor,
        chunk_kwargs: dict,
    ) -> dict[str, np.ndarray]:
        """Replay one chunk with gradients enabled.

        The batch dict's tensors may be moved across devices; state is defensively
        cloned before the (in-place) pi_aloha decode, so the caller's tensors are
        not mutated.
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

        # Grad mode is thread-local but a leaked `set_grad_enabled(False)` in the recorder thread
        # would drop every frame with a confusing RuntimeError — re-enable it for the replay.
        with torch.enable_grad():
            batch = policy._prepare_batch(batch)
            policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
            for k in batch:
                if k in policy._queues and k != ACTION:
                    batch[k] = torch.stack(list(policy._queues[k]), dim=1)

            img_keys = [k for k in policy.config.image_features if k in batch]
            for k in img_keys:
                batch[k] = batch[k].detach().clone().requires_grad_(True)

            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

            policy.model._saliency_preserve_graph = True
            try:
                actions = policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state, noise=noise, **chunk_kwargs
                )
            finally:
                policy.model._saliency_preserve_graph = False
            actions = actions[:, :, : policy.config.action_feature.shape[0]]
            if policy.config.adapt_to_pi_aloha:
                actions = policy._pi_aloha_encode_actions(actions)

            target_index = int(chunk_kwargs.get("target_action_index", 0))
            if not 0 <= target_index < actions.shape[1]:
                raise ValueError(
                    f"target_action_index={target_index} is outside the generated chunk "
                    f"of length {actions.shape[1]}"
                )
            # Target the action retained at the queue's actual index.
            # Same pixel map as the L2 norm up to a scalar that min-max removes.
            score = actions[0, target_index].pow(2).sum()
            # allow_unused=True: a camera that didn't influence the action yields grad=None
            # instead of a RuntimeError (with the default False it raises, never returns None).
            grads = torch.autograd.grad(score, [batch[k] for k in img_keys], allow_unused=True)

        saliency: dict[str, np.ndarray] = {}
        for key, grad in zip(img_keys, grads, strict=True):
            if grad is None:  # camera had no influence on this action (e.g. padded mask)
                img = batch[key].detach()
                saliency[key] = np.zeros(img.shape[-2:], dtype=np.float32)
                continue
            img = batch[key].detach()
            if img.ndim == 5:
                img, grad = img[:, -1], grad[:, -1]
            cam_map = (grad * img).abs().sum(dim=1)[0]  # (C,H,W) → (H,W)
            saliency[key] = cam_map.float().cpu().numpy()
        return saliency
