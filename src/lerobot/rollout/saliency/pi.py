"""Gradient replay explainers for the local PI0 and PI0.5 policies.

The adapters intentionally call the undecorated ``sample_actions`` body on the
sidecar copy.  The public policy method remains ``@torch.no_grad`` for control
and training code; no shared RTC implementation is modified to manufacture a
gradient for saliency.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .explainer import Explainer, register_explainer


class _PIGradExplainer(Explainer):
    """Common image-gradient replay for a PI policy copy."""

    def _sample_with_grad(self, batch: dict, noise: torch.Tensor, chunk_kwargs: dict):
        policy = self.policy
        model = policy.model
        sample_actions = getattr(model.sample_actions, "__wrapped__", None)
        if sample_actions is None:
            raise RuntimeError(
                "PI saliency requires an uncompiled explainer model; load the sidecar with compile_model=False"
            )
        images, img_masks = policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        target_kwargs = {k: v for k, v in chunk_kwargs.items() if k != "target_action_index"}
        model._saliency_preserve_graph = True
        try:
            if policy.name == "pi0":
                state = policy.prepare_state(batch)
                return sample_actions(
                    model, images, img_masks, tokens, masks, state, noise=noise, **target_kwargs
                )
            # MEM is rejected before loading this explainer. Without MEM PI05 has no state
            # history inputs; keeping this explicit prevents accidental temporal reconstruction.
            return sample_actions(
                model,
                images,
                img_masks,
                tokens,
                masks,
                states=None,
                state_masks=None,
                noise=noise,
                **target_kwargs,
            )
        finally:
            model._saliency_preserve_graph = False

    def compute_saliency(self, batch: dict, noise: torch.Tensor, chunk_kwargs: dict) -> dict[str, np.ndarray]:
        if noise is None:
            raise ValueError("PI saliency replay requires the injected noise tensor")
        policy = self.policy
        policy.eval()
        device = next(policy.parameters()).device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        image_keys = [key for key in policy.config.image_features if key in batch]
        if not image_keys:
            raise ValueError("PI saliency replay found no camera tensors in the preprocessed batch")
        for key in image_keys:
            batch[key] = batch[key].detach().clone().requires_grad_(True)
        noise = noise.to(device)
        chunk_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in chunk_kwargs.items()
        }
        with torch.enable_grad():
            actions = self._sample_with_grad(batch, noise, chunk_kwargs)
            action_dim = policy.config.output_features["action"].shape[0]
            actions = actions[:, :, :action_dim]
            target_index = int(chunk_kwargs.get("target_action_index", 0))
            if not 0 <= target_index < actions.shape[1]:
                raise ValueError(
                    f"target_action_index={target_index} is outside the generated chunk of length {actions.shape[1]}"
                )
            score = actions[0, target_index].pow(2).sum()
            grads = torch.autograd.grad(score, [batch[key] for key in image_keys], allow_unused=True)

        maps: dict[str, np.ndarray] = {}
        for key, grad in zip(image_keys, grads, strict=True):
            image = batch[key].detach()
            if grad is None:
                maps[key] = np.zeros(image.shape[-2:], dtype=np.float32)
                continue
            if image.ndim == 5:
                image, grad = image[:, -1], grad[:, -1]
            maps[key] = (grad * image).abs().sum(dim=1)[0].float().cpu().numpy()
        return maps


@register_explainer("pi0", method="grad_x_input")
class PI0Explainer(_PIGradExplainer):
    pass


@register_explainer("pi05", method="grad_x_input")
class PI05Explainer(_PIGradExplainer):
    pass
