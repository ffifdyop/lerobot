"""Action-expert attention explainers for local PI0 and PI0.5."""

from __future__ import annotations

import numpy as np
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .attention import AttentionAccumulator
from .explainer import Explainer, register_explainer


class _PIAttentionExplainer(Explainer):
    map_kind = "patch"

    def __init__(self, policy, *, has_state_query: bool):
        super().__init__(policy)
        self._model = policy.model
        self._has_state_query = has_state_query
        self._query_index = 0
        self._accumulator = AttentionAccumulator()
        self._key_count = 0
        self._query_count = 0
        self.last_budget: dict[str, float] | None = None
        self._model.attention_capture = self._on_attention

    def close(self) -> None:
        self._model.attention_capture = None
        self._model.attention_step = None
        expert = getattr(getattr(self._model, "paligemma_with_expert", None), "gemma_expert", None)
        if expert is not None and hasattr(expert, "model"):
            expert.model.attention_capture = None
            expert.model.attention_step = None

    def _on_attention(self, layer_idx: int, step: int | None, probs: torch.Tensor) -> None:
        if probs.ndim != 4 or probs.shape[2:] != (self._query_count, self._key_count):
            raise RuntimeError("PI attention key/query layout mismatch")
        row = probs[0, :, self._query_index, :].detach().float().mean(dim=0)
        self._accumulator.add(layer_idx, step, row)

    @staticmethod
    def _patch_grid(policy) -> tuple[int, int, int]:
        vision = policy.model.paligemma_with_expert.paligemma.model.vision_tower
        cfg = vision.config
        patch_size = int(getattr(cfg, "patch_size", 14))
        height, width = (int(value) for value in policy.config.image_resolution)
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"image_resolution={policy.config.image_resolution} is not divisible by patch_size={patch_size}"
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        return grid_h * grid_w, grid_h, grid_w

    def _layout(
        self, batch: dict, images: list[torch.Tensor]
    ) -> tuple[list[tuple[str, int]], int, int, int, int]:
        policy = self.policy
        configured = list(policy.config.image_features)
        # Both PI preprocessors append real images first and masked placeholders after them;
        # use that actual prefix order instead of a configured-camera index when a view is
        # missing in the snapshot.
        present_keys = [key for key in configured if key in batch]
        present = [(key, slot) for slot, key in enumerate(present_keys)]
        if len(images) != len(configured):
            raise RuntimeError("PI prepared image slots do not match configured cameras")
        image_tokens, grid_h, grid_w = self._patch_grid(policy)
        lang_len = int(batch[OBS_LANGUAGE_TOKENS].shape[1])
        return present, image_tokens, grid_h, grid_w, lang_len

    def compute_saliency(self, batch: dict, noise: torch.Tensor, chunk_kwargs: dict) -> dict[str, np.ndarray]:
        if noise is None:
            raise ValueError("PI attention replay requires the injected noise tensor")
        policy = self.policy
        policy.eval()
        device = next(policy.parameters()).device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        images, img_masks = policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        camera_slots, image_tokens, grid_h, grid_w, lang_len = self._layout(batch, images)
        # Validate against actual captured key/query dimensions during this replay;
        # obtaining embeddings just to count prefix tokens would encode every camera twice.
        prefix_len = len(images) * image_tokens + lang_len
        self._query_count = int(noise.shape[1]) + int(self._has_state_query)
        self._key_count = prefix_len + self._query_count
        target_index = int(chunk_kwargs.get("target_action_index", 0))
        if not 0 <= target_index < noise.shape[1]:
            raise ValueError("target_action_index is outside the generated chunk")
        self._query_index = target_index + (1 if self._has_state_query else 0)
        self._accumulator = AttentionAccumulator(
            getattr(self, "attention_layers", None), getattr(self, "attention_steps", None)
        )
        self.last_budget = None
        model = self._model
        model.attention_capture = self._on_attention
        sample_actions = getattr(model.sample_actions, "__wrapped__", None)
        if sample_actions is None:
            raise RuntimeError("PI attention saliency requires an uncompiled explainer model")
        replay_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in chunk_kwargs.items()
            if k != "target_action_index"
        }
        try:
            with torch.no_grad():
                if policy.name == "pi0":
                    sample_actions(
                        model,
                        images,
                        img_masks,
                        tokens,
                        masks,
                        policy.prepare_state(batch),
                        noise=noise.to(device),
                        **replay_kwargs,
                    )
                else:
                    sample_actions(
                        model,
                        images,
                        img_masks,
                        tokens,
                        masks,
                        states=None,
                        state_masks=None,
                        noise=noise.to(device),
                        **replay_kwargs,
                    )
        finally:
            model.attention_capture = None
            model.attention_step = None
            expert = getattr(getattr(model, "paligemma_with_expert", None), "gemma_expert", None)
            if expert is not None and hasattr(expert, "model"):
                expert.model.attention_capture = None
                expert.model.attention_step = None
        num_steps = model._attention_num_steps
        valid_layers = set(range(len(model.paligemma_with_expert.gemma_expert.model.layers)))
        mean_row = self._accumulator.mean(valid_layers, num_steps).cpu().numpy()
        # The suffix/action columns are retained in the budget but never rendered as images.
        budget: dict[str, float] = {}
        for key, camera_index in camera_slots:
            start = camera_index * image_tokens
            budget[key] = float(mean_row[start : start + image_tokens].sum())
        for key in policy.config.image_features:
            budget.setdefault(key, 0.0)
        prefix_image_end = len(policy.config.image_features) * image_tokens
        prompt_share = float(mean_row[prefix_image_end : prefix_image_end + lang_len].sum())
        if policy.name == "pi0":
            budget["language"] = prompt_share
            # PI0's state is the first suffix query/key, not part of the prefix. Keep the
            # category explicit and include the action-expert suffix below.
            budget["state"] = float(mean_row[prefix_len])
            action_start = prefix_len + 1
        else:
            budget["prompt"] = prompt_share
            action_start = prefix_len
        budget["action"] = float(mean_row[action_start:].sum())
        self.last_budget = budget

        maps: dict[str, np.ndarray] = {}
        for key, camera_index in camera_slots:
            start = camera_index * image_tokens
            maps[key] = mean_row[start : start + image_tokens].reshape(grid_h, grid_w).astype(np.float32)
        return maps


@register_explainer("pi0", method="attention")
class PI0AttentionExplainer(_PIAttentionExplainer):
    def __init__(self, policy):
        super().__init__(policy, has_state_query=True)


@register_explainer("pi05", method="attention")
class PI05AttentionExplainer(_PIAttentionExplainer):
    def __init__(self, policy):
        super().__init__(policy, has_state_query=False)
