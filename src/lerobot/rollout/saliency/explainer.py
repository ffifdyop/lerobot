# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Explainer registry: maps local policy types to saliency explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_EXPLAINER_REGISTRY: dict[str, type] = {}


def register_explainer(policy_type: str, explainer_cls: type | None = None):
    """Register an Explainer class for a policy type (draccus ChoiceRegistry name).

    Usable both directly (``register_explainer("smolvla", SmolVLAExplainer)``) and as a
    decorator (``@register_explainer("smolvla")``). Refuses to overwrite an existing
    registration — a duplicate would silently mask the first explainer.
    """
    if policy_type in _EXPLAINER_REGISTRY:
        existing = _EXPLAINER_REGISTRY[policy_type].__name__
        raise ValueError(
            f"Saliency explainer already registered for policy type '{policy_type}' ({existing}); "
            "refusing to overwrite. Register a different policy type or explainer class."
        )
    if explainer_cls is None:

        def decorator(cls: type) -> type:
            _EXPLAINER_REGISTRY[policy_type] = cls
            return cls

        return decorator
    _EXPLAINER_REGISTRY[policy_type] = explainer_cls
    return explainer_cls


def get_explainer_cls(policy_type: str) -> type:
    """Look up the Explainer for a policy type. Fail fast — never silently degrade."""
    try:
        return _EXPLAINER_REGISTRY[policy_type]
    except KeyError:
        registered = ", ".join(sorted(_EXPLAINER_REGISTRY)) or "(none)"
        raise ValueError(
            f"No saliency explainer registered for policy type '{policy_type}'. "
            f"Registered types: {registered}. Implement and register an Explainer for this policy, "
            "or run without --saliency.enabled."
        ) from None


class Explainer:
    """Base class: wraps a local policy copy and computes per-camera saliency maps."""

    def __init__(self, policy) -> None:
        self.policy = policy

    def compute_saliency(
        self,
        batch: dict,
        noise: torch.Tensor,
        chunk_kwargs: dict,
    ) -> dict[str, np.ndarray]:
        """Replay one chunk with gradients enabled.

        Args:
            batch: the preprocessed observation batch the policy acted on.
            noise: the exact noise tensor the executed chunk was sampled from.
            chunk_kwargs: the exact chunk kwargs (inference_delay, prev_chunk_left_over).

        Returns:
            {camera_key: (H, W) float32 saliency map at the batch image resolution}.
        """
        raise NotImplementedError
