# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rollout-time saliency: per-camera input-pixel importance maps of the executed action."""

from .config import SaliencyConfig
from .explainer import Explainer, get_explainer_cls, register_explainer
from .smolvla import SmolVLAExplainer
from .smolvla_attention import SmolVLAAttentionExplainer
from .pi import PI0Explainer, PI05Explainer
from .pi_attention import PI0AttentionExplainer, PI05AttentionExplainer

__all__ = [
    "Explainer",
    "SaliencyConfig",
    "SmolVLAAttentionExplainer",
    "SmolVLAExplainer",
    "PI0Explainer",
    "PI05Explainer",
    "PI0AttentionExplainer",
    "PI05AttentionExplainer",
    "get_explainer_cls",
    "register_explainer",
]
