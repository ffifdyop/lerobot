# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Configuration for rollout-time saliency visualization (see CONTEXT.md: Saliency Map / Overlay Video / Explainer)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..inference import InferenceEngineConfig, RTCInferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class SaliencyConfig:
    """Saliency overlay video settings (``--saliency.*`` flags on lerobot-rollout)."""

    # Master switch. Requires a local policy with a registered Explainer and the RTC inference engine.
    enabled: bool = False
    # Saliency method: "grad_x_input" = pixel sensitivity (∂‖a_d‖²/∂x), "attention" = patch
    # attention map (a_d's cross-attention share over each camera's patch tokens).
    # "integrated" is a planned upgrade.
    method: str = "grad_x_input"
    # Stream overlay frames to rerun entities ``saliency/<camera>`` (requires --display_data).
    display: bool = True
    # Device for the explainer's model copy. None = auto: a second GPU if available, else the policy device.
    device: str | None = None
    # Output directory for the per-camera mp4s. None = outputs/saliency/<run timestamp>/
    output_dir: str | None = None
    # Fixed fps of the saved mp4s (frames arrive at policy tick rate; dropped frames just skip).
    video_fps: float = 10.0
    # Blend weight of the heatmap in the overlay.
    alpha: float = 0.5
    # Gaussian smoothing sigma (px at camera resolution) applied to the raw saliency map. 0 disables.
    blur_sigma: float = 2.0
    # Color-scale scope: "run" = one online scale for the whole run (lo=0, hi = slowly decaying
    # running max — color depth ≈ absolute contribution, comparable across cameras AND time);
    # "chunk" = recompute the shared-across-cameras range per chunk (best local contrast, but a
    # quiet moment is stretched to full red just like a hot one).
    norm_scope: str = "run"
    # Optional zero-based action-expert attention layers/denoising steps. None means all.
    attention_layers: list[int] | None = None
    attention_steps: list[int] | None = None


def validate_saliency(
    saliency: SaliencyConfig,
    inference: InferenceEngineConfig,
    display_mode: str,
    display_data: bool,
    use_torch_compile: bool = False,
) -> None:
    """Fail fast on saliency flag combinations that cannot work. No-op when disabled."""
    if not saliency.enabled:
        return
    if not isinstance(inference, RTCInferenceConfig):
        raise ValueError(
            "--saliency.enabled=true currently requires --inference.type=rtc "
            "(noise injection lives in the RTC engine)"
        )
    if saliency.method not in ("grad_x_input", "attention"):
        raise ValueError(
            f"Unsupported --saliency.method={saliency.method!r} (implemented: 'grad_x_input', 'attention')"
        )
    if saliency.display and display_mode != "rerun":
        raise ValueError("--saliency.display=true requires --display_mode=rerun")
    if saliency.display and not display_data:
        raise ValueError(
            "--saliency.display=true requires --display_data=true (rerun is never initialized otherwise)"
        )
    if not 0.0 < saliency.alpha < 1.0:
        raise ValueError(f"--saliency.alpha must be in (0, 1), got {saliency.alpha}")
    if saliency.norm_scope not in ("run", "chunk"):
        raise ValueError(f"--saliency.norm_scope must be 'run' or 'chunk', got {saliency.norm_scope!r}")
    for name, values in (("attention_layers", saliency.attention_layers), ("attention_steps", saliency.attention_steps)):
        if values is not None:
            if not values:
                raise ValueError(f"--saliency.{name} cannot be empty; use null for all values")
            if any(not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"--saliency.{name} must contain unique non-negative integers")
            if len(set(values)) != len(values):
                raise ValueError(f"--saliency.{name} must not contain duplicates")
    if saliency.method != "attention" and (saliency.attention_layers is not None or saliency.attention_steps is not None):
        raise ValueError("--saliency.attention_layers/attention_steps require --saliency.method=attention")
    if use_torch_compile:
        # Not fatal: compiled inductor graphs may reorder FP ops, so the eager replay is
        # approximate rather than deterministic. Warn loudly instead of failing.
        logger.warning(
            "--saliency.enabled=true combined with --use_torch_compile: torch.compile may reorder "
            "FP ops in the compiled graph, so the explainer's eager replay may drift slightly "
            "from the compiled executed action"
        )
