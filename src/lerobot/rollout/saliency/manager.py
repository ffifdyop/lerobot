# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SaliencyManager: owns the explainer's model copy on a dedicated GPU and the recorder.

The control loop stays on its own GPU under no_grad; this model copy lives on another
device (when one exists) and replays chunks with gradients enabled. See CONTEXT.md:
Explainer.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import torch

from lerobot.configs import FeatureType

from .config import SaliencyConfig
from .explainer import get_explainer_cls
from .recorder import SaliencyRecorder

logger = logging.getLogger(__name__)


def resolve_explainer_device(cfg_device: str | None, policy_device: str) -> str:
    """Explicit --saliency.device wins; else a second GPU if present; else the policy device."""
    if cfg_device:
        return cfg_device
    if torch.cuda.device_count() > 1:
        if policy_device in ("cuda", "cuda:0"):
            return "cuda:1"
        # Policy on CPU or already on cuda:1: give the explainer the other GPU —
        # explicit --saliency.device overrides. Anything else (mps/xpu/...) stays on
        # the policy device; cross-vendor defaults would be surprising.
        if policy_device in ("cpu", "cuda:1"):
            return "cuda:0"
    return policy_device


class SaliencyManager:
    """Lifecycle owner for the saliency sidecar: model copy, explainer, recorder."""

    def __init__(
        self,
        saliency_cfg: SaliencyConfig,
        policy_config,
        policy_device: str,
        display_mode: str,
        load_policy: Callable[[], torch.nn.Module],
    ) -> None:
        explainer_cls = get_explainer_cls(
            policy_config.type, saliency_cfg.method
        )  # fail fast, before weights load

        if getattr(saliency_cfg, "enabled", True) and policy_config.type == "pi05" and (
            getattr(policy_config, "use_visual_memory", False)
            or getattr(policy_config, "use_proprioceptive_memory", False)
        ):
            raise ValueError(
                "PI0.5 saliency does not support MEM replay yet; disable use_visual_memory and "
                "use_proprioceptive_memory or run without saliency"
            )

        # Patch overlays need the policy's letterbox target to place the token grid. SmolVLA
        # stores it as (width, height) (modeling_smolvla.py passes [1]/[0] to resize_with_pad);
        # the overlay math wants (h, w). Attention saliency without letterboxing (None) has no
        # well-defined patch→pixel mapping: fail fast at startup rather than misrender.
        rwp = getattr(policy_config, "resize_imgs_with_padding", None)
        image_resolution = getattr(policy_config, "image_resolution", None)
        if saliency_cfg.method == "attention" and rwp is None and image_resolution is None:
            raise ValueError(
                "--saliency.method=attention requires the policy to letterbox images "
                "(resize_imgs_with_padding); got None"
            )
        if rwp is not None:
            letterbox_hw = (rwp[1], rwp[0])
        elif image_resolution is not None:
            letterbox_hw = (image_resolution[0], image_resolution[1])
        else:
            letterbox_hw = (512, 512)

        self.device = resolve_explainer_device(saliency_cfg.device, policy_device)
        policy_copy = load_policy()
        policy_copy.to(self.device)
        policy_copy.eval()
        for p in policy_copy.parameters():
            p.requires_grad_(False)  # differentiate wrt input pixels only
        self._explainer = explainer_cls(policy_copy)
        self._explainer.attention_layers = getattr(saliency_cfg, "attention_layers", None)
        self._explainer.attention_steps = getattr(saliency_cfg, "attention_steps", None)
        if saliency_cfg.method == "attention" and (
            self._explainer.attention_layers is not None or self._explainer.attention_steps is not None
        ):
            if policy_config.type == "smolvla":
                vlm = policy_copy.model.vlm_with_expert
                num_layers = int(getattr(vlm, "num_vlm_layers", 0))
                num_expert_layers = int(getattr(vlm, "num_expert_layers", num_layers))
                multiple = max(1, num_layers // max(1, num_expert_layers))
                self_every = int(getattr(vlm, "self_attn_every_n_layers", 0))
                valid_layers = {
                    layer
                    for layer in range(num_layers)
                    if layer % multiple == 0 and not (self_every > 0 and layer % self_every == 0)
                }
                valid_step_count = int(getattr(policy_copy.config, "num_steps", 0))
            else:
                valid_layers = set(range(len(policy_copy.model.paligemma_with_expert.gemma_expert.model.layers)))
                valid_step_count = int(
                    getattr(policy_copy.config, "num_inference_steps", getattr(policy_copy.config, "num_steps", 0))
                )
            if self._explainer.attention_layers is not None and not set(self._explainer.attention_layers).issubset(
                valid_layers
            ):
                raise ValueError(
                    f"--saliency.attention_layers contains a non-capturable layer; valid layers are "
                    f"{sorted(valid_layers)}"
                )
            if self._explainer.attention_steps is not None and any(
                value >= valid_step_count for value in self._explainer.attention_steps
            ):
                if valid_step_count <= 0:
                    raise ValueError(
                        "Unable to determine valid denoising steps for attention selection"
                    )
                raise ValueError(
                    f"--saliency.attention_steps contains an out-of-range index; valid range is "
                    f"[0, {valid_step_count})"
                )

        output_dir = saliency_cfg.output_dir or os.path.join(
            "outputs", "saliency", time.strftime("%Y%m%d_%H%M%S")
        )
        # input_features also holds non-visual entries (e.g. observation.state); log only cameras.
        cameras = [
            key
            for key, feature in getattr(policy_config, "input_features", {}).items()
            if feature.type is FeatureType.VISUAL
        ]
        logger.info(
            "Saliency explainer ready: policy copy on %s, cameras %s, output %s",
            self.device,
            cameras,
            output_dir,
        )
        self._recorder = SaliencyRecorder(
            explainer=self._explainer,
            output_dir=output_dir,
            video_fps=saliency_cfg.video_fps,
            alpha=saliency_cfg.alpha,
            blur_sigma=saliency_cfg.blur_sigma,
            display=saliency_cfg.display,
            display_mode=display_mode,
            norm_scope=saliency_cfg.norm_scope,
            letterbox_hw=letterbox_hw,
        )
        self._recorder.start()

    def submit(self, snapshot: dict | None) -> None:
        """Sink entry point called by the RTC engine (drop-oldest, never blocks)."""
        self._recorder.submit(snapshot)

    def set_view_mask(self, runtime) -> None:
        """Bind the session's View Mask runtime so stale revisions can be filtered.

        A snapshot computed under revision N must not be published once the mask has
        moved on: the overlay would describe a scene the operator already changed.
        """
        self._recorder.set_view_mask(runtime)

    def set_reset_epoch(self, epoch: int) -> None:
        self._recorder.set_reset_epoch(epoch)

    def stop(self) -> None:
        """Flush and close the overlay videos, and detach any model tap the explainer holds."""
        self._recorder.close()
        close = getattr(self._explainer, "close", None)
        if close is not None:
            close()
