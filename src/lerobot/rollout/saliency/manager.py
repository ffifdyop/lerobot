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
        explainer_cls = get_explainer_cls(policy_config.type)  # fail fast, before weights load

        self.device = resolve_explainer_device(saliency_cfg.device, policy_device)
        policy_copy = load_policy()
        policy_copy.to(self.device)
        policy_copy.eval()
        for p in policy_copy.parameters():
            p.requires_grad_(False)  # differentiate wrt input pixels only
        self._explainer = explainer_cls(policy_copy)

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
        )
        self._recorder.start()

    def submit(self, snapshot: dict | None) -> None:
        """Sink entry point called by the RTC engine (drop-oldest, never blocks)."""
        self._recorder.submit(snapshot)

    def stop(self) -> None:
        """Flush and close the overlay videos."""
        self._recorder.close()
