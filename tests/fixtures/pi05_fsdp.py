# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Small real PI05 architecture: production forwards, no downloads or full-size weights."""

from unittest.mock import patch

import torch
from transformers import SiglipVisionConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05 import modeling_pi05 as pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def tiny_pi05(*, checkpointing=False, device="cpu", config=None, pretrained_path=None):
    def gemma_config(variant):
        return pi05.GemmaConfig(
            width=32 if variant == "gemma_2b" else 16,
            depth=2,
            mlp_dim=64,
            num_heads=8,
            num_kv_heads=1,
            head_dim=4,
        )

    paligemma_type = pi05.PaliGemmaForConditionalGenerationWithPiGemma
    expert_type = pi05.PiGemmaForCausalLM

    def paligemma(config):
        config.text_config.vocab_size = 64
        config._vocab_size = 64
        config.projection_dim = 32
        config.vision_config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
            projection_dim=32,
        )
        return paligemma_type(config)

    def expert(config):
        config.vocab_size = 64
        return expert_type(config)

    if config is None:
        config = PI05Config(
            device=device,
            dtype="float32",
            image_resolution=(16, 16),
            max_action_dim=4,
            max_state_dim=4,
            chunk_size=2,
            n_action_steps=2,
            num_inference_steps=2,
            gradient_checkpointing=checkpointing,
            push_to_hub=False,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        )
    with (
        patch.object(pi05, "get_gemma_config", gemma_config),
        patch.object(pi05, "PaliGemmaForConditionalGenerationWithPiGemma", paligemma),
        patch.object(pi05, "PiGemmaForCausalLM", expert),
    ):
        if pretrained_path is not None:
            return pi05.PI05Policy.from_pretrained(pretrained_path, config=config)
        return pi05.PI05Policy(config)


def pi05_batch(*, batch_size=2, seed=10, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    return {
        key: value.to(device)
        for key, value in {
            OBS_STATE: torch.randn(batch_size, 4, generator=generator),
            "observation.images.top": torch.rand(batch_size, 3, 16, 16, generator=generator),
            OBS_LANGUAGE_TOKENS: torch.randint(0, 64, (batch_size, 3), generator=generator),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 3, dtype=torch.bool),
            ACTION: torch.randn(batch_size, 2, 4, generator=generator),
        }.items()
    }


def deterministic_flow(policy):
    """Remove rank-local flow randomness so global-batch parity is well-defined."""
    policy.model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    policy.model.sample_time = lambda size, device: torch.full((size,), 0.5, device=device)
