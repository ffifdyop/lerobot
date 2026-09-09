#!/usr/bin/env python

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

"""Identity-like processor pipelines for the DM05 remote policy.

Normalization happens server-side (norm_stats ship with the DM05 checkpoint), so
unlike local policies the pipeline has no normalize/unnormalize steps:

    pre:  rename_observations -> add_batch_dim -> to_device
    post: to_cpu

The transition adapters default to the standard batch converters on the
preprocessor and to the policy-action converters on the postprocessor
(:func:`lerobot.processor.make_policy_processor_pipelines`), matching what
``SyncInferenceEngine`` feeds in and consumes.
"""

from dataclasses import fields as dataclass_fields
from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_dm05 import DM05Config


def make_dm05_pre_post_processors(
    config: DM05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the DM05 pre/post pipelines from the policy config alone."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    return make_policy_processor_pipelines(
        input_steps=[steps.rename_observations, steps.add_batch_dim, steps.to_device],
        output_steps=[steps.to_cpu],
    )


def make_dm05_pre_post_processors_from_pretrained(
    config: DM05Config,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build DM05 pipelines for a config-only deployment directory.

    A DM05 deployment dir has no serialized pipeline files (it has no weights
    either), so unlike the generic ``pretrained_path`` branch this always builds
    fresh from the config and applies the caller's step overrides by hand —
    e.g. the ``device_processor`` device and the ``rename_observations_processor``
    map that the rollout context always passes.
    """
    preprocessor, postprocessor = make_dm05_pre_post_processors(
        config=config, dataset_stats=dataset_stats, dataset_meta=dataset_meta
    )
    _apply_step_overrides(preprocessor, preprocessor_overrides)
    _apply_step_overrides(postprocessor, postprocessor_overrides)
    return preprocessor, postprocessor


def _apply_step_overrides(pipeline: PolicyProcessorPipeline, overrides: dict[str, Any] | None) -> None:
    """Apply ``from_pretrained``-style step overrides to a freshly built pipeline.

    Override keys match a step's registry name (e.g. ``device_processor``) or, as
    a convenience, its class name; keys matching no step raise instead of being
    dropped silently. After ``setattr`` the step's ``__post_init__`` is re-run so
    derived attributes follow the overridden config (``DeviceProcessorStep``
    resolves its ``torch.device`` there — see groot's identical helper).
    """
    if not overrides:
        return
    for key, step_overrides in overrides.items():
        matching = [
            step
            for step in pipeline.steps
            if key == type(step).__name__ or key == getattr(type(step), "_registry_name", None)
        ]
        if not matching:
            available = [
                getattr(type(step), "_registry_name", None) or type(step).__name__ for step in pipeline.steps
            ]
            raise KeyError(
                f"Override key '{key}' matches no step of the DM05 pipeline. Available step keys: {available}."
            )
        if len(matching) > 1:
            raise ValueError(f"Override key {key!r} matches multiple DM05 pipeline steps.")
        step = matching[0]
        init_field_names = {f.name for f in dataclass_fields(step) if f.init}
        for field_name, value in dict(step_overrides).items():
            if field_name not in init_field_names:
                raise TypeError(
                    f"Override field '{field_name}' is not a config field of step '{key}'. "
                    f"Available fields: {sorted(init_field_names)}."
                )
            setattr(step, field_name, value)
        post_init = getattr(step, "__post_init__", None)
        if callable(post_init):
            post_init()
