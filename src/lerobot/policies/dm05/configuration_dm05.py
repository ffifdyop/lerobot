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

"""Configuration for the DM05 remote policy client.

``dm05`` is not a local model: the weights live in an OpenDM inference server
(started by ``/data/opendm/script/deploy.sh``) exposing ``POST /v1/infer``. The
``--policy.path`` directory therefore holds only this config — host/port plus
encoding knobs — never any safetensors. All normalization happens server-side
(norm_stats ship with the checkpoint), so the client's processor pipeline is
identity-like and the normalization mapping below is never applied.
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# Server-side image slots ("Head", "Left wrist", "Right wrist" in the deployment's
# --inference-config.image-prompts) -> observation keys after the rename step.
DEFAULT_CAMERA_KEYS = {
    "1": f"{OBS_IMAGES}.top",
    "2": f"{OBS_IMAGES}.left_wrist",
    "3": f"{OBS_IMAGES}.right_wrist",
}


@PreTrainedConfig.register_subclass("dm05")
@dataclass
class DM05Config(PreTrainedConfig):
    host: str = "127.0.0.1"
    port: int = 7891
    # Per-request timeout in seconds; a chunk request that exceeds it stops the rollout.
    timeout: float = 10.0

    # Server image slot ("1"/"2"/"3") -> post-rename observation key. The values must
    # match the keys the preprocessor emits: if --rename_map renames cameras (e.g.
    # top -> camera1), point the slots at the renamed keys.
    camera_keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_KEYS))

    # Frames of each returned chunk to actually enqueue and execute; the rest of the
    # chunk is discarded and a fresh request fires once the queue drains. None executes
    # the whole chunk. Re-planning cadence — and, with use_history, the history-frame
    # interval — equals this value.
    execution_horizon: int | None = None

    # Send main-camera history frames with each request (server must run with
    # --data-config.is-history). Old checkpoints' servers reject the field with 400.
    use_history: bool = False
    history_max_frames: int = 5

    # Action dimensionality returned by the server (deploy --inference-config.output_action_dim).
    output_action_dim: int = 12

    # Joint order of state/actions in the training data (e.g. ["left_shoulder.pos", ...]).
    # Populated at runtime from dataset metadata by make_policy; for rollout it is read
    # from config.json and drives the hardware feature alignment in the rollout context.
    action_feature_names: list[str] | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,  # normalized server-side
            "ACTION": NormalizationMode.IDENTITY,  # normalized server-side
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if not self.host:
            raise ValueError("host must be a non-empty string")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if not self.camera_keys:
            raise ValueError("camera_keys must map at least one server slot to an observation key")
        if self.execution_horizon is not None and self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be None or a positive int, got {self.execution_horizon}")
        if self.history_max_frames < 0:
            raise ValueError(f"history_max_frames must be >= 0, got {self.history_max_frames}")
        if self.output_action_dim <= 0:
            raise ValueError(f"output_action_dim must be positive, got {self.output_action_dim}")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        # Placeholder: a remote policy is never trained locally.
        return AdamWConfig()

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        """Declare the visual/state inputs and the action output the server speaks."""
        if not self.input_features:
            self.input_features = {
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in self.camera_keys.values()
            }
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.output_action_dim,)
            )
        if not self.output_features:
            self.output_features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.output_action_dim,))
            }
