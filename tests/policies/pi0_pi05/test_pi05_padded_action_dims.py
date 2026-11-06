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

"""Checkpoints saved by the OpenPI-aligned SO-101 recipe store
``train_on_padded_action_dims``; loading them must not fail."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


def test_config_loads_checkpoint_with_padded_action_dims(tmp_path):
    # Reproduces loading pi05 SO-101 checkpoints: the serialized config carries a
    # field the local dataclass must accept, otherwise draccus rejects the file.
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "pi05", "train_on_padded_action_dims": True})
    )
    config = PI05Config.from_pretrained(tmp_path)
    assert config.train_on_padded_action_dims is True


def test_config_defaults_to_legacy_action_dim_loss():
    assert PI05Config().train_on_padded_action_dims is False


class _ElementwiseLossModel:
    def __init__(self):
        self.loss_values = nn.Parameter(torch.ones(32))

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.zeros(batch_size, device=device)

    def forward(self, *args, **kwargs):
        actions = args[4]
        return self.loss_values.view(1, 1, -1).expand(actions.shape[0], actions.shape[1], -1)


def _loss_policy(train_on_padded_action_dims):
    config = SimpleNamespace(
        max_action_dim=32,
        output_features={ACTION: SimpleNamespace(shape=(12,))},
        use_proprioceptive_memory=False,
        rtc_training_max_delay=0,
        train_on_padded_action_dims=train_on_padded_action_dims,
    )
    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.anchor = nn.Parameter(torch.zeros(()))
    policy.model = _ElementwiseLossModel()
    policy._preprocess_images = lambda batch: ([], [])
    batch = {
        ACTION: torch.zeros(1, 2, 12),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }
    return policy, batch


def test_action_loss_switch_controls_padding_dimensions_and_gradients():
    padded_policy, padded_batch = _loss_policy(True)
    padded_loss, padded_info = padded_policy(padded_batch)
    padded_loss.backward()

    legacy_policy, legacy_batch = _loss_policy(False)
    legacy_loss, legacy_info = legacy_policy(legacy_batch)
    legacy_loss.backward()

    assert padded_info["loss"] == pytest.approx(1.0)
    assert legacy_info["loss"] == pytest.approx(1.0)
    assert torch.all(padded_policy.model.loss_values.grad[12:] > 0)
    assert torch.all(legacy_policy.model.loss_values.grad[12:] == 0)
    assert padded_policy.model.loss_values.grad[0] == pytest.approx(1 / 32)
    assert legacy_policy.model.loss_values.grad[0] == pytest.approx(1 / 12)
