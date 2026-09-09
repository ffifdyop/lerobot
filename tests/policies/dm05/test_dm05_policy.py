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

"""Unit tests for the DM05 remote policy client.

All HTTP traffic is mocked: no DM05 server is needed. The observation batch fed
to ``select_action`` mimics the output of the identity-like preprocessor
(``prepare_observation_for_inference`` + rename/batch/device steps): float32
CHW [0,1] image tensors with a leading batch dim and a (1, 12) state tensor.
"""

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import requests
import torch

from lerobot.policies.dm05.configuration_dm05 import DM05Config
from lerobot.policies.dm05.modeling_dm05 import DM05Policy

CHUNK = 50
ACTION_DIM = 12


def _make_config(**overrides) -> DM05Config:
    kwargs = {"host": "127.0.0.1", "port": 7891}
    kwargs.update(overrides)
    return DM05Config(**kwargs)


def _make_policy(config: DM05Config) -> DM05Policy:
    with patch("socket.create_connection", return_value=MagicMock()):
        return DM05Policy(config)


def _make_batch(state_value: float = 0.5, height: int = 16, width: int = 24) -> dict[str, torch.Tensor]:
    """Mimic the preprocessed observation the sync engine hands to select_action."""
    batch = {
        "observation.state": torch.full((1, ACTION_DIM), state_value, dtype=torch.float32),
        "task": ["fold the garment on the table"],
    }
    for key in ("top", "left_wrist", "right_wrist"):
        batch[f"observation.images.{key}"] = torch.rand(1, 3, height, width)
    return batch


class _FakeResponse:
    def __init__(self, actions, status_code=200):
        self._actions = actions
        self.status_code = status_code
        self.text = "fake body"

    def json(self):
        return {"actions": self._actions, "metadata": {"latency_ms": 1.0}}


def _make_actions(chunk_size: int = CHUNK, value: float = 0.0) -> list[list[float]]:
    """Distinct values per frame so tests can tell which chunk frame executes."""
    return [[value + t / 100.0] * ACTION_DIM for t in range(chunk_size)]


@pytest.fixture
def mock_post():
    """Patch requests.post; the test drives behavior through `calls` and `responses`."""
    calls = []
    responses = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if not responses:
            actions = _make_actions()
            responses.append(_FakeResponse(actions))
        return responses.pop(0)

    with patch.object(requests, "post", side_effect=fake_post):
        yield calls


class TestRegistration:
    def test_dm05_registered_and_resolvable(self):
        from lerobot.configs import PreTrainedConfig

        assert "dm05" in PreTrainedConfig.get_known_choices()
        from lerobot.policies import get_policy_class

        assert get_policy_class("dm05") is DM05Policy

    def test_config_validation_rejects_bad_values(self):
        with pytest.raises(ValueError):
            DM05Config(port=0)
        with pytest.raises(ValueError):
            DM05Config(execution_horizon=0)
        with pytest.raises(ValueError):
            DM05Config(camera_keys={})

    def test_fail_fast_when_server_unreachable(self):
        with patch("socket.create_connection", side_effect=OSError("refused")), pytest.raises(RuntimeError, match="deploy.sh"):
                DM05Policy(_make_config())


class TestRequestEncoding:
    def test_request_body_fields(self, mock_post):
        policy = _make_policy(_make_config())
        policy.select_action(_make_batch(state_value=0.25))

        assert len(mock_post) == 1
        body = mock_post[0]["json"]
        assert mock_post[0]["url"].endswith("/v1/infer")
        observation = body["observation"]
        assert observation["prompt"] == "fold the garment on the table"
        assert len(observation["state"]) == ACTION_DIM
        assert observation["state"][0] == pytest.approx(0.25)
        # Slots match the config's camera_keys, values decode as JPEG.
        assert sorted(observation["images"].keys()) == ["1", "2", "3"]
        for _slot, b64 in observation["images"].items():
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            assert frame is not None
            assert frame.shape == (16, 24, 3)
        assert "history_images" not in observation

    def test_missing_camera_key_raises(self, mock_post):
        policy = _make_policy(_make_config())
        batch = _make_batch()
        del batch["observation.images.left_wrist"]
        with pytest.raises(KeyError, match="left_wrist"):
            policy.select_action(batch)

    def test_history_images_sent_when_enabled(self, mock_post):
        config = _make_config(use_history=True)
        policy = _make_policy(config)
        batch = _make_batch()
        policy.select_action(batch)  # request 1: history empty
        body1 = mock_post[0]["json"]["observation"]
        assert body1["history_images"] == []
        # Requests 2..5: one appended main-camera frame each, oldest first.
        for expected_len in range(1, 5):
            for _ in range(CHUNK):  # drain the queue so a new request fires
                policy.select_action(batch)
            body = mock_post[-1]["json"]["observation"]
            assert len(body["history_images"]) == expected_len
            # History comes only from the main camera (slot "1" -> top).
            for b64 in body["history_images"]:
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                assert frame is not None
        # Request 7: deque saturated at history_max_frames=5.
        for _ in range(CHUNK):
            policy.select_action(batch)
        assert len(mock_post[-1]["json"]["observation"]["history_images"]) == 5

    def test_history_cleared_on_reset(self, mock_post):
        policy = _make_policy(_make_config(use_history=True))
        batch = _make_batch()
        policy.select_action(batch)
        policy.reset()
        for _ in range(CHUNK):  # fresh chunk after reset
            policy.select_action(batch)
        assert mock_post[-1]["json"]["observation"]["history_images"] == []


class TestChunkExecution:
    def test_whole_chunk_served_by_single_request(self, mock_post):
        policy = _make_policy(_make_config())
        batch = _make_batch()
        for t in range(CHUNK):
            action = policy.select_action(batch)
            assert action.shape == (1, ACTION_DIM)
            assert action[0, 0].item() == pytest.approx(t / 100.0)
        assert len(mock_post) == 1
        policy.select_action(batch)
        assert len(mock_post) == 2

    def test_execution_horizon_truncates_chunk(self, mock_post):
        policy = _make_policy(_make_config(execution_horizon=25))
        batch = _make_batch()
        for t in range(25):
            action = policy.select_action(batch)
            assert action[0, 0].item() == pytest.approx(t / 100.0)
        assert len(mock_post) == 1
        policy.select_action(batch)  # frame 25 of the first chunk is discarded
        assert len(mock_post) == 2

    def test_drop_queued_actions_forces_fresh_request(self, mock_post):
        policy = _make_policy(_make_config())
        batch = _make_batch()
        policy.select_action(batch)
        policy.drop_queued_actions()
        action = policy.select_action(batch)  # fresh chunk, frame 0 again
        assert action[0, 0].item() == pytest.approx(0.0)
        assert len(mock_post) == 2

    def test_wrong_action_dim_raises(self):
        policy = _make_policy(_make_config())
        bad = _FakeResponse([[0.0] * 7] * CHUNK)  # 7 != 12
        with patch.object(requests, "post", return_value=bad), pytest.raises(RuntimeError, match="horizon, 12"):
                policy.select_action(_make_batch())


class TestNetworkFailures:
    def test_connection_error_raises_runtime(self):
        policy = _make_policy(_make_config())
        with patch.object(requests, "post", side_effect=requests.ConnectionError("boom")), pytest.raises(RuntimeError, match="127.0.0.1:7891"):
                policy.select_action(_make_batch())

    def test_http_500_raises_runtime(self):
        policy = _make_policy(_make_config())
        with patch.object(requests, "post", return_value=_FakeResponse([], status_code=500)), pytest.raises(RuntimeError, match="500"):
                policy.select_action(_make_batch())

    def test_missing_actions_key_raises_runtime(self):
        policy = _make_policy(_make_config())

        class NoActions(_FakeResponse):
            def json(self):
                return {"metadata": {}}

        with patch.object(requests, "post", return_value=NoActions([])), pytest.raises(RuntimeError, match="actions"):
                policy.select_action(_make_batch())

    def test_failed_request_does_not_pollute_history(self, mock_post):
        policy = _make_policy(_make_config(use_history=True))
        batch = _make_batch()
        policy.select_action(batch)  # history now has 1 frame
        # Make the next request fail mid-episode.
        with patch.object(requests, "post", side_effect=requests.Timeout("slow")), pytest.raises(RuntimeError):
                for _ in range(CHUNK):
                    policy.select_action(batch)
        # The failed request must not have appended its main-camera frame.
        for _ in range(CHUNK):
            policy.select_action(batch)
        assert len(mock_post[-1]["json"]["observation"]["history_images"]) == 1


class TestFromPretrained:
    def test_config_only_load_skips_weights(self, tmp_path, monkeypatch):
        import json

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "type": "dm05",
                    "host": "10.0.0.5",
                    "port": 8000,
                    "execution_horizon": 25,
                    "camera_keys": {"1": "observation.images.camera1"},
                }
            )
        )
        with patch("socket.create_connection", return_value=MagicMock()):
            policy = DM05Policy.from_pretrained(str(tmp_path))
        assert policy.config.host == "10.0.0.5"
        assert policy.config.port == 8000
        assert policy.config.execution_horizon == 25
        assert policy.config.camera_keys == {"1": "observation.images.camera1"}
