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

"""DM05 remote policy client as a LeRobot policy plugin.

The weights live in an OpenDM inference server (``POST /v1/infer``, started by
``/data/opendm/script/deploy.sh``); this class only encodes observations into
the server's JSON schema and executes the returned action chunk. With
``--inference.type=sync`` the control loop calls :meth:`select_action` every
tick, but the HTTP request only fires when the local queue is empty: the server
returns a whole chunk (e.g. 50 frames), which is truncated to
``execution_horizon``, enqueued and consumed one frame per tick.

Units: state and actions are SO-101 "percent" values on both sides of the wire
(the training data was converted verbatim from the LeRobot dataset); no
conversion happens here. Normalization also lives server-side.

History images: with ``use_history=true`` a deque of main-camera frames is sent
with each request (oldest first, current frame excluded — it is appended after
the request). The queue drains every ``execution_horizon`` ticks, so the
history-frame interval automatically matches the re-planning cadence.
"""

import base64
import logging
import socket
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_dm05 import DM05Config

logger = logging.getLogger(__name__)

INFER_PATH = "/v1/infer"
# Seconds to wait when probing server reachability at policy construction.
CONNECT_TIMEOUT_S = 5.0


class DM05Policy(PreTrainedPolicy):
    config_class = DM05Config
    name = "dm05"

    def __init__(self, config: DM05Config, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        # Fail fast, before the rollout context touches any hardware (policies load
        # first in build_rollout_context).
        try:
            with socket.create_connection((config.host, config.port), timeout=CONNECT_TIMEOUT_S):
                pass
        except OSError as e:
            raise RuntimeError(
                f"Cannot reach DM05 policy server at {config.host}:{config.port} ({e}). "
                "Make sure the server is running, e.g. `bash /data/opendm/script/deploy.sh`, "
                "and check the host/port in the policy config."
            ) from e
        logger.info("DM05 policy server reachable at %s:%d", config.host, config.port)

        # Must be named `_action_queue`: the base class `drop_queued_actions`
        # (task switches in interactive rollouts) clears exactly this attribute.
        self._action_queue: deque[torch.Tensor] = deque()
        self._history_frames: deque[str] = (
            deque(maxlen=config.history_max_frames) if config.use_history else deque(maxlen=0)
        )

    # ------------------------------------------------------------------
    # Loading: config-only, weights stay on the server
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        revision: str | None = None,
        **kwargs,
    ):
        """Build the client from the config directory, loading no weights.

        The base implementation unconditionally loads ``model.safetensors`` for a
        local directory; a DM05 deployment dir holds only the connection config.
        """
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, revision=revision, **kwargs)
        logger.info("DM05 weights live on the remote server; skipping local weight load.")
        return cls(config)

    # ------------------------------------------------------------------
    # Base-class contracts
    # ------------------------------------------------------------------

    def get_optim_params(self) -> dict:
        return {}

    def supports_text_generation(self) -> bool:
        return False

    def reset(self):
        """Clear episode-scoped state: the pending action chunk and history frames."""
        self._action_queue.clear()
        self._history_frames.clear()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        # Inference-only remote policy; a local forward pass has no meaning.
        raise NotImplementedError("DM05 runs on the remote server; use select_action/predict_action_chunk.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the next action, requesting a fresh chunk only when the queue is empty."""
        if not self._action_queue:
            actions = self.predict_action_chunk(batch)
            # (1, horizon, action_dim) -> one (1, action_dim) entry per tick.
            for t in range(actions.shape[1]):
                self._action_queue.append(actions[:, t])
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """POST one inference request; return the chunk as (1, horizon, action_dim)."""
        observation = self._encode_observation(batch)
        history_payload = list(self._history_frames) if self.config.use_history else None

        actions = self._request(observation, history_payload)

        if self.config.use_history:
            # Append after the request: history excludes the current frame, and a
            # failed request must not pollute the history.
            self._history_frames.append(self._encode_image(batch[self.config.camera_keys["1"]]))

        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.config.output_action_dim:
            raise RuntimeError(
                f"DM05 server returned actions of shape {chunk.shape}, expected (horizon, "
                f"{self.config.output_action_dim})."
            )
        horizon = self.config.execution_horizon
        if horizon is not None:
            chunk = chunk[:horizon]
        return torch.from_numpy(chunk).unsqueeze(0)

    # ------------------------------------------------------------------
    # Encoding / transport
    # ------------------------------------------------------------------

    def _encode_observation(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Build the ``observation`` JSON object of the /v1/infer request body."""
        missing = [key for key in self.config.camera_keys.values() if key not in batch]
        if missing:
            raise KeyError(
                f"Observation batch is missing camera keys {missing}. Available: {sorted(batch)}. "
                "Adjust camera_keys in the policy config or the --rename_map."
            )
        if "observation.state" not in batch:
            raise KeyError(f"Observation batch is missing 'observation.state'. Available: {sorted(batch)}.")

        images = {slot: self._encode_image(batch[key]) for slot, key in self.config.camera_keys.items()}
        state = batch["observation.state"].detach().clone().squeeze(0).to(torch.float32).cpu().numpy()
        prompt = batch.get("task", "")
        if isinstance(prompt, list):
            prompt = prompt[0]
        return {
            "prompt": str(prompt),
            "state": state.tolist(),
            "images": images,
        }

    @staticmethod
    def _encode_image(image: torch.Tensor) -> str:
        """float32 CHW [0,1] batched tensor -> base64 JPEG (uint8 HWC)."""
        frame = image.detach().clone().squeeze(0).cpu()
        frame = (frame.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode the camera frame as JPEG.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _request(self, observation: dict[str, Any], history_payload: list[str] | None) -> list:
        """POST /v1/infer; map every transport failure to a RuntimeError."""
        url = f"http://{self.config.host}:{self.config.port}{INFER_PATH}"
        if history_payload is not None:
            observation = {**observation, "history_images": history_payload}
        try:
            response = requests.post(url, json={"observation": observation}, timeout=self.config.timeout)
        except requests.RequestException as e:
            raise RuntimeError(
                f"DM05 server request to {url} failed ({type(e).__name__}: {e}). "
                "Stopping the rollout rather than executing stale actions."
            ) from e
        if response.status_code != 200:
            raise RuntimeError(
                f"DM05 server at {url} returned HTTP {response.status_code}: "
                f"{response.text[:200]!r}. Stopping the rollout."
            )
        try:
            payload = response.json()
        except ValueError as e:
            raise RuntimeError(f"DM05 server at {url} returned invalid JSON: {response.text[:200]!r}") from e
        if "actions" not in payload:
            raise RuntimeError(f"DM05 server response is missing 'actions': {str(payload)[:200]!r}")
        return payload["actions"]
