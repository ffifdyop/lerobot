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

"""Rollout a policy served by an openpi policy server on a bimanual SO-101.

Unlike ``lerobot.async_inference`` (gRPC, lerobot-format checkpoints), this client
talks the openpi websocket + msgpack protocol, so it pairs with the server started
by ``openpi/scripts/serve_policy.py`` (see ``~/Dev/Manipulation/openpi/deploy.sh``).

Observations are sent with the raw keys expected by the server's data transforms
(``SO101Inputs`` in openpi): an ``images`` dict with ``cam_high`` / ``cam_left_wrist`` /
``cam_right_wrist`` uint8 (H, W, C) frames, a float32 12-dim ``state`` (left arm 6
joints then right arm 6 joints, in motor order), and the task ``prompt``. Note that
openpi's ``RepackTransform`` only runs in the training dataloader — inference-time
clients must send the post-repack keys directly. All normalization happens
server-side. The server returns an action chunk of shape (action_horizon, 12) which
is executed frame by frame.

Usage example::

    lerobot-openpi-rollout \
        --host=127.0.0.1 --port=8000 \
        --robot.type=bi_so_follower \
        --robot.left_arm_config.port=/dev/ttyACM2 \
        --robot.right_arm_config.port=/dev/ttyACM3 \
        --robot.id=bimanual_follower \
        --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: /dev/video0, width: 1920, height: 1080, fps: 30, fourcc: "MJPG"}, top: {type: opencv, index_or_path: /dev/video6, width: 1920, height: 1080, fps: 30, fourcc: "MJPG"} }' \
        --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: /dev/video8, width: 1920, height: 1080, fps: 30, fourcc: "MJPG"} }' \
        --task="fold the garment on the table" \
        --fps=30 --duration=120
"""

import logging
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (draccus choice registration)
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# Camera name in the server's "images" dict -> robot observation key. Defaults
# match the camera naming used in rollout.sh (left arm carries the top camera).
DEFAULT_CAMERA_KEYS = {
    "cam_high": "left_top",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}


@dataclass
class OpenPIRolloutConfig:
    """CLI config for ``lerobot-openpi-rollout``."""

    robot: RobotConfig
    # openpi policy server address (deploy.sh defaults to port 8000).
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    # Control loop frequency, matching the data collection rate.
    fps: float = 30.0
    # Max rollout duration in seconds; <= 0 runs until interrupted.
    duration: float = 0.0
    # Task instruction sent as the "prompt" key.
    task: str = ""
    # Camera name in the server's "images" dict -> robot observation key.
    camera_keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_KEYS))


def _make_policy_client(cfg: OpenPIRolloutConfig):
    """Connect to the openpi server, failing fast if it is unreachable.

    ``openpi-client`` is an optional dependency and imported lazily.
    """
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as e:
        raise RuntimeError(
            "openpi-client is not installed. Install it with:\n"
            "  uv pip install ~/Dev/Manipulation/openpi/packages/openpi-client"
        ) from e

    # WebsocketClientPolicy retries forever on connection failure; probe the
    # TCP port first so a missing server fails fast with a clear message.
    try:
        with socket.create_connection((cfg.host, cfg.port), timeout=5):
            pass
    except OSError as e:
        raise RuntimeError(
            f"Cannot reach openpi policy server at {cfg.host}:{cfg.port} ({e}).\n"
            "Make sure the server is running, e.g. `bash ~/Dev/Manipulation/openpi/deploy.sh`, "
            "and check --host/--port."
        ) from e

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port, api_key=cfg.api_key)
    logger.info("Connected to openpi server at %s:%s", cfg.host, cfg.port)
    logger.info("Server metadata: %s", client.get_server_metadata())
    return client


def _build_observation(
    robot_obs: dict, joint_keys: list[str], camera_keys: dict[str, str], prompt: str
) -> dict:
    """Assemble the post-repack dict expected by the server's input transforms."""
    missing_cams = [src for src in camera_keys.values() if src not in robot_obs]
    if missing_cams:
        raise KeyError(
            f"Robot observation is missing camera keys {missing_cams}. "
            f"Available keys: {sorted(robot_obs)}. Adjust --camera_keys or the robot camera config."
        )

    return {
        "images": {cam: np.asarray(robot_obs[src_key]) for cam, src_key in camera_keys.items()},
        "state": np.array([robot_obs[k] for k in joint_keys], dtype=np.float32),
        "prompt": prompt,
    }


def _split_action(action: np.ndarray, joint_keys: list[str]) -> dict[str, float]:
    """Split a flat 12-dim action into the robot's keyed action dict."""
    if action.shape[0] != len(joint_keys):
        raise ValueError(
            f"Server returned action dim {action.shape[0]}, expected {len(joint_keys)} "
            f"(left arm + right arm joints)."
        )
    return {key: float(action[i]) for i, key in enumerate(joint_keys)}


@parser.wrap()
def rollout(cfg: OpenPIRolloutConfig):
    init_logging()

    policy_client = _make_policy_client(cfg)

    robot: Robot = make_robot_from_config(cfg.robot)
    robot.connect()
    # action_features keys are "left_<motor>.pos" then "right_<motor>.pos", in
    # motor order — identical to the state/action layout of the training data.
    joint_keys = list(robot.action_features.keys())
    logger.info("Joint order (state/action layout): %s", joint_keys)
    logger.info("Camera mapping: %s", cfg.camera_keys)

    action_queue: deque[np.ndarray] = deque()
    dt_s = 1.0 / cfg.fps
    start_t = time.perf_counter()
    interrupted = False
    n_ticks = 0
    n_chunks = 0
    last_rate_log_t = start_t

    try:
        while True:
            if cfg.duration > 0 and time.perf_counter() - start_t >= cfg.duration:
                logger.info("Duration %.1fs reached, stopping.", cfg.duration)
                break

            loop_start_t = time.perf_counter()

            if not action_queue:
                robot_obs = robot.get_observation()
                obs = _build_observation(robot_obs, joint_keys, cfg.camera_keys, cfg.task)
                infer_start_t = time.perf_counter()
                result = policy_client.infer(obs)
                infer_ms = (time.perf_counter() - infer_start_t) * 1000
                actions = np.asarray(result["actions"])
                if actions.ndim != 2:
                    raise ValueError(f"Expected action chunk of shape (horizon, 12), got {actions.shape}")
                action_queue.extend(actions)
                n_chunks += 1
                logger.info(
                    "Chunk #%d: inference %.0fms (%.1f Hz), chunk of %d frames covers %.2fs @ %.0f fps",
                    n_chunks,
                    infer_ms,
                    1000 / infer_ms,
                    len(actions),
                    len(actions) * dt_s,
                    cfg.fps,
                )

            action = action_queue.popleft()
            robot.send_action(_split_action(action, joint_keys))
            n_ticks += 1

            precise_sleep(max(dt_s - (time.perf_counter() - loop_start_t), 0.0))

            now = time.perf_counter()
            if now - last_rate_log_t >= 5.0:
                logger.info(
                    "Control loop: %.1f Hz avg over last %.1fs (%d ticks, %d chunk requests)",
                    n_ticks / (now - start_t),
                    now - last_rate_log_t,
                    n_ticks,
                    n_chunks,
                )
                last_rate_log_t = now
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Interrupted by user, stopping.")
    finally:
        # Never send further actions past this point; always release the robot.
        if interrupted:
            logger.info("Stopping action sending and disconnecting robot.")
        robot.disconnect()

    logger.info("Rollout finished")


def main():
    """CLI entry point for ``lerobot-openpi-rollout``."""
    register_third_party_plugins()
    try:
        rollout()
    except RuntimeError as e:
        # Covers missing openpi-client and unreachable server: fail before any
        # hardware is touched, with an actionable message.
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
