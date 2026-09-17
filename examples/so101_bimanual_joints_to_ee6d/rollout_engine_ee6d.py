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

"""
Rollout the ee6d SmolVLA checkpoint through the `lerobot-rollout` engine
(build_rollout_context + BaseStrategy + RTC inference), bimanual edition of
`examples/so100_to_so100_EE/rollout.py`. Custom processors cannot be expressed
on the `lerobot-rollout` CLI, so this script mirrors its orchestration
(init_visualization / signal handler / strategy lifecycle) in Python.

Three bimanual kinematic processors are wired in (see bimanual_kinematic_processor.py):

- robot_observation_processor: FK, 12 raw joints -> 14 `left/right_ee.*` (policy input)
- robot_action_processor:      IK, 14 `left/right_ee.*` -> 12 joint commands (policy output)
- teleop_action_processor:     FK action variant; never runs at runtime in the base
                               strategy — its `transform_features` makes the engine's
                               `dataset_features[ACTION]` carry the ee6d names, which is
                               what maps policy output tensors to ee keys.

Units: same "normalized-units-as-degrees" convention as the dataset conversion
(`use_degrees=false`); do not combine with use_degrees=true robot configs.

Run from the repository root (camera/serial/checkpoint come from env vars,
set by rollout_ee6d.sh):

    ./rollout_ee6d.sh
"""

import os

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import RTCInferenceConfig
from lerobot.rollout.strategies import BaseStrategy
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

from bimanual_kinematic_processor import (
    BimanualForwardKinematicsJointsToEEAction,
    BimanualForwardKinematicsJointsToEEObservation,
    BimanualInverseKinematicsEEToJoints,
)

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"环境变量 {name} 未设置,请在 rollout_ee6d.sh 中配置后运行")
    return value


def _env_bool(name: str) -> bool:
    value = _require_env(name).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"环境变量 {name} 必须是 true/false,当前值: {value!r}")


URDF_PATH = "SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

FPS = 30
DURATION_SEC = 360
EXECUTION_HORIZON = 10

CHECKPOINT = _require_env("ROLLOUT_CHECKPOINT")
TASK = _require_env("ROLLOUT_TASK")
LEFT_FOLLOWER_PORT = _require_env("LEFT_FOLLOWER_PORT")
RIGHT_FOLLOWER_PORT = _require_env("RIGHT_FOLLOWER_PORT")
LEFT_WRIST_CAM = _require_env("LEFT_WRIST_CAM")
RIGHT_WRIST_CAM = _require_env("RIGHT_WRIST_CAM")
TOP_CAM = _require_env("TOP_CAM")
DISPLAY_DATA = _env_bool("ROLLOUT_DISPLAY_DATA")

# Same as training (train_config.json): robot keys -> policy camera keys.
RENAME_MAP = {
    "observation.images.top": _require_env("POLICY_CAM_TOP"),
    "observation.images.left_wrist": _require_env("POLICY_CAM_LEFT_WRIST"),
    "observation.images.right_wrist": _require_env("POLICY_CAM_RIGHT_WRIST"),
}


def main():
    init_logging()

    cam = lambda path: OpenCVCameraConfig(index_or_path=path, width=1280, height=720, fps=FPS, fourcc="MJPG")
    robot_config = BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(
            port=LEFT_FOLLOWER_PORT, use_degrees=False, cameras={"wrist": cam(LEFT_WRIST_CAM)}
        ),
        right_arm_config=SOFollowerConfig(
            port=RIGHT_FOLLOWER_PORT, use_degrees=False, cameras={"wrist": cam(RIGHT_WRIST_CAM)}
        ),
        cameras={"top": cam(TOP_CAM)},
        id="bimanual_follower",
    )

    kinematics = {
        side: RobotKinematics(
            urdf_path=URDF_PATH, target_frame_name=TARGET_FRAME, joint_names=ARM_JOINT_NAMES
        )
        for side in ("left", "right")
    }

    robot_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            BimanualForwardKinematicsJointsToEEObservation(
                kinematics=kinematics, motor_names=ARM_JOINT_NAMES
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    robot_ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            BimanualInverseKinematicsEEToJoints(
                kinematics=kinematics, motor_names=ARM_JOINT_NAMES, initial_guess_current_joints=True
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    teleop_joints_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            BimanualForwardKinematicsJointsToEEAction(kinematics=kinematics, motor_names=ARM_JOINT_NAMES)
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    policy_config = PreTrainedConfig.from_pretrained(CHECKPOINT)
    policy_config.pretrained_path = CHECKPOINT

    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=RTCInferenceConfig(rtc=RTCConfig(execution_horizon=EXECUTION_HORIZON)),
        fps=FPS,
        duration=DURATION_SEC,
        task=TASK,
        rename_map=RENAME_MAP,
        display_data=DISPLAY_DATA,
        display_mode="rerun",
    )

    # Same orchestration as `lerobot_rollout.rollout()`: visualization init,
    # signal handler, strategy setup/run/teardown.
    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event,
        teleop_action_processor=teleop_joints_to_ee,
        robot_action_processor=robot_ee_to_joints,
        robot_observation_processor=robot_joints_to_ee,
    )

    if cfg.display_data:
        init_visualization(cfg.display_mode, session_name="rollout", ip=cfg.display_ip, port=cfg.display_port)

    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)


if __name__ == "__main__":
    main()
