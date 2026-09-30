# !/usr/bin/env python

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
Rollout a SmolVLA checkpoint trained on the bimanual ee6d dataset
(`ffifdyop/so101_bimanual_folding-cloth-720p_ee6d`) on the bi_so_follower robot.

Self-contained control loop (no lerobot-rollout engine): per tick the 12 joint
observations are mapped to 14 ee6d values by forward kinematics, the policy produces
ee6d action chunks, and each action is mapped back to joint commands by inverse
kinematics. A new chunk is computed every EXECUTION_HORIZON steps (open-loop horizon,
same spirit as `--inference.rtc.execution_horizon`).

Why not `lerobot-rollout`: the engine derives action key names from robot hardware
features (12 joint keys), which cannot express the policy's 14 ee6d action names;
wiring the bimanual FK/IK processors through `build_rollout_context` would need a
matching teleop_action_processor just to fix key naming. This loop keeps the key
spaces explicit instead.

Units: the dataset was recorded with `use_degrees=false` (RANGE_M100_100 normalized
units) and the ee6d conversion fed those values to the kinematics solver unchanged.
The FK here repeats that exact convention, so observation/action spaces line up with
training; ee "meters" are therefore not physical meters, but FK -> policy -> IK is
self-consistent. Do not mix this checkpoint with `use_degrees=true` robot configs.

Run from the repository root:

    uv run python examples/so101_bimanual_joints_to_ee6d/rollout_smolvla_ee6d.py
"""

import logging
import time
from collections import deque

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from bimanual_kinematic_processor import (
    EE_NAMES,
    BimanualForwardKinematicsJointsToEEObservation,
    BimanualInverseKinematicsEEToJoints,
)

CHECKPOINT = "models/lehome/train/smolvla/smolvla-real_720p_ft_ee6d_20260914_191954/checkpoints/080000/pretrained_model"
URDF_PATH = "SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

FPS = 30
DURATION_SEC = 360
EXECUTION_HORIZON = 10  # recompute an action chunk every N ticks, execute its first N actions
TASK = "fold the garment on the table"

# Use /dev/serial/by-id/ (adapter serial), never /dev/ttyACM<N>: those numbers are
# assigned in attach order and get reused, so a re-plug can silently swap the two
# follower arms. When that happens lerobot validates each arm's motors against the
# *other* arm's calibration file, `is_calibrated` fails, and connect() prompts to
# recalibrate — answering it writes the wrong homing offsets to the motors.
LEFT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042915-if00"
RIGHT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D046378-if00"
LEFT_WRIST_CAM = "/dev/video0"
RIGHT_WRIST_CAM = "/dev/video8"
TOP_CAM = "/dev/video6"

# build_dataset_frame selects/converts values by this spec; image shapes only document the
# camera stream, the model resizes internally (resize_imgs_with_padding=512).
OBS_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": [len(EE_NAMES)], "names": EE_NAMES},
    "observation.images.top": {"dtype": "video", "shape": [720, 1280, 3], "names": ["height", "width", "channels"]},
    "observation.images.left_wrist": {"dtype": "video", "shape": [720, 1280, 3], "names": ["height", "width", "channels"]},
    "observation.images.right_wrist": {"dtype": "video", "shape": [720, 1280, 3], "names": ["height", "width", "channels"]},
}


def make_robot() -> BiSOFollower:
    cam = lambda path: OpenCVCameraConfig(index_or_path=path, width=1280, height=720, fps=FPS, fourcc="MJPG")
    cfg = BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(
            port=LEFT_FOLLOWER_PORT, use_degrees=False, cameras={"wrist": cam(LEFT_WRIST_CAM)}
        ),
        right_arm_config=SOFollowerConfig(
            port=RIGHT_FOLLOWER_PORT, use_degrees=False, cameras={"wrist": cam(RIGHT_WRIST_CAM)}
        ),
        cameras={"top": cam(TOP_CAM)},
        id="bimanual_follower",
    )
    return BiSOFollower(cfg)


def main():
    init_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CHECKPOINT
    )

    kinematics = {
        side: RobotKinematics(
            urdf_path=URDF_PATH, target_frame_name=TARGET_FRAME, joint_names=ARM_JOINT_NAMES
        )
        for side in ("left", "right")
    }
    joints_to_ee = BimanualForwardKinematicsJointsToEEObservation(
        kinematics=kinematics, motor_names=ARM_JOINT_NAMES
    )
    ee_to_joints = BimanualInverseKinematicsEEToJoints(
        kinematics=kinematics, motor_names=ARM_JOINT_NAMES, initial_guess_current_joints=True
    )

    robot = make_robot()
    robot.connect()

    action_queue: deque[torch.Tensor] = deque()
    logging.info("Starting rollout (task='%s', %ds at %d Hz)...", TASK, DURATION_SEC, FPS)
    try:
        start_t = time.perf_counter()
        timestamp = 0.0
        while timestamp < DURATION_SEC:
            loop_t = time.perf_counter()

            obs_raw = robot.get_observation()
            obs_ee = joints_to_ee.observation(obs_raw)

            if not action_queue:
                frame = build_dataset_frame(OBS_FEATURES, obs_ee, prefix="observation")
                frame = prepare_observation_for_inference(frame, device, TASK, robot.name)
                batch = preprocessor(frame)
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(batch)
                chunk = postprocessor(chunk).squeeze(0).cpu()
                action_queue.extend(chunk[:EXECUTION_HORIZON])

            action_ee = action_queue.popleft().numpy()
            action_dict = {name: float(v) for name, v in zip(EE_NAMES, action_ee, strict=True)}
            # IK seeds from the current raw joints, which travel in the transition's observation.
            transition = ee_to_joints(
                {TransitionKey.OBSERVATION: obs_raw, TransitionKey.ACTION: action_dict}
            )
            robot.send_action(transition[TransitionKey.ACTION])

            dt = time.perf_counter() - loop_t
            if dt > 1 / FPS:
                logging.warning("Control loop running slower (%.1f Hz) than target %d Hz", 1 / dt, FPS)
            precise_sleep(max(1 / FPS - dt, 0.0))
            timestamp = time.perf_counter() - start_t
    finally:
        robot.disconnect()
        logging.info("Robot disconnected, rollout done.")


if __name__ == "__main__":
    main()
