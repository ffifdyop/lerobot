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
Bimanual (left_/right_ prefixed) joints <-> ee6d processor steps for bi_so_follower robots.

The single-arm steps in `lerobot.robots.so_follower.robot_kinematic_processor` hardcode the
`ee.*` / `{motor}.pos` key convention and cannot be chained for two arms (the second step's
`ee.x` lookup collides with the first). These steps handle both arms in one pass:

- observation/state: `left_shoulder_pan.pos` ... `right_gripper.pos` (12) ->
  `left_ee.{x,y,z,wx,wy,wz,gripper_pos}` + `right_ee.{...}` (14)
- action: the reverse via IK, gripper passed through unchanged.

Convention matches `convert_joints_to_ee6d.py` in this directory: position in the arm's own
base frame, orientation as rotation vector, gripper as the raw joint value. The joint values
are handed to placo as-is — the dataset was recorded with `use_degrees=false`
(RANGE_M100_100 normalized units), so FK/IK deliberately reuse that "wrong-but-consistent"
unit convention; the policy was trained in that space and IK inverts the same FK.
"""

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model import RobotKinematics
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation

ARM_SIDES = ("left", "right")
EE_COMPONENTS = ("x", "y", "z", "wx", "wy", "wz", "gripper_pos")
EE_NAMES = [f"{side}_ee.{comp}" for side in ARM_SIDES for comp in EE_COMPONENTS]


def joints_to_ee6d(kinematics: RobotKinematics, q: np.ndarray) -> list[float]:
    """6 arm joint values -> [x, y, z, wx, wy, wz, gripper_pos] (gripper passed through)."""
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    rotvec = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2]), float(q[5])]


@ProcessorStepRegistry.register("bimanual_forward_kinematics_joints_to_ee_observation")
@dataclass
class BimanualForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """Replace the 12 `{side}_{motor}.pos` observation keys with 14 `{side}_ee.*` keys."""

    kinematics: dict[str, RobotKinematics]  # {"left": ..., "right": ...}
    motor_names: list[str]  # e.g. ["shoulder_pan", ..., "gripper"]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        obs = dict(observation)
        for side in ARM_SIDES:
            q = np.array([obs.pop(f"{side}_{m}.pos") for m in self.motor_names], dtype=float)
            ee = joints_to_ee6d(self.kinematics[side], q)
            for comp, value in zip(EE_COMPONENTS, ee, strict=True):
                obs[f"{side}_ee.{comp}"] = value
        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for side in ARM_SIDES:
            for m in self.motor_names:
                features[PipelineFeatureType.OBSERVATION].pop(f"{side}_{m}.pos", None)
            for comp in EE_COMPONENTS:
                features[PipelineFeatureType.OBSERVATION][f"{side}_ee.{comp}"] = PolicyFeature(
                    type=FeatureType.STATE, shape=(1,)
                )
        return features


@ProcessorStepRegistry.register("bimanual_forward_kinematics_joints_to_ee_action")
@dataclass
class BimanualForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """Replace the 12 `{side}_{motor}.pos` action keys with 14 `{side}_ee.*` keys.

    At rollout time only `transform_features` matters: mounting this step as the
    `teleop_action_processor` makes `dataset_features[ACTION]` carry the ee6d names, so
    the rollout engine maps policy output tensors to `left_ee.*`/`right_ee.*` keys that
    `BimanualInverseKinematicsEEToJoints` (the `robot_action_processor`) consumes.
    """

    kinematics: dict[str, RobotKinematics]  # {"left": ..., "right": ...}
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        action = dict(action)
        for side in ARM_SIDES:
            q = np.array([action.pop(f"{side}_{m}.pos") for m in self.motor_names], dtype=float)
            ee = joints_to_ee6d(self.kinematics[side], q)
            for comp, value in zip(EE_COMPONENTS, ee, strict=True):
                action[f"{side}_ee.{comp}"] = value
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for side in ARM_SIDES:
            for m in self.motor_names:
                features[PipelineFeatureType.ACTION].pop(f"{side}_{m}.pos", None)
            for comp in EE_COMPONENTS:
                features[PipelineFeatureType.ACTION][f"{side}_ee.{comp}"] = PolicyFeature(
                    type=FeatureType.ACTION, shape=(1,)
                )
        return features


@ProcessorStepRegistry.register("bimanual_inverse_kinematics_ee_to_joints")
@dataclass
class BimanualInverseKinematicsEEToJoints(RobotActionProcessorStep):
    """Translate 14 `{side}_ee.*` action keys back to 12 `{side}_{motor}.pos` joint commands.

    The current joint positions (read from the transition's observation) seed the IK solver;
    the gripper channel bypasses IK and is forwarded verbatim.
    """

    kinematics: dict[str, RobotKinematics]  # {"left": ..., "right": ...}
    motor_names: list[str]
    q_curr: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    initial_guess_current_joints: bool = True
    orientation_weight: float = 0.01

    def action(self, action: RobotAction) -> RobotAction:
        action = dict(action)
        raw_observation = self.transition.get(TransitionKey.OBSERVATION)
        if raw_observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        for side in ARM_SIDES:
            values = [action.pop(f"{side}_ee.{comp}", None) for comp in EE_COMPONENTS]
            if any(v is None for v in values):
                raise ValueError(
                    f"Missing end-effector components for {side} arm: "
                    + ", ".join(f"{side}_ee.{c}" for c in EE_COMPONENTS)
                )
            x, y, z, wx, wy, wz, gripper_pos = (float(v) for v in values)

            q_obs = np.array(
                [float(raw_observation[f"{side}_{m}.pos"]) for m in self.motor_names], dtype=float
            )
            if self.initial_guess_current_joints or side not in self.q_curr:
                self.q_curr[side] = q_obs

            t_des = np.eye(4, dtype=float)
            t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            t_des[:3, 3] = [x, y, z]

            q_target = self.kinematics[side].inverse_kinematics(
                self.q_curr[side], t_des, orientation_weight=self.orientation_weight
            )
            self.q_curr[side] = q_target

            for i, name in enumerate(self.motor_names):
                if name != "gripper":
                    action[f"{side}_{name}.pos"] = float(q_target[i])
                else:
                    action[f"{side}_gripper.pos"] = gripper_pos
        return action

    def reset(self):
        self.q_curr.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for side in ARM_SIDES:
            for comp in EE_COMPONENTS:
                features[PipelineFeatureType.ACTION].pop(f"{side}_ee.{comp}", None)
            for m in self.motor_names:
                features[PipelineFeatureType.ACTION][f"{side}_{m}.pos"] = PolicyFeature(
                    type=FeatureType.ACTION, shape=(1,)
                )
        return features
