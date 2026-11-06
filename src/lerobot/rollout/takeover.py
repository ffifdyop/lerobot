# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Control-thread-only leader feedback and aligned human handover."""

from __future__ import annotations

import math
import time
from threading import Event
from typing import TYPE_CHECKING

from lerobot.common.control_utils import teleop_supports_feedback

if TYPE_CHECKING:
    from lerobot.teleoperators import Teleoperator

    from .configs import TeleopTakeoverConfig
    from .robot_wrapper import ThreadSafeRobot


class AlignmentTimeoutError(TimeoutError):
    """Measured leader/follower error did not converge within the handover deadline."""


class LeaderFollower:
    """Mirror measured joint positions; never write hardware from the command listener."""

    def __init__(
        self, robot: ThreadSafeRobot, teleop: Teleoperator, config: TeleopTakeoverConfig, fps: float
    ) -> None:
        if not teleop_supports_feedback(teleop):
            raise ValueError("Teleop takeover requires leader position feedback and torque control")
        keys = set(robot.action_features)
        if (
            not keys
            or any(not key.endswith(".pos") for key in keys)
            or keys != set(teleop.action_features)
            or keys != set(teleop.feedback_features)
        ):
            raise ValueError("Teleop takeover requires identical leader/follower position joint keys")
        robot.configure_position_hold()
        self.robot = robot
        self.teleop = teleop
        self.config = config
        self.keys = sorted(keys)
        self.period = 1.0 / fps
        self.following = False
        self.failed = False

    def positions(self, values: dict) -> dict[str, float]:
        """Select all joints, rejecting partial/non-finite captures before motor writes."""
        result = {key: float(values[key]) for key in self.keys}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Teleop takeover requires finite joint positions")
        return result

    def sync(self, observation: dict) -> None:
        if self.following:
            self.teleop.send_feedback(self.positions(observation))

    def align(self, cancelled: Event) -> bool:
        """Ramp to the held follower, then verify measured error; False means cancelled.

        A timeout keeps the follower held and is reported separately from I/O failure.
        The caller owns releasing leader torque on cancellation, timeout or exit.
        """
        if cancelled.is_set():
            return False
        current = self.positions(self.teleop.get_action())
        ramp = not self.following
        if ramp:
            # Seed a current-position target BEFORE enabling torque (no stale goal jump).
            self.teleop.send_feedback(current)
            self.following = True
            self.teleop.enable_torque()
        started = time.monotonic()
        while not cancelled.is_set():
            self.robot.hold()
            target = self.positions(self.robot.get_observation())
            elapsed = time.monotonic() - started
            alpha = min(elapsed / self.config.alignment_duration_s, 1.0) if ramp else 1.0
            if cancelled.is_set():
                return False
            self.teleop.send_feedback(
                {key: current[key] + alpha * (target[key] - current[key]) for key in self.keys}
            )
            measured = self.positions(self.teleop.get_action())
            error = max(abs(measured[key] - target[key]) for key in self.keys)
            if alpha == 1.0 and error <= self.config.max_position_error:
                return not cancelled.is_set()
            if elapsed >= self.config.alignment_timeout_s:
                raise AlignmentTimeoutError(f"Leader alignment timed out (max joint error {error:.2f})")
            cancelled.wait(self.period)
        return False

    def release(self) -> None:
        """Release before human control or after a following segment; safe to repeat."""
        if self.following:
            self.teleop.disable_torque()
            self.following = False
