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

"""Action interpolation for smoother robot control.

Provides configurable Nx control rate by interpolating between consecutive actions.
Useful with RTC and action-chunking policies to reduce jerkiness.
"""

from torch import Tensor


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Usage:
        interpolator = ActionInterpolator(multiplier=2)  # 2x control rate

        # In control loop:
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)

        # Recording stays at the base FPS: only the tick that emits the
        # policy's own action contributes a dataset frame.
        if interpolator.emitted_policy_action:
            dataset.add_frame(...)
    """

    def __init__(self, multiplier: int = 1):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.)
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0
        self._emitted_policy_action = False
        # The most recent action actually handed to the robot, kept across a flush so a
        # View Mask transition can hold position instead of coasting on stale actions.
        self._last_sent: Tensor | None = None

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    @property
    def emitted_policy_action(self) -> bool:
        """Whether the action last returned by :meth:`get` was the policy's own output.

        :meth:`add` stores the policy action last in the interpolated buffer, so this
        is ``True`` exactly on the tick that emits ``buffer[-1]`` and ``False`` on the
        intermediate ticks leading up to it.  Strategies gate dataset recording on it:
        frames then land at ``fps`` regardless of ``multiplier``, and each one stores a
        genuine policy action paired with the observation that produced it.

        Read this *after* :meth:`get` (or after ``send_next_action``) — it describes
        the action already handed out, not the one the next call will return.
        :meth:`needs_new_action` is the question to ask *before* dispatching; reading
        that one instead would record ``buffer[0]``, the least-advanced intermediate.
        """
        return self._emitted_policy_action

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0
        self._emitted_policy_action = False
        self._last_sent = None

    def flush(self) -> Tensor | None:
        """Drop the not-yet-sent interpolated tail, keeping the last sent action as hold.

        Used by a View Mask transition: the remaining interpolated steps were derived
        from actions generated under the previous mask, so continuing to emit them
        would move the robot along a stale trajectory.  The robot instead holds the
        last action it actually received (:meth:`last_sent`) until a fresh one
        arrives; a later :meth:`add` interpolates from that same value, so motion
        resumes continuously rather than jumping.

        Returns the held action, or ``None`` if nothing was ever sent.
        """
        # ``_prev`` is the endpoint of the whole chunk that was most recently
        # added.  It may be several interpolation steps ahead of the action that
        # actually reached the robot.  After discarding the tail, only the last
        # command we really sent is a safe interpolation anchor.
        hold = self.last_sent()
        self._prev = None if hold is None else hold.clone()
        self._buffer = []
        self._idx = 0
        self._emitted_policy_action = False
        return hold

    def last_sent(self) -> Tensor | None:
        """The most recent action handed out by :meth:`get` (the hold value)."""
        return None if self._last_sent is None else self._last_sent.clone()

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                self._buffer.append(interp)
            # The end point is the policy's action itself, appended verbatim rather
            # than computed as ``prev + 1.0 * (action - prev)``, which can land an ULP
            # away.  ``emitted_policy_action`` promises the recorded frame carries the
            # policy's own output, so make that exact.
            self._buffer.append(action.clone())
        else:
            # First step: no previous action yet, so run at base FPS without interpolation.
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        """Get the next interpolated action.

        Returns:
            Next action tensor, or None if buffer is exhausted.
        """
        if self._idx >= len(self._buffer):
            self._emitted_policy_action = False
            return None
        action = self._buffer[self._idx]
        self._last_sent = action
        self._idx += 1
        self._emitted_policy_action = self._idx == len(self._buffer)
        return action
