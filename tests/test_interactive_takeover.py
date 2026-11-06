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

"""Hardware-free control-ownership, leader feedback and fresh-planning regressions."""

from __future__ import annotations

import os
import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout import BaseStrategy, BaseStrategyConfig, RolloutController, RolloutEvent  # noqa: E402
from lerobot.rollout.configs import RolloutConfig, TeleopTakeoverConfig  # noqa: E402
from lerobot.rollout.robot_wrapper import ThreadSafeRobot  # noqa: E402
from lerobot.rollout.takeover import AlignmentTimeoutError, LeaderFollower  # noqa: E402
from tests.interactive_helpers import _FakeEngine, _loop_ctx  # noqa: E402

pytestmark = pytest.mark.dataset


class Follower:
    action_features = {"left_joint.pos": float, "right_joint.pos": float}
    supports_position_hold = True
    robot_type = "mock"

    def __init__(self):
        self.position = dict.fromkeys(self.action_features, 0.0)
        self.sent = []
        self.track_commands = True

    def get_observation(self):
        return dict(self.position)

    def send_action(self, action):
        self.sent.append(dict(action))
        if self.track_commands:
            self.position = dict(action)
        return dict(action)


class Leader:
    action_features = Follower.action_features
    feedback_features = Follower.action_features

    def __init__(self):
        self.position = dict.fromkeys(self.action_features, 0.0)
        self.goal = dict(self.position)
        self.torque = False
        self.track_feedback = True
        self.events = []

    def get_action(self):
        return dict(self.position)

    def send_feedback(self, target):
        self.goal = dict(target)
        self.events.append(("feedback", dict(target)))
        if self.torque and self.track_feedback:
            self.position = dict(target)

    def enable_torque(self):
        self.events.append(("enable", dict(self.goal)))
        self.torque = True

    def disable_torque(self):
        self.events.append(("disable", dict(self.goal)))
        self.torque = False


def _config(**overrides):
    return TeleopTakeoverConfig(
        enabled=True,
        **{"alignment_duration_s": 0.01, "alignment_timeout_s": 0.1, "max_position_error": 0.05, **overrides},
    )


def _hardware():
    robot = Follower()
    wrapper = ThreadSafeRobot(robot)
    wrapper.get_observation()
    return robot, wrapper, Leader()


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert predicate()


def _session(goal="insert the tool"):
    robot, wrapper, teleop = _hardware()
    engine = _FakeEngine()
    engine.get_action.return_value = torch.tensor([10.0, 20.0])
    ctx = _loop_ctx(engine, robot=wrapper, takeover=_config(), autosteer_goal=goal, autosteer_interval_s=100)
    ctx.hardware.teleop = teleop
    ctx.hardware.initial_position = dict(robot.position)
    ctx.processors.teleop_action_processor = lambda pair: pair[0]
    ctx.processors.robot_action_processor = lambda pair: pair[0]
    ctx.data.ordered_action_keys = list(robot.action_features)
    ctx.data.dataset_features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": list(robot.action_features)}
    }
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy.setup(ctx)
    events = []
    controller = RolloutController(strategy, ctx, on_event=lambda event, payload: events.append(event))
    thread = Thread(target=controller.serve)
    return controller, thread, robot, teleop, engine, events, strategy, ctx


def _finish(controller, thread):
    controller.stop()
    thread.join(2)
    assert not thread.is_alive()


def test_mirror_uses_measured_position_and_seeds_before_torque():
    robot, wrapper, teleop = _hardware()
    teleop.position = {"left_joint.pos": -5.0, "right_joint.pos": 5.0}
    mirror = LeaderFollower(wrapper, teleop, _config(), 500)
    assert mirror.align(Event())
    assert teleop.events[0] == ("feedback", {"left_joint.pos": -5.0, "right_joint.pos": 5.0})
    assert teleop.events[1] == ("enable", teleop.events[0][1])
    robot.track_commands = False
    wrapper.send_action({"left_joint.pos": 80.0, "right_joint.pos": 90.0})
    robot.position = {"left_joint.pos": 1.0, "right_joint.pos": 2.0}
    mirror.sync(wrapper.get_observation())
    assert teleop.goal == robot.position
    assert teleop.goal != robot.sent[-1]
    mirror.release()
    assert not teleop.torque


def test_alignment_checks_both_arms_and_can_be_cancelled():
    robot, wrapper, teleop = _hardware()
    teleop.track_feedback = False
    teleop.position["right_joint.pos"] = 30.0
    mirror = LeaderFollower(wrapper, teleop, _config(), 500)
    with pytest.raises(AlignmentTimeoutError):
        mirror.align(Event())
    mirror.release()
    cancelled = Event()
    cancelled.set()
    before = len(teleop.events)
    assert mirror.align(cancelled) is False
    assert len(teleop.events) == before


@pytest.mark.parametrize("problem", ["keys", "feedback", "hold"])
def test_reject_incompatible_hardware_without_movement(problem):
    robot, wrapper, teleop = _hardware()
    if problem == "keys":
        teleop.action_features = {"other.pos": float}
    elif problem == "feedback":
        teleop.feedback_features = {}
    else:
        robot.supports_position_hold = False
    with pytest.raises(ValueError):
        LeaderFollower(wrapper, teleop, _config(), 100)
    assert not robot.sent and not teleop.events


@pytest.mark.parametrize("goal", ["insert the tool", None])
def test_policy_human_policy_cycle_replans_from_corrected_scene(goal):
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session(goal)
    thread.start()
    try:
        assert controller.start()
        _wait(lambda: any(action["left_joint.pos"] == 10 for action in robot.sent))
        assert teleop.torque
        assert controller.teleop()
        _wait(lambda: controller.teleoperating)
        assert not teleop.torque and not controller.running
        assert controller.autosteer_goal == goal
        predictions = engine.get_action.call_count
        plans = engine.subtasks_planned
        corrected = {"left_joint.pos": 32.0, "right_joint.pos": 44.0}
        teleop.position = corrected
        _wait(lambda: robot.position == corrected)
        assert engine.get_action.call_count == predictions
        assert engine.subtasks_planned == plans
        assert not controller.start()
        assert controller.resume()
        _wait(lambda: events.count(RolloutEvent.SEGMENT_STARTED) == 2)
        if goal is not None:
            assert engine.subtasks_planned == plans + 1
            assert engine.seen_query_obs[-1] == corrected
        else:
            assert engine.task == "pick up the cube"
        assert engine.reset.call_count >= 3
        assert teleop.torque
        assert ctx.hardware.leader_follower.following
    finally:
        _finish(controller, thread)
    assert not teleop.torque


@pytest.mark.parametrize("command", ["stop", "reset", "parent"])
def test_manual_control_stops_for_lifecycle_commands(command):
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    # Avoid a real-time homing ramp; assert ownership and invocation instead.
    strategy.return_to_initial_position = MagicMock(return_value=True)
    thread.start()
    try:
        assert controller.teleop()  # Also allowed before the first /start.
        _wait(lambda: controller.teleoperating)
        if command == "parent":
            ctx.runtime.shutdown_event.parent.set()
        else:
            getattr(controller, command)()
        _wait(lambda: not controller.teleoperating)
        if command == "reset":
            _wait(lambda: RolloutEvent.RESET_DONE in events)
            strategy.return_to_initial_position.assert_called_once()
        assert not controller.running
        engine.get_action.assert_not_called()
    finally:
        _finish(controller, thread)


def test_alignment_timeout_keeps_follower_held_without_manual_or_policy_control():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    teleop.track_feedback = False
    teleop.position["right_joint.pos"] = 30.0
    thread.start()
    try:
        assert controller.teleop()
        _wait(lambda: RolloutEvent.TELEOP_ENDED in events)
        assert RolloutEvent.ALIGNMENT_FAILED in events
        assert RolloutEvent.TELEOP_STARTED not in events
        assert not controller.teleoperating and not controller.failed
        assert all(action == {"left_joint.pos": 0.0, "right_joint.pos": 0.0} for action in robot.sent)
        engine.get_action.assert_not_called()
    finally:
        _finish(controller, thread)


@pytest.mark.parametrize("operation", ["feedback", "read", "send"])
def test_hardware_failure_stops_session_and_releases_leader(operation):
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    thread.start()
    try:
        assert controller.teleop()
        _wait(lambda: controller.teleoperating)
        if operation == "feedback":
            teleop.send_feedback = MagicMock(side_effect=OSError("leader feedback failed"))
            controller.resume()
        elif operation == "read":
            teleop.get_action = MagicMock(side_effect=OSError("leader read failed"))
        else:
            robot.send_action = MagicMock(side_effect=OSError("follower send failed"))
        _wait(lambda: controller.stopped)
        assert controller.failed
        assert "failed" in controller.failure_traceback
        assert not teleop.torque
        assert RolloutEvent.STRATEGY_FAILED in events
        assert ctx.hardware.leader_follower.failed
        ctx.runtime.cfg.return_to_initial_position = True
        strategy._teardown_hardware = MagicMock()
        strategy.teardown(ctx)
        strategy._teardown_hardware.assert_called_once_with(ctx.hardware, return_to_initial_position=False)
    finally:
        _finish(controller, thread)


def test_takeover_during_blocking_sync_action_discards_returned_action():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session(None)
    entered, release = Event(), Event()

    def blocked_action(_frame):
        entered.set()
        assert release.wait(2)
        return torch.tensor([99.0, 99.0])

    engine.get_action.side_effect = blocked_action
    thread.start()
    try:
        controller.start()
        assert entered.wait(2)
        assert controller.teleop()
        release.set()
        _wait(lambda: controller.teleoperating)
        assert all(action["left_joint.pos"] != 99 for action in robot.sent)
    finally:
        release.set()
        _finish(controller, thread)


def test_takeover_during_planning_rejects_late_subtask():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    entered, release = Event(), Event()
    generate = engine._generate_text

    def blocked_text(obs, query):
        entered.set()
        assert release.wait(2)
        return generate(obs, query)

    engine._generate_text = blocked_text
    thread.start()
    try:
        controller.start()
        assert entered.wait(2)
        controller.teleop()
        release.set()
        _wait(lambda: controller.teleoperating)
        assert engine.task == "pick up the cube"
        assert RolloutEvent.SEGMENT_STARTED not in events
        engine.get_action.assert_not_called()
    finally:
        release.set()
        _finish(controller, thread)


@pytest.mark.parametrize("option,value", [("interactive", False), ("teleop", None), ("fps", 0)])
def test_takeover_config_rejects_invalid_combinations(option, value):
    from tests.mocks.mock_robot import MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    args = {
        "robot": MockRobotConfig(),
        "teleop": MockTeleopConfig(),
        "policy": SimpleNamespace(device="cpu"),
        "interactive": True,
        "takeover": _config(),
    }
    args[option] = value
    with pytest.raises(ValueError, match="takeover"):
        RolloutConfig(**args)


def test_takeover_config_rejects_so_unit_mismatch():
    from lerobot.robots.bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeaderConfig

    with pytest.raises(ValueError, match="use_degrees"):
        RolloutConfig(
            robot=BiSOFollowerConfig(
                left_arm_config=SOFollowerConfig(port="/mock/follower-left", use_degrees=False),
                right_arm_config=SOFollowerConfig(port="/mock/follower-right", use_degrees=False),
            ),
            teleop=BiSOLeaderConfig(
                left_arm_config=SOLeaderConfig(port="/mock/leader-left", use_degrees=False),
                right_arm_config=SOLeaderConfig(port="/mock/leader-right", use_degrees=True),
            ),
            policy=SimpleNamespace(device="cpu"),
            interactive=True,
            takeover=_config(),
        )


def test_stop_cancels_alignment_before_manual_control():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    teleop.track_feedback = False
    teleop.position["right_joint.pos"] = 30.0
    thread.start()
    try:
        controller.teleop()
        _wait(lambda: teleop.torque)
        controller.stop()
        _wait(lambda: controller.stopped)
        assert RolloutEvent.TELEOP_STARTED not in events
        assert not teleop.torque
        engine.get_action.assert_not_called()
    finally:
        _finish(controller, thread)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_position_never_reaches_leader(bad):
    robot, wrapper, teleop = _hardware()
    mirror = LeaderFollower(wrapper, teleop, _config(), 100)
    mirror.following = True
    with pytest.raises(ValueError, match="finite"):
        mirror.sync({"left_joint.pos": bad, "right_joint.pos": 0.0})
    assert teleop.events == []


def test_reset_cancels_queued_resume():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    entered, release = Event(), Event()
    strategy.return_to_initial_position = MagicMock(return_value=True)
    thread.start()
    try:
        controller.teleop()
        _wait(lambda: controller.teleoperating)

        def blocked_read():
            entered.set()
            assert release.wait(2)
            return dict(teleop.position)

        teleop.get_action = blocked_read
        assert entered.wait(2)
        assert controller.resume()
        controller.reset()
        release.set()
        _wait(lambda: RolloutEvent.RESET_DONE in events)
        assert RolloutEvent.SEGMENT_STARTED not in events
        engine.get_action.assert_not_called()
    finally:
        release.set()
        _finish(controller, thread)


def test_frontend_routes_takeover_and_resume_and_reports_disabled(capsys):
    from lerobot.rollout import InteractiveSession
    from lerobot.rollout.interactive import InteractiveCommand
    from tests.interactive_helpers import _make_ctx

    ctx, strategy, _engine, _parent, _run_started = _make_ctx()
    session = InteractiveSession(strategy, ctx)
    session._cmd_teleop(InteractiveCommand("teleop", ""))
    assert "disabled" in capsys.readouterr().out
    session.controller = MagicMock(takeover_enabled=True)
    session.controller.teleop.return_value = True
    session.controller.resume.return_value = True
    session._handle_line("/teleop")
    session._handle_line("/resume")
    session.controller.teleop.assert_called_once()
    session.controller.resume.assert_called_once()


def test_real_rtc_handover_discards_inflight_chunk_and_replans():
    from lerobot.inference import RTCInferenceEngine
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from tests.inference.rtc_helpers import ControlledPolicy, Pipeline

    _unused, _thread, robot, teleop, _fake, _events, _strategy, ctx = _session()
    policy = ControlledPolicy()
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=Pipeline(),
        postprocessor=Pipeline(width=2),
        robot_wrapper=ctx.hardware.robot_wrapper,
        rtc_config=RTCConfig(enabled=True, execution_horizon=4),
        dataset_features=ctx.data.dataset_features,
        task="initial",
        fps=100,
        device="cpu",
        action_timeout_s=2,
        startup_timeout_s=2,
    )
    ctx.policy.inference = engine
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy.setup(ctx)
    events = []
    controller = RolloutController(strategy, ctx, on_event=lambda event, payload: events.append(event))
    thread = Thread(target=controller.serve)
    thread.start()
    try:
        controller.start()
        _wait(lambda: len(policy.calls) == 1)
        # The model is inside a chunk call when the operator takes over.
        controller.teleop()
        _wait(lambda: controller.teleoperating)
        policy.release.release()
        corrected = {"left_joint.pos": 32.0, "right_joint.pos": 44.0}
        teleop.position = corrected
        _wait(lambda: robot.position == corrected)
        assert all(action["left_joint.pos"] != 1.0 for action in robot.sent)
        controller.resume()
        _wait(lambda: len(policy.text_observations) >= 2)
        assert policy.text_observations[-1].flatten().tolist() == [32.0, 44.0]
        assert controller.autosteer_goal == "insert the tool"
        policy.release.release(20)
        _wait(lambda: any(action["left_joint.pos"] == 1.0 for action in robot.sent))
        assert not controller.failed
    finally:
        controller.stop()
        policy.release.release(20)
        thread.join(2)
        engine.stop()
        assert not thread.is_alive()


def test_toggle_follows_actual_control_state_and_ignores_pending_resume():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session()
    entered, release = Event(), Event()
    assert controller.toggle_teleop() is None  # Idle does not start a robot.
    thread.start()
    try:
        controller.start()
        _wait(lambda: RolloutEvent.SEGMENT_STARTED in events)
        assert controller.toggle_teleop() == "teleop"
        _wait(lambda: controller.teleoperating)

        def blocked_read():
            entered.set()
            assert release.wait(2)
            return dict(teleop.position)

        teleop.get_action = blocked_read
        assert entered.wait(2)
        assert controller.toggle_teleop() == "resume"
        assert controller.toggle_teleop() is None
        assert not controller._teleop_requested.is_set()
        release.set()
        _wait(lambda: events.count(RolloutEvent.SEGMENT_STARTED) == 2)
        assert controller.autosteer_goal == "insert the tool"
    finally:
        release.set()
        _finish(controller, thread)


def test_toggle_while_takeover_waits_does_not_schedule_resume():
    controller, thread, robot, teleop, engine, events, strategy, ctx = _session(None)
    entered, release = Event(), Event()

    def blocked_action(_frame):
        entered.set()
        assert release.wait(2)
        return torch.tensor([99.0, 99.0])

    engine.get_action.side_effect = blocked_action
    thread.start()
    try:
        controller.start()
        assert entered.wait(2)
        assert controller.toggle_teleop() == "teleop"
        assert controller.toggle_teleop() is None
        assert not controller._start_requested.is_set()
        release.set()
        _wait(lambda: controller.teleoperating)
    finally:
        release.set()
        _finish(controller, thread)


def test_terminal_space_drives_full_interactive_takeover_cycle():
    termios = pytest.importorskip("termios")
    from lerobot.rollout import InteractiveSession

    _old_controller, _old_thread, robot, teleop, engine, _events, strategy, ctx = _session()
    master, slave = os.openpty()
    reader = os.fdopen(slave, "r", encoding="utf-8")
    original = termios.tcgetattr(slave)
    session = InteractiveSession(strategy, ctx, input_stream=reader)
    session._listener._SPACE_REPEAT_GAP_S = 0.01
    thread = Thread(target=session.run)
    thread.start()
    try:
        _wait(lambda: session._listener.space_hotkey_enabled)
        os.write(master, b"/start\n")
        _wait(lambda: engine.get_action.call_count > 0)
        os.write(master, b" ")
        _wait(lambda: session.controller.teleoperating)
        assert not teleop.torque
        corrected = {"left_joint.pos": 32.0, "right_joint.pos": 44.0}
        teleop.position = corrected
        _wait(lambda: robot.position == corrected)
        time.sleep(0.02)
        os.write(master, b" ")
        _wait(lambda: engine.subtasks_planned == 2)
        assert engine.seen_query_obs[-1] == corrected
        assert session.controller.autosteer_goal == "insert the tool"
        assert teleop.torque
        os.write(master, b"/stop\n")
        thread.join(2)
        assert not thread.is_alive()
        assert termios.tcgetattr(slave) == original
    finally:
        session.controller.stop()
        thread.join(2)
        reader.close()
        os.close(master)
        assert not thread.is_alive()
