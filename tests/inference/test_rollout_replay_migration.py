"""Opt-in rollout diagnostics share the upstream prediction and execution boundary."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.inference import ExecutionMode
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.context import _policy_action_keys
from tests.inference.policy_helpers import local_engine_for, tiny_config
from tests.inference.test_prediction_parity import ObservedPolicy


@pytest.mark.parametrize("mode", [ExecutionMode.CHUNK, ExecutionMode.RTC_GUIDED, ExecutionMode.RTC_TRAINED])
def test_saliency_replays_exact_noise_and_reanchored_prefix_after_native_accept(monkeypatch, mode):
    config = tiny_config()
    config.max_action_dim = 3
    config.rtc_training_max_delay = 3
    config.rtc_config = RTCConfig(
        enabled=mode is not ExecutionMode.CHUNK,
        mode="trained" if mode is ExecutionMode.RTC_TRAINED else "guided",
        execution_horizon=4,
    )
    policy = ObservedPolicy(config)
    engine = local_engine_for(policy, relative=mode is not ExecutionMode.CHUNK, queue_threshold=8)
    snapshots = []
    engine.saliency_sink = SimpleNamespace(submit=snapshots.append)
    engine.resume()
    previous = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    if mode is not ExecutionMode.CHUNK:
        engine.action_queue.merge(previous, previous * 2 + torch.tensor([1.0, 2.0, 3.0]) + 10, 0)
    engine._runtime.turnarounds.append(1 / 30)
    engine.notify_observation(
        {"a.pos": 20.0, "b.pos": 20.0, "c.pos": 20.0, "env_a": 0.0, "env_b": 0.0, "env_c": 0.0}
    )
    original_predict = policy.predict_action_chunk
    inputs = []

    def predict(batch, **kwargs):
        inputs.append(batch["observation.state"].clone())
        result = original_predict(batch, **kwargs)
        # Real policies may decode state in place; the replay must keep the input.
        batch["observation.state"].add_(1000)
        return result

    monkeypatch.setattr(policy, "predict_action_chunk", predict)
    original_accept = engine._runtime.accept

    def accept(request, chunk, *, task_version):
        engine._shutdown_event.set()
        return original_accept(request, chunk, task_version=task_version)

    monkeypatch.setattr(engine._runtime, "accept", accept)
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    torch.testing.assert_close(snapshot["noise"], policy.last_kwargs["noise"])
    torch.testing.assert_close(snapshot["preprocessed"]["observation.state"], inputs[0])
    assert snapshot["first_queued_action_index"] == engine._runtime.last_accept["trimmed_actions"]
    assert snapshot["accepted_epoch"] == engine._reset_epoch
    if mode is ExecutionMode.CHUNK:
        assert snapshot["chunk_kwargs"] == {}
    else:
        assert snapshot["chunk_kwargs"]["inference_delay"] == 1
        prefix = snapshot["chunk_kwargs"]["prev_chunk_left_over"]
        torch.testing.assert_close(prefix, policy.last_kwargs["prev_chunk_left_over"])
        torch.testing.assert_close(prefix, previous[:4] - 5)


@pytest.mark.parametrize("invalidate", ["reject", "reset"])
def test_saliency_never_submits_a_discarded_or_pre_reset_chunk(monkeypatch, invalidate):
    config = tiny_config()
    config.max_action_dim = 3
    config.rtc_config = RTCConfig(enabled=False)
    policy = ObservedPolicy(config)
    engine = local_engine_for(policy, queue_threshold=8)
    snapshots = []
    epochs = []
    engine.saliency_sink = SimpleNamespace(submit=snapshots.append, set_reset_epoch=epochs.append)
    engine.resume()
    engine.notify_observation(
        {"a.pos": 1.0, "b.pos": 1.0, "c.pos": 1.0, "env_a": 0.0, "env_b": 0.0, "env_c": 0.0}
    )
    if invalidate == "reset":
        original = policy.predict_action_chunk

        def predict(batch, **kwargs):
            result = original(batch, **kwargs)
            engine.reset()
            engine._shutdown_event.set()
            return result

        monkeypatch.setattr(policy, "predict_action_chunk", predict)
    else:

        def reject(*args, **kwargs):
            engine._shutdown_event.set()
            return False

        monkeypatch.setattr(engine._runtime, "accept", reject)
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    assert snapshots == []
    assert engine.action_queue.empty()
    if invalidate == "reset":
        assert epochs == [1]


def test_robot_processor_action_keys_use_ee_space_and_checkpoint_order():
    features = {
        "action": {"names": ["left.ee.x", "right.ee.x"]},
        "observation.state": {"names": ["left.ee.x", "right.ee.x"]},
    }
    checkpoint = ["right.ee.x", "left.ee.x"]
    assert (
        _policy_action_keys(features, {"left.joint.pos": float, "right.joint.pos": float}, checkpoint)
        == checkpoint
    )
    assert features["action"]["names"] == checkpoint
    assert features["observation.state"]["names"] == checkpoint


def test_robot_processor_different_state_space_remains_unchanged():
    features = {"action": {"names": ["ee.x"]}, "observation.state": {"names": ["joint.pos"]}}
    assert _policy_action_keys(features, {"joint.pos": float}, ["ee.x"]) == ["ee.x"]
    assert features["observation.state"]["names"] == ["joint.pos"]


def _run_one_chunk(monkeypatch, *, sink=None):
    config = tiny_config()
    config.max_action_dim = 3
    config.rtc_config = RTCConfig(enabled=False)
    policy = ObservedPolicy(config)
    engine = local_engine_for(policy, queue_threshold=8)
    engine.saliency_sink = sink
    engine.resume()
    engine.notify_observation(
        {"a.pos": 1.0, "b.pos": 1.0, "c.pos": 1.0, "env_a": 0.0, "env_b": 0.0, "env_c": 0.0}
    )
    original_accept = engine._runtime.accept

    def accept(request, chunk, *, task_version):
        engine._shutdown_event.set()
        return original_accept(request, chunk, task_version=task_version)

    monkeypatch.setattr(engine._runtime, "accept", accept)
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    return engine, policy


def test_saliency_disabled_never_samples_noise_or_builds_replay(monkeypatch):
    from lerobot.inference import rtc

    def forbidden(*args, **kwargs):
        pytest.fail("disabled saliency must not sample or capture")

    monkeypatch.setattr(rtc, "sample_noise", forbidden)
    monkeypatch.setattr(rtc, "_build_saliency_snapshot", forbidden)
    engine, policy = _run_one_chunk(monkeypatch)
    assert engine.action_queue.qsize() == 3
    assert "noise" not in policy.last_kwargs


def test_saliency_sink_failure_preserves_motion_and_disables_future_capture(monkeypatch):
    def fail(snapshot):
        raise RuntimeError("diagnostic output unavailable")

    engine, policy = _run_one_chunk(monkeypatch, sink=SimpleNamespace(submit=fail))
    assert engine.saliency_sink is None
    assert "noise" in policy.last_kwargs  # The accepted first chunk used the captured noise.
    assert engine.action_queue.qsize() == 3
    engine._shutdown_event.clear()
    engine._rtc_loop()
    assert not engine.failed
    assert "noise" not in policy.last_kwargs
    assert engine.action_queue.qsize() == 6


def test_view_mask_external_planner_combination_fails_before_hardware_setup():
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.planner import PlannerConfig
    from lerobot.rollout.view_mask.config import ViewMaskConfig

    with pytest.raises(ValueError, match="planner images bypass"):
        RolloutConfig(view_mask=ViewMaskConfig(enabled=True), planner=PlannerConfig())
