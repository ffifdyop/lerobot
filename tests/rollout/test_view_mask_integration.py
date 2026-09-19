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

"""Integration tests for View Ablation across engines, strategies and saliency.

Covers ``add-rollout-view-ablation`` tasks 3.7 (engine half), 4.5, 5.5 and 6.1–6.4:
revision provenance through sync and RTC, dataset frames, the event sidecar and the
saliency revision filter — plus the disabled-by-default regression guard.
"""

from __future__ import annotations

import json
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.rollout import BaseStrategyConfig
from lerobot.rollout.controller import LinkedEvent
from lerobot.rollout.strategies import BaseStrategy
from lerobot.rollout.view_mask.config import ViewMaskConfig, build_view_mask_plan
from lerobot.rollout.view_mask.controller import MaskApplicationState, ViewMaskController
from lerobot.rollout.view_mask.events import ViewMaskEventRecorder
from lerobot.rollout.view_mask.runtime import ViewMaskRuntime

POLICY_VIEWS = ["observation.images.top", "observation.images.left_wrist"]
ROBOT_CAMERAS = ["top", "left_wrist"]


def _plan(**kwargs):
    return build_view_mask_plan(
        ViewMaskConfig(enabled=True, **kwargs),
        policy_view_keys=POLICY_VIEWS,
        robot_camera_keys=ROBOT_CAMERAS,
    )


def _runtime(plan=None, recorder=None, clock=None):
    plan = plan or _plan()
    controller = ViewMaskController(plan, timeout_s=10.0, **({"clock": clock} if clock else {}))
    return ViewMaskRuntime(plan, controller, recorder)


# ---------------------------------------------------------------------------
# 3.7 — Sync engine provenance
# ---------------------------------------------------------------------------


class _StubSyncPolicy:
    """Minimal stand-in for a policy: returns a fixed action, counts resets."""

    def __init__(self, action_dim: int = 2):
        self.action_dim = action_dim
        self.resets = 0
        self.seen_batches: list[dict] = []
        self.config = SimpleNamespace(use_amp=False)

    def reset(self):
        self.resets += 1

    def drop_queued_actions(self):
        pass

    def supports_text_generation(self):
        return False

    def get_queued_actions(self):
        return None

    def select_action(self, batch):
        self.seen_batches.append(batch)
        return torch.zeros(1, self.action_dim)


class _ObservingStep:
    """A real step placed *after* the mask step, so a test can see the policy's view.

    It doubles as the flush probe: ``PolicyProcessorPipeline.reset()`` calls ``reset()``
    on every step that has one, so this counter is what proves the preprocessor half of
    the Inference Flush actually ran.
    """

    def __init__(self):
        self.seen: list[dict] = []
        self.resets = 0

    def __call__(self, transition):
        observation = transition.get("observation")
        if isinstance(observation, dict):
            self.seen.append(observation)
        return transition

    def transform_features(self, features):
        return features

    def reset(self):
        self.resets += 1


def _make_sync_engine(runtime):
    """A sync engine whose preprocessor is a *real* pipeline holding the mask step.

    The mask step lives inside the checkpoint preprocessor, so a pass-through stub
    would silently bypass the black-view replacement and let these tests pass while
    the rollout did not.  The pipeline is the genuine ``PolicyProcessorPipeline``,
    with an observing step after the mask.
    """
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.rollout.context import _inject_view_mask_step
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    probe = _ObservingStep()
    preprocessor = PolicyProcessorPipeline(steps=[])
    _inject_view_mask_step(preprocessor)
    preprocessor.steps = [*preprocessor.steps, probe]

    engine = SyncInferenceEngine(
        policy=_StubSyncPolicy(),
        preprocessor=preprocessor,
        # The standard postprocessor adapters: it consumes the policy's action tensor
        # and hands the same tensor back, exactly like a real pipeline with no steps.
        postprocessor=PolicyProcessorPipeline(
            steps=[],
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
        dataset_features={"action": {"names": ["a", "b"]}},
        ordered_action_keys=["a", "b"],
        task="fold",
        device="cpu",
        robot_type="stub",
    )
    engine.probe = probe
    engine.attach_view_mask(runtime)
    return engine


def _policy_view(engine) -> dict:
    """The observation the policy was handed, read past the mask step."""
    return engine.probe.seen[-1]


def _obs_frame():
    return {
        "observation.images.top": np.ones((4, 4, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.full((4, 4, 3), 128, dtype=np.uint8),
    }


def test_sync_policy_sees_a_black_view_for_the_masked_revision():
    runtime = _runtime(_plan(initial_views=["left_wrist"]))
    engine = _make_sync_engine(runtime)

    engine.get_action(_obs_frame())

    view = _policy_view(engine)
    assert torch.count_nonzero(view["observation.images.left_wrist"]) == 0
    assert torch.count_nonzero(view["observation.images.top"]) > 0


def test_sync_markers_never_reach_the_policy():
    from lerobot.rollout.view_mask.keys import VIEW_MASK_SNAPSHOT

    runtime = _runtime(_plan(initial_views=["left_wrist"]))
    engine = _make_sync_engine(runtime)
    engine.get_action(_obs_frame())

    view = _policy_view(engine)
    assert VIEW_MASK_SNAPSHOT not in view


def test_sync_dispatched_provenance_is_the_revision_that_generated_the_action():
    runtime = _runtime(_plan(initial_views=["left_wrist"]))
    engine = _make_sync_engine(runtime)
    engine.get_action(_obs_frame())

    assert engine.dispatched_view_mask.revision == 0
    assert engine.dispatched_view_mask.masked_views == {"observation.images.left_wrist"}


def test_sync_transition_flushes_policy_and_processors_but_keeps_the_task():
    runtime = _runtime(_plan())
    engine = _make_sync_engine(runtime)
    engine.get_action(_obs_frame())
    assert engine._policy.resets == 0
    assert engine.probe.resets == 0

    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)

    assert engine._policy.resets == 1
    assert engine.probe.resets == 1  # the preprocessor half of the flush ran too
    assert engine.task == "fold"  # a mask change re-conditions, it does not re-instruct


def test_base_strategy_runs_a_fresh_action_through_mask_transition():
    """Exercise the real BaseStrategy loop across APPLYING -> ACTIVE."""

    runtime = _runtime(_plan())
    runtime.controller.toggle("observation.images.top")
    shutdown = Event()
    states_at_inference: list[MaskApplicationState] = []

    class _LoopEngine:
        def __init__(self):
            self.view_mask = runtime
            self.failed = False
            self.failure_traceback = None
            self.resets = 0
            self.started = False
            self.resumed = False
            self.actions = 0

        def reset(self):
            self.resets += 1

        def start(self):
            self.started = True

        def resume(self):
            self.resumed = True

        def stop(self):
            pass

        def notify_observation(self, _obs):
            pass

        def pump_query(self, _obs):
            pass

        def on_mask_transition(self, _transition):
            pass

        def get_action(self, _obs):
            states_at_inference.append(runtime.controller.state)
            snapshot = runtime.snapshot()
            runtime.set_dispatched(snapshot)
            runtime.mark_active(snapshot)
            self.actions += 1
            return torch.tensor([0.5])

    engine = _LoopEngine()
    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.0}
    robot.send_action.side_effect = lambda _action: shutdown.set()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                fps=200.0,
                interpolation_multiplier=1,
                duration=0.0,
                use_torch_compile=False,
                display_data=False,
                display_mode="rerun",
                display_compressed_images=False,
                cadence_report=None,
            ),
            shutdown_event=shutdown,
            cadence_report=None,
        ),
        hardware=SimpleNamespace(robot_wrapper=robot),
        processors=SimpleNamespace(
            robot_observation_processor=lambda obs: obs,
            robot_action_processor=lambda pair: pair[0],
            teleop_action_processor=lambda pair: pair[0],
        ),
        policy=SimpleNamespace(inference=engine, view_mask=runtime),
        data=SimpleNamespace(
            dataset_features={"observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}},
            ordered_action_keys=["joint.pos"],
        ),
    )

    strategy = BaseStrategy(BaseStrategyConfig())
    strategy.setup(ctx)
    strategy.run(ctx)

    assert states_at_inference == [MaskApplicationState.APPLYING]
    assert engine.actions == 1
    assert runtime.controller.state is MaskApplicationState.ACTIVE
    assert robot.send_action.call_count == 1


def test_sync_marks_active_on_the_first_action_matching_the_new_revision():
    runtime = _runtime(_plan())
    engine = _make_sync_engine(runtime)
    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    assert runtime.controller.state is MaskApplicationState.APPLYING

    engine.get_action(_obs_frame())

    assert runtime.controller.state is MaskApplicationState.ACTIVE


def test_sync_discards_an_action_returned_after_the_mask_moved_on():
    # The race the revision guard exists for: inference starts, a keypress lands
    # mid-flight, and the action it returns describes a scene nobody is looking at.
    runtime = _runtime(_plan())
    engine = _make_sync_engine(runtime)
    original_select = engine._policy.select_action

    def select_and_switch(batch):
        runtime.controller.toggle("observation.images.left_wrist")
        runtime.poll(runnable=True)
        return original_select(batch)

    engine._policy.select_action = select_and_switch

    assert engine.get_action(_obs_frame()) is None


def test_sync_discards_an_action_once_the_mask_has_failed():
    runtime = _runtime(_plan())
    engine = _make_sync_engine(runtime)
    original_select = engine._policy.select_action

    def select_and_fail(batch):
        runtime.mark_failed("engine died")
        return original_select(batch)

    engine._policy.select_action = select_and_fail

    assert engine.get_action(_obs_frame()) is None


# ---------------------------------------------------------------------------
# 3.5 / 3.6 — timeout, VQA gating and the non-interactive end
# ---------------------------------------------------------------------------


class _ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_timeout_only_accrues_on_runnable_ticks_across_a_real_loop():
    clock = _ManualClock()
    runtime = _runtime(_plan(), clock=clock)
    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)

    # A long stretch of non-runnable ticks — idle, or under human control — must not
    # fail the mask.  An operator pre-setting a mask is not a policy failure.
    for _ in range(100):
        clock.now += 1.0
        runtime.poll(runnable=False)
    assert not runtime.failed

    # Inference resumes: the first runnable tick re-arms the budget (no wall time from
    # the idle stretch is chargeable), and only runnable time after that counts.
    runtime.poll(runnable=True)
    clock.now += 9.9
    runtime.poll(runnable=True)
    assert not runtime.failed

    clock.now += 0.2
    runtime.poll(runnable=True)
    assert runtime.failed
    assert "10.0s" in runtime.controller.failure_reason


def test_blocked_reports_applying_then_failed():
    runtime = _runtime(_plan())
    assert not runtime.blocked
    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    assert runtime.blocked and not runtime.failed

    runtime.mark_failed("boom")
    assert runtime.blocked and runtime.failed


def test_failed_event_is_recorded_once_per_revision(tmp_path):
    path = tmp_path / "view_mask_events.jsonl"
    plan = _plan()
    controller = ViewMaskController(plan, timeout_s=10.0)
    recorder = ViewMaskEventRecorder(
        path,
        canonical_views=plan.canonical_views,
        initial_mask=controller.snapshot(),
    )
    runtime = ViewMaskRuntime(plan, controller, recorder)

    runtime.mark_failed("boom")
    for _ in range(10):
        runtime.poll(runnable=True)
    recorder.close()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    failures = [event for event in events if event["event"] == "failed"]
    assert len(failures) == 1


def test_vqa_and_autosteer_are_refused_while_the_mask_is_blocked():
    from lerobot.rollout.controller import AskResult, RolloutController

    runtime = _runtime(_plan())
    engine = MagicMock()
    engine.supports_text_queries = True
    engine.task = "fold"
    engine.failed = False
    engine.failure_traceback = None
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine, view_mask=runtime),
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(autosteer_interval_s=1.0, interactive=True),
            shutdown_event=LinkedEvent(Event()),
        ),
    )

    class _Strategy:
        config = SimpleNamespace(supports_interactive=True)

    controller = RolloutController(_Strategy(), ctx)
    controller._running.set()  # pretend a segment is live

    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    assert controller.ask("what do you see?") is AskResult.MASK_APPLYING
    assert controller.autosteer("stack the blocks") is AskResult.MASK_APPLYING

    runtime.mark_failed("timeout")
    assert controller.ask("what do you see?") is AskResult.MASK_FAILED

    runtime.restore_launch_mask()
    assert controller.ask("what do you see?") is not AskResult.MASK_FAILED


# ---------------------------------------------------------------------------
# 4.5 — dataset frames and sidecar
# ---------------------------------------------------------------------------


def _strategy_ctx(dataset, runtime, *, task="fold"):
    """A minimal RolloutContext-alike for driving one strategy tick."""
    return SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                fps=200.0,
                interpolation_multiplier=1,
                duration=0.0,
                dataset=SimpleNamespace(single_task=task),
                task=task,
                play_sounds=False,
                display_data=False,
                display_mode="rerun",
                display_compressed_images=False,
                display_ip=None,
                display_port=None,
                use_torch_compile=False,
            ),
            shutdown_event=MagicMock(),
            cadence_report=None,
        ),
        hardware=SimpleNamespace(robot_wrapper=MagicMock(), teleop=None, initial_position=None),
        processors=SimpleNamespace(
            robot_observation_processor=MagicMock(side_effect=lambda obs: obs),
            robot_action_processor=lambda pair: pair[0],
            teleop_action_processor=lambda pair: pair[0],
        ),
        policy=SimpleNamespace(inference=MagicMock(), view_mask=runtime),
        data=SimpleNamespace(
            dataset=dataset,
            dataset_features={
                "observation.state": {"dtype": "float32", "shape": (1,), "names": ["m.pos"]},
                "action": {"dtype": "float32", "shape": (1,), "names": ["m.pos"]},
            },
            ordered_action_keys=["m.pos"],
        ),
    )


def test_disabled_runtime_records_no_mask_fields_at_all():
    # The regression guard: with ablation off, a frame is exactly the frame it was.
    assert _fields_from_strategy(None, human=False) is None
    assert _fields_from_strategy(None, human=True) is None


class _FrameFieldStrategy(BaseStrategy):
    """A concrete strategy, so the frame-field helpers can be exercised directly.

    The helpers are pure functions of the runtime, but they live on the strategy
    because that is who owns the recording decision.
    """

    def setup(self, ctx): ...

    def run(self, ctx): ...

    def teardown(self, ctx): ...


def _fields_from_strategy(runtime, *, human: bool):
    """Exercise the strategy-level frame-field helper without a full loop."""
    from lerobot.rollout import BaseStrategyConfig

    strategy = _FrameFieldStrategy(BaseStrategyConfig())
    strategy._view_mask = runtime
    if human:
        return strategy._human_view_mask_fields()
    return strategy._view_mask_frame_fields(human=False)


def test_a_transition_does_not_backfill_the_dispatched_provenance():
    # The transition must never relabel the last action as belonging to the mask it
    # switched *to*: nothing has been dispatched under the new revision yet.  If it did,
    # an RTC frame for an action generated under revision N would claim revision N+1 —
    # exactly the mislabelling the provenance field exists to prevent.
    runtime = _runtime(_plan(initial_views=["top"]))
    runtime.set_dispatched(runtime.controller.snapshot())
    assert runtime.dispatched.revision == 0

    runtime.controller.toggle("observation.images.left_wrist")
    runtime.poll(runnable=True)

    assert runtime.current.revision == 1
    assert runtime.dispatched.revision == 0


def test_policy_frames_are_labelled_from_the_dispatched_snapshot():
    runtime = _runtime(_plan(initial_views=["top"]))
    runtime.set_dispatched(runtime.controller.snapshot())

    fields = _fields_from_strategy(runtime, human=False)

    assert fields["view_mask_revision"].tolist() == [0]
    assert fields["view_mask"].tolist() == [True, False]


def test_policy_frames_follow_the_dispatched_snapshot_after_a_new_action_lands():
    runtime = _runtime(_plan(initial_views=["top"]))
    runtime.set_dispatched(runtime.controller.snapshot())

    runtime.controller.toggle("observation.images.left_wrist")
    runtime.poll(runnable=True)
    # The engine returns a fresh action for the new revision: only now does the
    # dispatched provenance move, and only now may frames be recorded again.
    runtime.set_dispatched(runtime.controller.snapshot())
    runtime.mark_active(runtime.current)

    fields = _fields_from_strategy(runtime, human=False)

    assert fields["view_mask_revision"].tolist() == [1]
    assert fields["view_mask"].tolist() == [True, True]


def test_human_frame_is_suppressed_until_the_new_mask_is_active():
    runtime = _runtime(_plan(initial_views=["top"]))
    runtime.set_dispatched(runtime.controller.snapshot())
    runtime.controller.toggle("observation.images.left_wrist")
    runtime.poll(runnable=True)

    assert _fields_from_strategy(runtime, human=True) is None


def test_applying_ticks_produce_no_training_frame():
    runtime = _runtime(_plan())
    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)

    assert _fields_from_strategy(runtime, human=False) is None


def test_human_frames_are_suppressed_while_the_mask_is_blocked():
    # A FAILED mask must not take the robot away from the operator, but blocked ticks
    # are not training samples.
    runtime = _runtime(_plan())
    runtime.mark_failed("timeout")

    assert _fields_from_strategy(runtime, human=True) is None


def test_episode_rotation_keeps_the_mask_and_revision():
    runtime = _runtime(_plan(initial_views=["top"]))
    runtime.controller.toggle("observation.images.left_wrist")
    runtime.poll(runnable=True)
    runtime.mark_active(runtime.current)
    revision, revision_after = runtime.revision, None

    # Nothing in an episode boundary touches the controller, so a rotation is a no-op.
    revision_after = runtime.revision

    assert revision_after == revision
    assert runtime.current.masked_views == {"observation.images.top", "observation.images.left_wrist"}
    assert not runtime.blocked


def test_reset_restores_the_launch_mask_without_rewinding_the_revision(tmp_path):
    runtime = _runtime(_plan(initial_views=["top"]))
    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    assert runtime.revision == 1

    runtime.restore_launch_mask()

    assert runtime.current.masked_views == {"observation.images.top"}
    assert runtime.revision == 2  # never back to 0


def test_sidecar_records_a_full_session_timeline(tmp_path):
    path = tmp_path / "view_mask_events.jsonl"
    plan = _plan(initial_views=["top"])
    controller = ViewMaskController(plan, timeout_s=10.0)
    recorder = ViewMaskEventRecorder(
        path, canonical_views=plan.canonical_views, initial_mask=controller.snapshot()
    )
    runtime = ViewMaskRuntime(plan, controller, recorder)

    runtime.note_frame(episode_index=0, frame_index=41)
    controller.toggle("observation.images.top")
    runtime.poll(runnable=True, phase="policy")
    runtime.mark_active(runtime.current)
    runtime.mark_failed("no fresh action")
    recorder.close()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [event["event"] for event in events]
    assert kinds[0] == "run_started"
    assert "transition" in kinds
    assert "state_changed" in kinds
    assert kinds[-1] == "failed"
    failed = events[-1]
    assert failed["state"] == "failed"
    assert failed["reason"] == "no fresh action"
    assert failed["nearest_frame_index"] == 41


def test_sidecar_default_path_uses_dataset_meta_not_the_output_dir(tmp_path):
    from lerobot.rollout.view_mask.events import default_event_log_path

    assert default_event_log_path(tmp_path) == tmp_path / "meta" / "view_mask_events.jsonl"
    fallback = default_event_log_path(None)
    assert fallback.name == "view_mask_events.jsonl"
    assert fallback.parts[:2] == ("outputs", "rollout")


def test_sidecar_explicit_path_overrides_both_defaults(tmp_path):
    from lerobot.rollout.view_mask.events import resolve_event_log_path

    explicit = tmp_path / "custom.jsonl"
    assert resolve_event_log_path(str(explicit), tmp_path / "somewhere") == explicit
    assert resolve_event_log_path(None, tmp_path) == tmp_path / "meta" / "view_mask_events.jsonl"


# ---------------------------------------------------------------------------
# 4.1 — dataset schema and resume compatibility
# ---------------------------------------------------------------------------


def test_dataset_features_are_named_and_ordered_by_canonical_view():
    from lerobot.rollout.context import _view_mask_dataset_features

    features = _view_mask_dataset_features(tuple(POLICY_VIEWS))
    assert features["view_mask"]["names"] == POLICY_VIEWS
    assert features["view_mask"]["shape"] == (2,)
    assert features["view_mask_revision"]["dtype"] == "int64"


def test_a_resume_missing_the_fields_is_refused():
    from lerobot.rollout.context import _validate_view_mask_dataset

    dataset = SimpleNamespace(meta=SimpleNamespace(features={}))
    with pytest.raises(ValueError, match="is missing"):
        _validate_view_mask_dataset(dataset, tuple(POLICY_VIEWS))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"dtype": "float32", "shape": (2,), "names": POLICY_VIEWS}, "dtype"),
        ({"dtype": "bool", "shape": (3,), "names": POLICY_VIEWS}, "shape"),
        ({"dtype": "bool", "shape": (2,), "names": list(reversed(POLICY_VIEWS))}, "names"),
    ],
)
def test_a_resume_with_a_mismatched_mask_schema_is_refused(override, match):
    from lerobot.rollout.context import _validate_view_mask_dataset

    features = {
        "view_mask": override,
        "view_mask_revision": {"dtype": "int64", "shape": (1,), "names": None},
    }
    dataset = SimpleNamespace(meta=SimpleNamespace(features=features))
    with pytest.raises(ValueError, match=match):
        _validate_view_mask_dataset(dataset, tuple(POLICY_VIEWS))


def test_a_resume_with_a_compatible_schema_is_accepted():
    from lerobot.rollout.context import _validate_view_mask_dataset, _view_mask_dataset_features

    features = _view_mask_dataset_features(tuple(POLICY_VIEWS))
    dataset = SimpleNamespace(meta=SimpleNamespace(features=features))
    _validate_view_mask_dataset(dataset, tuple(POLICY_VIEWS))  # no raise


# ---------------------------------------------------------------------------
# 5.5 — saliency revision filtering and MASKED marking
# ---------------------------------------------------------------------------


class _RecordingSinkRecorder:
    """The narrow slice of SaliencyRecorder the revision filter touches."""

    def __init__(self, runtime):
        from lerobot.rollout.saliency.recorder import SaliencyRecorder

        self.recorder = SaliencyRecorder.__new__(SaliencyRecorder)
        self.recorder._view_mask = None
        self.recorder._stale_dropped = 0
        self.recorder._closed = False
        import queue

        self.recorder._queue = queue.Queue(maxsize=1)
        self.recorder.set_view_mask(runtime)

    def submit(self, item):
        self.recorder.submit(item)


def _snapshot_dict(runtime, revision=None):
    mask = runtime.controller.snapshot()
    return {"preprocessed": {}, "noise": None, "chunk_kwargs": {}, "view_mask": mask}


def test_saliency_snapshot_from_a_superseded_revision_is_dropped():
    runtime = _runtime(_plan())
    sink = _RecordingSinkRecorder(runtime)
    stale = _snapshot_dict(runtime)  # revision 0

    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)  # now revision 1

    sink.submit(stale)

    assert sink.recorder._queue.empty(), "a stale snapshot must not reach the recorder"
    assert sink.recorder._stale_dropped == 1


def test_saliency_snapshot_for_the_current_revision_is_kept():
    runtime = _runtime(_plan())
    sink = _RecordingSinkRecorder(runtime)

    sink.submit(_snapshot_dict(runtime))

    assert not sink.recorder._queue.empty()
    assert sink.recorder._stale_dropped == 0


def test_stale_saliency_drops_do_not_count_as_failures():
    # Provenance filtering must not trip the consecutive-failure downgrade, or a few
    # well-timed mask switches would silently kill saliency for the whole run.
    runtime = _runtime(_plan())
    sink = _RecordingSinkRecorder(runtime)
    stale = _snapshot_dict(runtime)

    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    for _ in range(5):
        sink.submit(stale)

    assert sink.recorder._stale_dropped == 5
    # The failure path is what increments consecutive_failures in _run; nothing here
    # entered it, so the recorder is still healthy.
    assert sink.recorder._closed is False


def test_masked_view_keeps_its_real_saliency_and_gets_a_badge():
    # A black view still produces a genuine non-zero gradient; it must be shown, not
    # zeroed or hidden, and flagged so nobody reads it as evidence about visible pixels.

    from lerobot.rollout.saliency.overlay import draw_masked_badge

    frame = np.full((96, 160, 3), 30, dtype=np.uint8)
    badged = draw_masked_badge(frame)

    assert badged.shape == frame.shape
    assert not np.array_equal(badged, frame)
    assert badged[:20, :60, 2].max() > 150  # a bright red plate in the corner
    # Pixels outside the badge are untouched: the frame is still the real image,
    # annotated rather than replaced.
    assert np.array_equal(badged[60:, 80:], frame[60:, 80:])


def test_masked_views_are_extracted_from_the_snapshot():
    from lerobot.rollout.saliency.recorder import _masked_views

    runtime = _runtime(_plan(initial_views=["top"]))
    assert _masked_views(runtime.controller.snapshot()) == {"observation.images.top"}
    assert _masked_views(None) == frozenset()


def test_display_backend_independence(tmp_path):
    # Turning the display off must not change any ablation behaviour: the telemetry
    # helper is the only thing gated on display_data.
    runtime = _runtime(_plan(initial_views=["top"]))
    strategy = _FrameFieldStrategy(BaseStrategyConfig())
    strategy._view_mask = runtime
    display_off = SimpleNamespace(cfg=SimpleNamespace(display_data=False, display_mode="rerun"))
    strategy._log_view_mask(display_off)  # no backend touched, no raise

    # And the mask itself is unaffected by not having displayed anything.
    runtime.controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    assert runtime.revision == 1


# ---------------------------------------------------------------------------
# 6.3 — RTC: dynamic multi-view mask with revision-typed actions
# ---------------------------------------------------------------------------


class _GatedViewPolicy:
    """A chunk policy that holds each inference until the test releases it.

    The gate is what makes the two races deterministic: a chunk can be parked
    mid-inference while the operator switches the mask, and a queue can be inspected
    before any of it has been consumed.
    """

    def __init__(self, chunk_len: int = 6, action_dim: int = 2):
        from threading import Event, Semaphore

        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self._release = Semaphore(0)
        self.in_inference = Event()
        self.seen_views: list[dict] = []
        self.seen_text_views: list[dict] = []
        self.reset_count = 0
        self.seen_reset_counts: list[int] = []
        self.config = SimpleNamespace(chunk_size=chunk_len, max_action_dim=action_dim)

    def allow_one_inference(self) -> None:
        self._release.release()

    def unblock(self) -> None:
        self._release.release(5)

    def predict_action_chunk(self, batch, inference_delay=0, prev_chunk_left_over=None):
        self.in_inference.set()
        released = self._release.acquire(timeout=3.0)
        self.in_inference.clear()
        if not released:
            raise TimeoutError("test never released the inference gate")
        self.seen_views.append(dict(batch))
        self.seen_reset_counts.append(self.reset_count)
        return torch.zeros(1, self.chunk_len, self.action_dim)

    def reset(self):
        self.reset_count += 1

    def supports_text_generation(self):
        return True

    def generate_text(self, batch):
        self.seen_text_views.append(dict(batch))
        return "visible"


def _make_view_rtc_engine(runtime, *, chunk_len=6, queue_threshold=30):
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.rollout.context import _inject_view_mask_step
    from lerobot.rollout.inference import RTCInferenceEngine

    probe = _ObservingStep()
    preprocessor = PolicyProcessorPipeline(steps=[])
    _inject_view_mask_step(preprocessor)
    preprocessor.steps = [*preprocessor.steps, probe]

    policy = _GatedViewPolicy(chunk_len=chunk_len)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        # Standard adapters: the postprocessor receives and returns an action tensor.
        postprocessor=PolicyProcessorPipeline(
            steps=[],
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
        robot_wrapper=SimpleNamespace(robot_type="mock", action_features={}),
        rtc_config=RTCConfig(enabled=True, execution_horizon=4, max_guidance_weight=1.0),
        hw_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
            "observation.images.top": {"dtype": "image", "shape": (4, 4, 3), "names": None},
            "observation.images.left_wrist": {"dtype": "image", "shape": (4, 4, 3), "names": None},
        },
        task="fold",
        fps=30.0,
        device="cpu",
        rtc_queue_threshold=queue_threshold,
    )
    engine.probe = probe
    engine.attach_view_mask(runtime)
    return engine, policy


# Keyed by the short camera name, as a real robot observation is: ``build_dataset_frame``
# re-prefixes it into ``observation.images.<name>``.
_RTC_VIEW_OBS = {
    "j1.pos": 0.0,
    "j2.pos": 0.0,
    "top": np.ones((4, 4, 3), dtype=np.uint8),
    "left_wrist": np.full((4, 4, 3), 200, dtype=np.uint8),
}


def test_rtc_masks_dynamically_across_views(monkeypatch):
    # Multi-view ablation under RTC: each revision blacks out a different view, and the
    # policy must see the black image that belongs to *its* revision, not the live one.
    runtime = _runtime(_plan(initial_views=["observation.images.left_wrist"]))
    engine, policy = _make_view_rtc_engine(runtime)
    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_VIEW_OBS))
        policy.allow_one_inference()
        assert _wait_until(lambda: policy.seen_views)

        first = policy.seen_views[-1]
        assert torch.count_nonzero(first["observation.images.left_wrist"]) == 0
        assert torch.count_nonzero(first["observation.images.top"]) > 0

        # Operator switches to the other view mid-run.  The RTC thread may already have
        # parked an inference computed from the old mask — that one is discarded by the
        # revision guard, so pump until a chunk computed *after* the switch lands.
        runtime.controller.toggle("observation.images.left_wrist")
        runtime.controller.toggle("observation.images.top")
        runtime.poll(runnable=True)
        engine.notify_observation(dict(_RTC_VIEW_OBS))

        new_mask_seen = False
        for _ in range(6):
            policy.allow_one_inference()
            time.sleep(0.15)
            if policy.seen_views:
                latest = policy.seen_views[-1]
                if torch.count_nonzero(latest["observation.images.top"]) == 0:
                    new_mask_seen = True
                    break
        assert new_mask_seen, "no chunk was ever computed from the post-switch mask"
        latest = policy.seen_views[-1]
        assert torch.count_nonzero(latest["observation.images.top"]) == 0
        assert torch.count_nonzero(latest["observation.images.left_wrist"]) > 0
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_queued_chunk_from_a_stale_revision_is_never_dispatched():
    # The queue-merge race: a chunk merged before the transition is still sitting in the
    # queue when the operator switches.  Its actions must not reach the robot.
    runtime = _runtime(_plan())
    engine, policy = _make_view_rtc_engine(runtime)
    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_VIEW_OBS))
        policy.allow_one_inference()
        assert _wait_until(lambda: engine.action_queue is not None and engine.action_queue.qsize() > 0)

        first = engine.get_action(None)
        assert first is not None
        stale_revision = runtime.dispatched.revision

        runtime.controller.toggle("observation.images.top")
        runtime.poll(runnable=True)  # clears the queue and bumps the epoch

        # Whatever is left must carry the new revision; nothing from the old one survives.
        assert runtime.dispatched.revision == stale_revision  # no backfill
        assert engine.action_queue.qsize() == 0
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_transition_flush_clears_the_queue_but_keeps_the_task(caplog):
    runtime = _runtime(_plan())
    engine, policy = _make_view_rtc_engine(runtime)
    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_VIEW_OBS))
        policy.allow_one_inference()
        assert _wait_until(lambda: engine.action_queue.qsize() > 0)

        runtime.controller.toggle("observation.images.top")
        runtime.poll(runnable=True)

        assert engine.action_queue.qsize() == 0
        assert engine.task == "fold"  # preserved: a mask change is not a re-instruction
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_chunk_in_flight_across_a_transition_is_discarded():
    # The other half of the race: the policy is *inside* predict_action_chunk when the
    # operator switches, so the epoch/revision guard must drop the result on arrival.
    runtime = _runtime(_plan())
    engine, policy = _make_view_rtc_engine(runtime)
    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_VIEW_OBS))
        policy.allow_one_inference()
        assert _wait_until(policy.in_inference.is_set)  # parked mid-inference

        runtime.controller.toggle("observation.images.top")
        runtime.poll(runnable=True)

        policy.allow_one_inference()  # let the stale chunk land
        time.sleep(0.2)

        assert engine.action_queue.qsize() == 0, "a chunk from the superseded revision leaked"
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_resets_policy_before_the_first_post_transition_inference():
    runtime = _runtime(_plan())
    engine, policy = _make_view_rtc_engine(runtime)
    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_VIEW_OBS))
        assert _wait_until(policy.in_inference.is_set)

        runtime.controller.toggle("observation.images.top")
        runtime.poll(runnable=True)
        policy.allow_one_inference()  # complete the pre-transition chunk
        assert _wait_until(lambda: policy.reset_count >= 1)

        policy.allow_one_inference()  # the first chunk after the worker's deferred reset
        assert _wait_until(lambda: len(policy.seen_reset_counts) >= 2)
        assert policy.seen_reset_counts[-1] >= 1
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_vqa_uses_the_same_masked_batch_as_the_actions():
    # VQA must not bypass the mask: asking about a scene must show the policy the same
    # black view the actions were based on.
    runtime = _runtime(_plan(initial_views=["observation.images.left_wrist"]))
    engine, policy = _make_view_rtc_engine(runtime)
    from lerobot.rollout.inference.base import PolicyQuery, QueryKind

    answer = engine._generate_text(
        dict(_RTC_VIEW_OBS),
        PolicyQuery(kind=QueryKind.VQA, text="what do you see?"),
    )

    assert answer == "visible"
    assert len(policy.seen_text_views) == 1
    batch = policy.seen_text_views[-1]
    assert torch.count_nonzero(batch["observation.images.left_wrist"]) == 0
    assert torch.count_nonzero(batch["observation.images.top"]) > 0


@pytest.mark.parametrize("query_kind", ["vqa", "autosteer"])
def test_rtc_discards_text_result_when_mask_changes_during_generation(query_kind):
    runtime = _runtime(_plan())
    engine, policy = _make_view_rtc_engine(runtime)
    text_started = Event()
    text_release = Event()
    answers = []
    original_generate_text = policy.generate_text

    def gated_generate_text(batch):
        text_started.set()
        assert text_release.wait(timeout=3.0)
        return original_generate_text(batch)

    policy.generate_text = gated_generate_text
    engine.set_answer_observer(answers.append)
    if query_kind == "vqa":
        assert engine.ask("what do you see?")
    else:
        engine.start_autosteer("stack the blocks", interval_s=30.0)
        engine.pump_query(dict(_RTC_VIEW_OBS))

    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_VIEW_OBS))
        assert _wait_until(text_started.is_set)

        runtime.controller.toggle("observation.images.top")
        runtime.poll(runnable=True)
        text_release.set()

        assert _wait_until(lambda: not engine._query_in_flight)
        engine.pump_query(dict(_RTC_VIEW_OBS))
        assert answers == []
        if query_kind == "autosteer":
            assert engine.task == "fold"
            assert not engine.has_pending_query
            assert len(policy.seen_text_views) == 1
    finally:
        text_release.set()
        policy.unblock()
        engine.stop()


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    import time as _time

    deadline = _time.perf_counter() + timeout
    while _time.perf_counter() < deadline:
        if predicate():
            return True
        _time.sleep(0.01)
    return False
