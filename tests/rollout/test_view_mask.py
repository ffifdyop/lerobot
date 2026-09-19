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

"""Unit tests for View Ablation: config, name resolution, black-view step, controller.

Covers ``add-rollout-view-ablation`` tasks 1.4, 2.5, 3.7 (controller half) and the
disabled-by-default / no-side-effect guarantees they all rest on.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from lerobot.rollout.view_mask.config import (
    ViewMaskConfig,
    build_view_mask_plan,
    validate_hotkey_conflicts,
    validate_view_mask,
)
from lerobot.rollout.view_mask.controller import (
    MaskApplicationState,
    ViewMaskController,
)
from lerobot.rollout.view_mask.events import ViewMaskEventRecorder
from lerobot.rollout.view_mask.operator import FullFrameViewMask
from lerobot.rollout.view_mask.processor import ViewMaskProcessorStep
from lerobot.rollout.view_mask.runtime import ViewMaskRuntime

# The three cameras of the DM05 deployment: canonical policy keys in declaration order.
POLICY_VIEWS = [
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
ROBOT_CAMERAS = ["top", "left_wrist", "right_wrist"]


def _plan(**kwargs):
    return build_view_mask_plan(
        ViewMaskConfig(enabled=True, **kwargs),
        policy_view_keys=POLICY_VIEWS,
        robot_camera_keys=ROBOT_CAMERAS,
    )


def _controller(plan, *, timeout_s=10.0, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    return ViewMaskController(plan, timeout_s=timeout_s, **kwargs)


# ---------------------------------------------------------------------------
# 1.1 / 1.2 — config validation and name resolution
# ---------------------------------------------------------------------------


def test_default_config_is_disabled_and_inert():
    cfg = ViewMaskConfig()
    assert cfg.enabled is False
    assert cfg.is_default
    validate_view_mask(cfg)  # no raise


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_views": ["observation.images.top"]},
        {"hotkeys": {"observation.images.top": "f1"}},
        {"transition_timeout_s": 5.0},
        {"event_log_path": "events.jsonl"},
    ],
)
def test_sub_options_without_explicit_enable_are_rejected(overrides):
    with pytest.raises(ValueError, match="--view_mask.enabled=false"):
        validate_view_mask(ViewMaskConfig(**overrides))


def test_explicitly_disabled_default_values_are_accepted():
    # draccus cannot distinguish "unset" from "set to the default", and both are inert.
    validate_view_mask(ViewMaskConfig(enabled=False, transition_timeout_s=10.0))


def test_enabled_config_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="transition_timeout_s must be > 0"):
        validate_view_mask(ViewMaskConfig(enabled=True, transition_timeout_s=0.0))


def test_raw_camera_key_resolves_to_canonical_key():
    plan = _plan(initial_views=["left_wrist"])
    assert plan.initial_views == {"observation.images.left_wrist"}


def test_bare_and_prefixed_names_resolve_to_the_same_view():
    assert (
        _plan(initial_views=["top"]).initial_views
        == _plan(initial_views=["observation.images.top"]).initial_views
    )


def test_rename_map_remaps_raw_camera_keys():
    plan = build_view_mask_plan(
        ViewMaskConfig(enabled=True, initial_views=["top"]),
        policy_view_keys=["observation.images.camera1", "observation.images.camera2"],
        robot_camera_keys=["top", "wrist"],
        rename_map={
            "observation.images.top": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
    )
    assert plan.initial_views == {"observation.images.camera1"}


def test_declaration_order_is_preserved_not_sorted():
    # Sorting would rename the model's visual slots; the policy's order is identity.
    plan = build_view_mask_plan(
        ViewMaskConfig(enabled=True),
        policy_view_keys=["observation.images.wrist", "observation.images.top"],
        robot_camera_keys=["top", "wrist"],
    )
    assert plan.canonical_views == ("observation.images.wrist", "observation.images.top")
    assert plan.default_key_for("observation.images.wrist") == "f1"
    assert plan.default_key_for("observation.images.top") == "f2"


def test_default_hotkeys_follow_canonical_order():
    plan = _plan()
    assert plan.view_to_key == {
        "observation.images.top": "f1",
        "observation.images.left_wrist": "f2",
        "observation.images.right_wrist": "f3",
    }


def test_explicit_hotkeys_override_the_defaults():
    plan = _plan(hotkeys={"left_wrist": "f5"})
    assert plan.view_to_key["observation.images.left_wrist"] == "f5"
    assert plan.view_to_key["observation.images.top"] == "f1"  # untouched default


def test_whole_config_is_rejected_when_one_view_is_unknown():
    # A partially applied mask would ablate the wrong view with no visible symptom.
    with pytest.raises(ValueError, match="Unknown view 'nope'") as exc:
        _plan(initial_views=["top", "nope"])
    assert "observation.images.top" in str(exc.value)  # lists the available views


def test_ambiguous_view_name_is_rejected():
    # "top" names the policy's top view *and* the robot's top camera — and that camera
    # has been renamed into a different slot. Both readings are exact, so the resolver
    # refuses instead of picking one and silently ablating the wrong view.
    with pytest.raises(ValueError, match="Ambiguous view") as exc:
        build_view_mask_plan(
            ViewMaskConfig(enabled=True, initial_views=["top"]),
            policy_view_keys=["observation.images.top", "observation.images.wrist"],
            robot_camera_keys=["top", "wrist"],
            rename_map={"observation.images.top": "observation.images.wrist"},
        )
    # Both readings are named, so the operator can disambiguate by hand.
    assert "observation.images.top" in str(exc.value)
    assert "observation.images.wrist" in str(exc.value)


def test_all_views_masked_is_accepted_without_special_casing():
    plan = _plan(initial_views=["top", "left_wrist", "right_wrist"])
    assert plan.initial_views == set(POLICY_VIEWS)


def test_duplicate_hotkey_binding_is_rejected():
    with pytest.raises(ValueError, match="bound to two views"):
        _plan(hotkeys={"left_wrist": "f1"})  # f1 is already `top`


def test_hotkey_conflicts_are_reported_with_both_sides():
    plan = _plan()
    with pytest.raises(ValueError) as exc:
        validate_hotkey_conflicts(plan, {"f1": "DAgger (pause/resume policy)"})
    message = str(exc.value)
    assert "DAgger (pause/resume policy)" in message
    assert "toggle observation.images.top" in message


def test_no_conflict_is_a_no_op():
    validate_hotkey_conflicts(_plan(), {"space": "DAgger (pause/resume policy)"})


def test_policy_without_visual_features_is_rejected():
    with pytest.raises(ValueError, match="at least one visual input feature"):
        build_view_mask_plan(
            ViewMaskConfig(enabled=True),
            policy_view_keys=[],
            robot_camera_keys=[],
        )


# ---------------------------------------------------------------------------
# 1.3 — the black-view processor step
# ---------------------------------------------------------------------------


class _Snapshot:
    """Structural stand-in for ViewMaskSnapshot (keeps this test torch-only)."""

    def __init__(self, *, revision, masked_views, all_views):
        self.revision = revision
        self.operator = FullFrameViewMask(frozenset(masked_views))
        self.all_views = tuple(all_views)

    @property
    def masked_views(self):
        return self.operator.masked_views


def _batch(*, with_snapshot=True, masked=frozenset({"observation.images.left_wrist"})):
    batch = {
        "observation.images.top": torch.ones(1, 3, 4, 4),
        "observation.images.left_wrist": torch.full((1, 3, 4, 4), 0.5),
        "observation.images.right_wrist": torch.full((1, 3, 4, 4), 0.25),
        "observation.state": torch.ones(1, 6),
        "task": ["fold"],
    }
    if with_snapshot:
        from lerobot.rollout.view_mask.inference import stamp_view_mask

        snapshot = _Snapshot(revision=3, masked_views=masked, all_views=tuple(POLICY_VIEWS))
        stamp_view_mask(batch, snapshot)
    return batch


def _run_step(batch, step=None):
    """Push a batch through a real policy pipeline holding the mask step.

    The step is an ``ObservationProcessorStep``, so it only ever sees the transition's
    observation mapping — driving it through the pipeline is what the rollout does, and
    it keeps the marker-stripping assertion honest about where the strip happens.
    """
    from lerobot.processor import PolicyProcessorPipeline

    pipeline = PolicyProcessorPipeline(steps=[step or ViewMaskProcessorStep()])
    return pipeline(batch)


def test_masked_view_is_zeroed_in_place_of_the_original_and_others_untouched():
    original = _batch()
    top_before = original["observation.images.top"].clone()
    state_before = original["observation.state"].clone()

    out = _run_step(original)

    assert torch.count_nonzero(out["observation.images.left_wrist"]) == 0
    assert torch.equal(out["observation.images.top"], top_before)
    assert torch.equal(out["observation.state"], state_before)


def test_snapshot_markers_are_stripped_before_the_policy_sees_the_batch():
    from lerobot.rollout.view_mask.keys import VIEW_MASK_SNAPSHOT

    out = _run_step(_batch())
    assert VIEW_MASK_SNAPSHOT not in out


def test_original_observation_is_not_mutated():
    # The same dict feeds the dataset and telemetry; blacking it out there would
    # corrupt the recording and hide the scene from the operator.
    original = _batch()
    wrist_before = original["observation.images.left_wrist"].clone()

    _run_step(original)

    assert torch.equal(original["observation.images.left_wrist"], wrist_before)


def test_disabled_step_is_a_no_op_on_the_original_values():
    batch = _batch(with_snapshot=False)
    out = _run_step(batch)
    assert torch.count_nonzero(out["observation.images.left_wrist"]) > 0


def test_unstamped_batch_is_unchanged():
    batch = {"observation.images.top": torch.ones(1, 3, 2, 2)}
    out = _run_step(batch)
    assert torch.count_nonzero(out["observation.images.top"]) > 0


def test_missing_key_stays_missing_rather_than_becoming_a_black_slot():
    out = _run_step(_batch(masked=frozenset({"observation.images.absent"})))
    assert "observation.images.absent" not in out


def test_dtype_shape_and_device_are_preserved():
    batch = _batch()
    batch["observation.images.left_wrist"] = batch["observation.images.left_wrist"].to(torch.float32)
    out = _run_step(batch)
    masked = out["observation.images.left_wrist"]
    assert masked.dtype == batch["observation.images.left_wrist"].dtype
    assert masked.shape == batch["observation.images.left_wrist"].shape
    assert masked.device == batch["observation.images.left_wrist"].device


def test_zeroed_black_frame_survives_renormalization():
    # Black (0 in [0,1] space) runs through normalization like any other frame: the
    # step hands downstream an ordinary tensor, never a gap it would have to special-case.
    out = _run_step(_batch())
    masked = out["observation.images.left_wrist"]
    normalized = (masked - 0.5) / 0.25  # a stand-in normalization
    assert torch.allclose(normalized, torch.full_like(normalized, -2.0))


def test_full_frame_mask_is_the_default_snapshot_operator():
    snapshot = _controller(_plan(initial_views=["top"])).snapshot()

    assert isinstance(snapshot.operator, FullFrameViewMask)
    assert snapshot.masked_views == {"observation.images.top"}


@dataclass(frozen=True)
class _CenterPatchViewMask:
    """Test-only spatial operator proving the processor is not tied to full frames."""

    masked_views: frozenset[str]

    def apply(self, observation):
        targets = [view for view in self.masked_views if view in observation]
        if not targets:
            return observation
        processed = dict(observation)
        for view in targets:
            image = processed[view].clone()
            image[..., 1:3, 1:3] = 0
            processed[view] = image
        return processed

    def to_metadata(self):
        return {
            "type": "center_patch",
            "views": sorted(self.masked_views),
            "bounds": [1, 1, 3, 3],
        }


def test_custom_spatial_operator_uses_the_same_snapshot_and_processor_boundary():
    plan = _plan(initial_views=["left_wrist"])
    controller = ViewMaskController(
        plan,
        timeout_s=10.0,
        operator_factory=_CenterPatchViewMask,
    )
    original = _batch(with_snapshot=False)
    wrist_before = original["observation.images.left_wrist"].clone()

    from lerobot.rollout.view_mask.inference import stamp_view_mask

    stamp_view_mask(original, controller.snapshot())
    out = _run_step(original)

    masked = out["observation.images.left_wrist"]
    assert torch.count_nonzero(masked[..., 1:3, 1:3]) == 0
    assert torch.equal(masked[..., 0, 0], wrist_before[..., 0, 0])
    assert torch.equal(original["observation.images.left_wrist"], wrist_before)
    assert controller.snapshot().operator.to_metadata()["type"] == "center_patch"


# ---------------------------------------------------------------------------
# 2.1 / 2.4 / 3.5 — controller state machine
# ---------------------------------------------------------------------------


class _ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_revision_starts_at_zero_with_the_launch_mask():
    controller = _controller(_plan(initial_views=["top"]))
    snapshot = controller.snapshot()
    assert snapshot.revision == 0
    assert snapshot.masked_views == {"observation.images.top"}
    assert snapshot.as_vector() == (True, False, False)
    assert controller.state is MaskApplicationState.ACTIVE


def test_toggle_commits_a_new_revision_only_after_a_tick():
    controller = _controller(_plan())
    controller.toggle("observation.images.top")
    assert controller.revision == 0  # queued, not yet committed
    transition = controller.drain_transition()
    assert transition is not None
    assert transition.new_revision == 1
    assert controller.state is MaskApplicationState.APPLYING


def test_toggle_adds_then_removes_the_view():
    controller = _controller(_plan())
    controller.toggle("observation.images.top")
    controller.drain_transition()
    assert controller.target_views == {"observation.images.top"}

    controller.toggle("observation.images.top")
    controller.drain_transition()
    assert controller.target_views == frozenset()
    assert controller.revision == 2


def test_no_actual_change_does_not_advance_the_revision():
    controller = _controller(_plan(initial_views=["top"]))
    controller.toggle("observation.images.top")  # toggles it back off
    controller.toggle("observation.images.top")  # and on again -> net zero
    assert controller.drain_transition() is None
    assert controller.revision == 0


def test_multiple_edges_in_one_tick_merge_into_one_revision_and_one_flush():
    class _Sink:
        def __init__(self):
            self.calls = []

        def on_mask_transition(self, transition):
            self.calls.append(transition)

    controller = _controller(_plan())
    sink = _Sink()
    controller.set_sink(sink)
    controller.toggle("observation.images.top")
    controller.toggle("observation.images.left_wrist")

    transition = controller.drain_transition()

    assert transition is not None
    assert controller.revision == 1
    assert controller.target_views == {"observation.images.top", "observation.images.left_wrist"}
    assert len(sink.calls) == 1


def test_unknown_toggle_is_refused_and_changes_nothing():
    controller = _controller(_plan())
    assert controller.toggle("observation.images.nope") is False
    assert controller.drain_transition() is None
    assert controller.revision == 0


def test_latest_wins_while_a_transition_is_applying():
    controller = _controller(_plan())
    controller.toggle("observation.images.top")
    controller.drain_transition()
    assert controller.state is MaskApplicationState.APPLYING

    # A second key while APPLYING supersedes the first transition: higher revision,
    # another flush, and the first one's late result is now unstamped.
    controller.toggle("observation.images.left_wrist")
    transition = controller.drain_transition()

    assert transition.new_revision == 2
    assert transition.old_revision == 1
    assert controller.target_views == {"observation.images.top", "observation.images.left_wrist"}


def test_stale_revision_cannot_mark_itself_active():
    controller = _controller(_plan())
    controller.toggle("observation.images.top")
    controller.drain_transition()  # revision 1
    controller.toggle("observation.images.left_wrist")
    controller.drain_transition()  # revision 2

    assert controller.mark_active(1) is False  # a late result from the superseded revision
    assert controller.state is MaskApplicationState.APPLYING
    assert controller.mark_active(2) is True
    assert controller.state is MaskApplicationState.ACTIVE


def test_timeout_latches_failed_after_the_budget_of_runnable_time():
    clock = _ManualClock()
    controller = _controller(_plan(), timeout_s=10.0, clock=clock)
    controller.toggle("observation.images.top")
    controller.drain_transition()

    controller.tick(runnable=True, now=clock.now)
    clock.advance(9.9)
    controller.tick(runnable=True, now=clock.now)
    assert controller.state is MaskApplicationState.APPLYING

    clock.advance(0.2)
    controller.tick(runnable=True, now=clock.now)
    assert controller.state is MaskApplicationState.FAILED
    assert "within 10.0s" in controller.failure_reason


def test_timeout_does_not_accrue_while_inference_is_paused():
    clock = _ManualClock()
    controller = _controller(_plan(), timeout_s=1.0, clock=clock)
    controller.toggle("observation.images.top")
    controller.drain_transition()

    # Idle / Human / Paused: an operator pre-setting a mask must not cause a failure.
    for _ in range(50):
        clock.advance(1.0)
        controller.tick(runnable=False, now=clock.now)
    assert controller.state is MaskApplicationState.APPLYING

    controller.tick(runnable=True, now=clock.now)
    clock.advance(1.1)
    controller.tick(runnable=True, now=clock.now)
    assert controller.state is MaskApplicationState.FAILED


def test_failed_is_latching_and_refuses_further_toggles():
    controller = _controller(_plan())
    controller.mark_failed("engine died")
    assert controller.state is MaskApplicationState.FAILED
    assert controller.blocked

    controller.toggle("observation.images.top")
    assert controller.drain_transition() is None  # no revision moves while FAILED
    assert controller.mark_active(0) is False


def test_failed_is_idempotent():
    controller = _controller(_plan())
    assert controller.mark_failed("first") is True
    assert controller.mark_failed("second") is False
    assert controller.failure_reason == "first"


def test_reset_restores_the_launch_mask_at_a_new_revision():
    controller = _controller(_plan(initial_views=["top"]))
    controller.toggle("observation.images.top")  # off
    controller.toggle("observation.images.left_wrist")  # and left_wrist on
    controller.drain_transition()
    assert controller.revision == 1

    transition = controller.restore_launch_mask()

    assert transition is not None
    assert transition.new_revision == 2  # never back to 0
    assert controller.target_views == {"observation.images.top"}
    assert controller.revision == 2


def test_reset_with_the_launch_mask_already_in_effect_does_not_bump_the_revision():
    controller = _controller(_plan(initial_views=["top"]))
    assert controller.restore_launch_mask() is None
    assert controller.revision == 0


def test_reset_clears_a_failed_mask():
    controller = _controller(_plan(initial_views=["top"]))
    controller.mark_failed("timeout")
    assert controller.state is MaskApplicationState.FAILED

    controller.restore_launch_mask()

    assert controller.state is MaskApplicationState.ACTIVE
    assert controller.failure_reason is None


def test_snapshot_is_immutable():
    controller = _controller(_plan())
    controller.toggle("observation.images.top")
    controller.drain_transition()
    snapshot = controller.snapshot()
    assert snapshot.masked_views == {"observation.images.top"}

    controller.toggle("observation.images.top")
    controller.drain_transition()
    # The previously handed-out snapshot still describes revision 1.
    assert snapshot.revision == 1
    assert snapshot.masked_views == {"observation.images.top"}


def test_toggles_from_the_keyboard_thread_are_not_lost_mid_transition():
    # The real race: keypresses arrive while the control loop is committing a
    # transition.  Every edge must land in exactly one committed set, never dropped.
    controller = _controller(_plan())
    committed = []

    def press_repeatedly(view):
        for _ in range(20):
            controller.toggle(view)
            committed.append(controller.drain_transition())

    threads = [threading.Thread(target=press_repeatedly, args=(view,)) for view in POLICY_VIEWS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The final set is deterministic even though the interleaving is not: 20 toggles
    # per view is even, so every view ends up out of the mask.
    assert controller.pending_toggles() == 0
    assert controller.target_views == frozenset()
    revisions = [t.new_revision for t in committed if t is not None]
    assert revisions == sorted(set(revisions)), "revisions must be strictly increasing"


# ---------------------------------------------------------------------------
# Runtime: frame provenance, hold value, telemetry
# ---------------------------------------------------------------------------


def _runtime(plan=None, controller=None, recorder=None):
    plan = plan or _plan(initial_views=["top"])
    return ViewMaskRuntime(plan, controller or _controller(plan), recorder)


def test_frame_fields_name_every_canonical_view_in_order():
    runtime = _runtime()
    fields = runtime.frame_fields(runtime.current)
    assert fields["view_mask"].dtype == np.bool_
    assert fields["view_mask"].tolist() == [True, False, False]
    assert fields["view_mask_revision"].dtype == np.int64
    assert fields["view_mask_revision"].tolist() == [0]


def test_dispatched_provenance_tracks_the_action_not_the_live_controller():
    runtime = _runtime()
    dispatched = runtime.dispatched
    runtime.controller.toggle("observation.images.left_wrist")
    runtime.controller.drain_transition()

    # A policy frame uses the action's provenance...
    assert runtime.dispatched.revision == dispatched.revision
    # ...while a Human frame uses the live snapshot.
    assert runtime.current.revision == 1


def test_telemetry_reports_every_view_specifically():
    runtime = _runtime()
    telemetry = runtime.telemetry()
    assert telemetry["masked"] == {
        "observation.images.top": True,
        "observation.images.left_wrist": False,
        "observation.images.right_wrist": False,
    }
    assert telemetry["revision"] == 0
    assert telemetry["state"] == "active"


# ---------------------------------------------------------------------------
# 4.3 / 4.4 — event sidecar
# ---------------------------------------------------------------------------


def test_event_sidecar_records_one_utc_header_then_the_timeline(tmp_path):
    path = tmp_path / "meta" / "view_mask_events.jsonl"
    plan = _plan(initial_views=["top"])
    controller = _controller(plan)
    recorder = ViewMaskEventRecorder(
        path, canonical_views=plan.canonical_views, initial_mask=controller.snapshot()
    )
    runtime = ViewMaskRuntime(plan, controller, recorder)

    controller.toggle("observation.images.top")  # off
    transition = runtime.poll(runnable=True)
    runtime.mark_active(runtime.current)
    recorder.close()

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["event"] == "run_started"
    assert lines[0]["utc_start"].endswith("Z")
    assert lines[0]["initial_mask"] == [True, False, False]
    assert lines[0]["initial_mask_operator"]["type"] == "full_frame"
    assert len([line for line in lines if line["event"] == "run_started"]) == 1

    moved = next(line for line in lines if line["event"] == "transition")
    assert moved["revision"] == transition.new_revision
    assert moved["old_mask"] == [True, False, False]
    assert moved["mask"] == [False, False, False]
    assert moved["mask_operator"] == {"type": "full_frame", "views": []}
    assert moved["t"] >= 0.0  # run-relative monotonic

    states = [line["state"] for line in lines if line["event"] == "state_changed"]
    assert states == ["active"]


def test_event_sidecar_aligns_an_event_between_frames(tmp_path):
    path = tmp_path / "view_mask_events.jsonl"
    plan = _plan()
    controller = _controller(plan)
    recorder = ViewMaskEventRecorder(
        path, canonical_views=plan.canonical_views, initial_mask=controller.snapshot()
    )
    runtime = ViewMaskRuntime(plan, controller, recorder)

    # The operator switched masks after frame 17 of episode 2.
    runtime.note_frame(episode_index=2, frame_index=17)
    runtime.note_phase("policy")
    controller.toggle("observation.images.top")
    runtime.poll(runnable=True)
    recorder.close()

    event = next(json.loads(line) for line in path.read_text().splitlines() if '"transition"' in line)
    assert event["episode_index"] == 2
    assert event["nearest_frame_index"] == 17
    assert event["control_phase"] == "policy"


def test_event_sidecar_failure_disables_recording_without_raising(tmp_path):
    # A directory where the file should be: opening it fails, and the rollout must carry
    # on.  Event recording is a diagnostic sidecar — it may never take the robot down.
    unwritable = tmp_path / "meta"
    unwritable.mkdir()
    plan = _plan()
    controller = _controller(plan)
    recorder = ViewMaskEventRecorder(
        unwritable, canonical_views=plan.canonical_views, initial_mask=controller.snapshot()
    )

    controller.toggle("observation.images.top")
    transition = controller.drain_transition()
    recorder.record_transition(transition)  # must not raise
    recorder.close()


# ---------------------------------------------------------------------------
# Disabled-by-default: no side effects anywhere
# ---------------------------------------------------------------------------


def test_processor_step_is_not_injected_when_disabled():
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.rollout.context import _inject_view_mask_step
    from lerobot.rollout.view_mask.processor import ViewMaskProcessorStep

    pipeline = PolicyProcessorPipeline(steps=[])
    assert not any(isinstance(step, ViewMaskProcessorStep) for step in pipeline.steps)
    # The injector is only called on the enabled path; calling it twice must not duplicate.
    _inject_view_mask_step(pipeline)
    _inject_view_mask_step(pipeline)
    assert sum(isinstance(step, ViewMaskProcessorStep) for step in pipeline.steps) == 1


def test_mask_step_lands_after_rename_and_before_normalization():
    from lerobot.processor import (
        DeviceProcessorStep,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
    )
    from lerobot.rollout.context import _inject_view_mask_step

    pipeline = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            DeviceProcessorStep(device="cpu"),
        ]
    )
    _inject_view_mask_step(pipeline)

    names = [type(step).__name__ for step in pipeline.steps]
    assert names.index("ViewMaskProcessorStep") == 1
    assert names[0] == "RenameObservationsProcessorStep"
    assert names.index("DeviceProcessorStep") > 1


def test_dataset_features_are_absent_when_disabled_and_named_when_enabled():
    from lerobot.rollout.context import _view_mask_dataset_features

    features = _view_mask_dataset_features(POLICY_VIEWS)
    assert features["view_mask"]["dtype"] == "bool"
    assert features["view_mask"]["shape"] == (3,)
    assert features["view_mask"]["names"] == POLICY_VIEWS  # order is slot identity
    assert features["view_mask_revision"]["dtype"] == "int64"
    assert features["view_mask_revision"]["shape"] == (1,)
    # Nothing named view_mask* exists unless this helper is called.
    assert set(features) == {"view_mask", "view_mask_revision"}
