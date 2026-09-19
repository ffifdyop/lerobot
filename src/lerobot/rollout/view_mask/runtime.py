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

"""Session-scoped View Mask runtime: controller + hotkeys + sidecar, one entry point.

Strategies and the interactive controller talk to this object instead of to the
controller, the keyboard listener and the recorder separately, so the whole
capability stays behind a small surface:

* :meth:`poll` — one call per control tick: accumulate the transition timeout and
  commit at most one queued mask change.
* :meth:`blocked` — whether the policy may currently be trusted (APPLYING/FAILED).
* :meth:`frame_fields` — the per-frame dataset fields a recording strategy writes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np

from .config import ViewMaskPlan
from .controller import (
    MaskApplicationState,
    MaskTransition,
    ViewMaskController,
    ViewMaskSnapshot,
)
from .events import ViewMaskEventRecorder

logger = logging.getLogger(__name__)


class ViewMaskRuntime:
    """Owns everything one rollout session needs for view ablation.

    Only constructed when ``--view_mask.enabled=true``; when disabled the
    strategies see ``ctx.policy.view_mask is None`` and every code path is a no-op,
    so an ordinary rollout is byte-for-byte unchanged.
    """

    def __init__(
        self,
        plan: ViewMaskPlan,
        controller: ViewMaskController,
        recorder: ViewMaskEventRecorder | None,
        *,
        listener: Any | None = None,
    ) -> None:
        self.plan = plan
        self.controller = controller
        self.recorder = recorder
        self._listener = listener
        # Extra flushes to run on a transition, appended by whoever owns the
        # corresponding state (the strategy owns the action interpolator).
        self._flush_hooks: list[Callable[[MaskTransition], None]] = []
        self._dispatched: ViewMaskSnapshot = controller.snapshot()
        # FAILED is an edge-triggered state.  Keep the last revision whose failure
        # event was written so a blocked control loop cannot append one JSONL row per
        # tick.
        self._failed_event_revision: int | None = None
        self._failed_event_lock = Lock()
        # Last action the robot actually received, kept across a transition so the
        # control loop can hold position instead of going uncommanded.
        self.hold_action = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add_flush_hook(self, hook: Callable[[MaskTransition], None]) -> None:
        """Register extra state to clear on every transition (e.g. the interpolator)."""
        self._flush_hooks.append(hook)

    def start(self) -> None:
        """Start the hotkey listener, if any."""
        if self._listener is not None:
            logger.info(
                "View ablation ready: masked %s, hotkeys %s",
                sorted(self.controller.target_views) or ["(none)"],
                dict(self.controller.plan.key_to_view),
            )

    def stop(self) -> None:
        """Stop the hotkey listener and close the sidecar."""
        if self._listener is not None:
            stop = getattr(self._listener, "stop", None)
            if callable(stop):
                stop()
            self._listener = None
        if self.recorder is not None:
            self.recorder.close()

    # ------------------------------------------------------------------
    # Control-loop surface
    # ------------------------------------------------------------------

    @property
    def blocked(self) -> bool:
        """True while the newest revision is not yet ACTIVE (APPLYING or FAILED)."""
        return self.controller.blocked

    @property
    def revision(self) -> int:
        return self.controller.revision

    def snapshot(self) -> ViewMaskSnapshot:
        """The current immutable mask revision (same object the engine stamps)."""
        return self.controller.snapshot()

    def is_executable(self, revision: int) -> bool:
        """Whether an action produced under ``revision`` may still reach the robot."""
        return self.controller.is_executable(revision)

    def is_current(self, revision: int) -> bool:
        """Whether ``revision`` is still the newest one (stale results are dropped)."""
        return self.controller.is_current(revision)

    @property
    def failed(self) -> bool:
        return self.controller.state is MaskApplicationState.FAILED

    @property
    def failure_reason(self) -> str | None:
        """Reason for the latched failure, for interactive status reporting."""
        return self.controller.failure_reason

    def poll(self, *, runnable: bool, phase: str | None = None) -> MaskTransition | None:
        """One control tick: accumulate the timeout, then commit a pending change.

        ``runnable`` is True only when the policy could actually have produced a
        fresh action this tick.  Ordering matters: the timeout is charged *before*
        the new transition is committed, so a tick that both commits and would
        otherwise be due can never fail the revision it just created.
        """
        if phase is not None and self.recorder is not None:
            self.recorder.note_phase(phase)
        self.controller.tick(runnable=runnable)
        transition = self.controller.drain_transition()
        if transition is not None:
            self._on_transition(transition)
        self._record_failed_once()
        return transition

    def _on_transition(self, transition: MaskTransition) -> None:
        for hook in self._flush_hooks:
            try:
                hook(transition)
            except Exception:  # a broken flush must not take the control loop down
                logger.exception("View mask flush hook failed")
        # ``dispatched`` deliberately keeps describing the *last action that ran*, which
        # was generated under the old revision: nothing has been dispatched under the new
        # one yet.  It is refreshed by the engine when a fresh action is returned, so a
        # frame recorded during APPLYING could never be mislabelled with the new mask —
        # and APPLYING frames are not recorded at all.
        if self.recorder is not None:
            self.recorder.record_transition(transition)

    def note_frame(self, *, episode_index: int | None, frame_index: int | None) -> None:
        """Record the dataset's current position so the next event can be aligned to it."""
        if self.recorder is not None:
            self.recorder.note_frame(episode_index=episode_index, frame_index=frame_index)

    def note_phase(self, phase: str | None) -> None:
        """Record the current Control Phase, when the strategy has that vocabulary."""
        if self.recorder is not None:
            self.recorder.note_phase(phase)

    def mark_active(self, snapshot: ViewMaskSnapshot) -> None:
        """Called by the inference engine when a fresh action for ``snapshot`` is executable."""
        if self.controller.mark_active(snapshot.revision) and self.recorder is not None:
            self.recorder.record_state(MaskApplicationState.ACTIVE, revision=self.controller.revision)

    def mark_failed(self, reason: str) -> None:
        """Called by the inference engine on an unrecoverable inference error."""
        self.controller.mark_failed(reason)
        self._record_failed_once()

    def _record_failed_once(self) -> None:
        """Write one FAILED event per revision, even when two threads observe the edge."""
        if self.recorder is None or not self.failed:
            return
        revision = self.controller.revision
        with self._failed_event_lock:
            if not self.failed or self.controller.revision != revision:
                return
            if self._failed_event_revision == revision:
                return
            # Claim the edge before writing: the recorder itself is thread-safe, and
            # this ordering prevents poll() and an inference callback from both
            # appending the same failure when they race after mark_failed().
            self._failed_event_revision = revision
            reason = self.controller.failure_reason
        self.recorder.record_state(
            MaskApplicationState.FAILED,
            revision=revision,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Provenance for recording
    # ------------------------------------------------------------------

    @property
    def dispatched(self) -> ViewMaskSnapshot:
        """Snapshot of the most recently dispatched action (Policy frames use this)."""
        return self._dispatched

    def set_dispatched(self, snapshot: ViewMaskSnapshot) -> None:
        """Record the snapshot an action was generated under (called by the engine)."""
        self._dispatched = snapshot

    @property
    def current(self) -> ViewMaskSnapshot:
        """The controller's live snapshot (Human frames use this — no action involved)."""
        return self.controller.snapshot()

    def frame_fields(self, snapshot: ViewMaskSnapshot) -> dict[str, Any]:
        """The dataset fields describing the mask an action was taken under."""
        return {
            "view_mask": np.array(snapshot.as_vector(), dtype=bool),
            "view_mask_revision": np.array([snapshot.revision], dtype=np.int64),
        }

    def restore_launch_mask(self) -> MaskTransition | None:
        """Explicit session reset: back to the CLI launch mask, at a new revision."""
        transition = self.controller.restore_launch_mask()
        if not self.failed:
            # A reset can clear FAILED without changing the mask/revision.  Permit a
            # later failure of that same revision to produce a new edge event.
            with self._failed_event_lock:
                self._failed_event_revision = None
        if transition is not None:
            self._on_transition(transition)
        return transition

    def telemetry(self) -> dict[str, Any]:
        """Per-view mask state for visualization backends (display-only)."""
        snapshot = self.current
        return {
            "revision": snapshot.revision,
            "state": self.controller.state.value,
            "masked": {view: view in snapshot.masked_views for view in self.plan.canonical_views},
            "display_paths": dict(self.plan.canonical_to_raw),
        }
