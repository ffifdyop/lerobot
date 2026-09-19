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

"""View-mask controller: revisioned, latest-wins ablation state.

One object owns the whole View Mask Control lifecycle: the launch/target mask, the
monotonic revision, the :class:`MaskApplicationState`, the deferred keyboard toggles
and the per-revision immutable snapshot that inference and recording consume.

Deliberately orthogonal to the rollout's Control Phase (Policy/Human/Paused): a mask
transition is never a phase change, and a phase change is never a mask transition.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Protocol

from .config import ViewMaskPlan
from .operator import FullFrameViewMask, ViewMaskOperator

logger = logging.getLogger(__name__)


class MaskApplicationState(Enum):
    """Whether the newest View Mask revision has actually reached the policy."""

    ACTIVE = "active"
    """The newest revision produced an executed action (or no transition is pending)."""

    APPLYING = "applying"
    """A transition is in flight: old actions/history were flushed, awaiting a fresh action."""

    FAILED = "failed"
    """Latched: no fresh action arrived in time, or inference failed. Only /reset or a restart clears it."""


@dataclass(frozen=True)
class ViewMaskSnapshot:
    """Everything a single inference needs to know about the active View Mask.

    Immutable and cheap to pass around: inference captures one at the start of a
    call and hands it to the preprocessor, so a mask change mid-inference cannot
    relabel the resulting action.
    """

    revision: int
    operator: ViewMaskOperator
    all_views: tuple[str, ...]

    @property
    def masked_views(self) -> frozenset[str]:
        """Canonical views affected by the spatial operator."""
        return self.operator.masked_views

    def as_vector(self) -> tuple[bool, ...]:
        """Per-canonical-view mask in ``all_views`` order (the dataset field order)."""
        return tuple(view in self.masked_views for view in self.all_views)


@dataclass(frozen=True)
class MaskTransition:
    """A committed mask change, describing the state it moved *from* and *to*."""

    old_revision: int
    new_revision: int
    old_views: frozenset[str]
    new_views: frozenset[str]
    old_operator: ViewMaskOperator
    new_operator: ViewMaskOperator
    source: str
    state: MaskApplicationState


class ViewMaskTransitionSink(Protocol):
    """What the controller calls when a transition commits.

    Implemented by the inference engines: the flush must invalidate every action,
    queued chunk, interpolated step and in-flight inference belonging to the old
    revision.  Kept as a structural protocol so the controller never imports torch.
    """

    def on_mask_transition(self, transition: MaskTransition) -> None:
        """Discard all state computed under ``transition.old_revision``."""


class ViewMaskController:
    """Thread-safe owner of the current View Mask and its revision.

    Threading contract:

    * :meth:`toggle` / :meth:`request_mask` are called from the keyboard (or any
      other) thread and only queue intent.
    * :meth:`tick` and :meth:`drain_transition` are called from the control-loop
      thread, which is the only one that commits a new revision — so revisions are
      strictly increasing and never reused.
    * :meth:`snapshot` / :meth:`state` / :meth:`revision` are safe from everywhere.

    ``timeout_s`` is a budget of *runnable* policy inference, not wall-clock time:
    the control loop reports through ``runnable`` whether inference could actually
    have produced a fresh action this tick, so an operator pre-setting a mask while
    the robot is idle or under human control cannot cause a false failure.
    """

    def __init__(
        self,
        plan: ViewMaskPlan,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.perf_counter,
        operator_factory: Callable[[frozenset[str]], ViewMaskOperator] = FullFrameViewMask,
    ) -> None:
        self._plan = plan
        self._timeout_s = timeout_s
        self._clock = clock
        self._operator_factory = operator_factory

        self._lock = Lock()
        self._launch_views = frozenset(plan.initial_views)
        self._target_views = frozenset(plan.initial_views)
        self._target_operator = self._make_operator(self._target_views)
        self._revision = 0
        self._state = MaskApplicationState.ACTIVE
        self._pending: list[str] = []
        # Explicit ``request_views`` target (``/reset``): queued separately from the
        # toggle edges so a reset and a keypress landing in the same tick cannot
        # silently interleave into a third, unintended set.
        self._requested: frozenset[str] | None = None
        self._requested_source = "operator"
        # Timeout bookkeeping: the budget measures *runnable* policy time, so it is a
        # running total of runnable intervals rather than a wall-clock deadline.  A
        # non-runnable stretch freezes the total instead of restarting it, and an
        # idle operator pre-setting a mask can therefore never cause a false failure.
        self._runnable_elapsed = 0.0
        self._last_runnable_at: float | None = None
        self._failure_reason: str | None = None
        self._sink: ViewMaskTransitionSink | None = None

    # ------------------------------------------------------------------
    # Introspection (any thread)
    # ------------------------------------------------------------------

    @property
    def plan(self) -> ViewMaskPlan:
        return self._plan

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def state(self) -> MaskApplicationState:
        with self._lock:
            return self._state

    @property
    def launch_views(self) -> frozenset[str]:
        """The mask the CLI launched with; explicit ``/reset`` restores it."""
        with self._lock:
            return self._launch_views

    @property
    def target_views(self) -> frozenset[str]:
        """The mask the newest revision requests (equal to the applied mask)."""
        with self._lock:
            return self._target_views

    @property
    def failure_reason(self) -> str | None:
        with self._lock:
            return self._failure_reason

    @property
    def blocked(self) -> bool:
        """True while the policy must not be trusted: APPLYING or FAILED.

        Controls gate VQA, autosteer and *new training frames* on this; it is the
        single place the "can the policy act on this mask yet?" question is answered.
        """
        with self._lock:
            return self._state is not MaskApplicationState.ACTIVE

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def snapshot(self) -> ViewMaskSnapshot:
        """The current immutable mask revision."""
        with self._lock:
            return ViewMaskSnapshot(
                revision=self._revision,
                operator=self._target_operator,
                all_views=self._plan.canonical_views,
            )

    def _make_operator(self, views: frozenset[str]) -> ViewMaskOperator:
        """Build the immutable operator carried by snapshots for ``views``."""
        operator = self._operator_factory(views)
        if operator.masked_views != views:
            raise ValueError(
                "View Mask operator factory returned mismatched views: "
                f"expected {sorted(views)}, got {sorted(operator.masked_views)}"
            )
        return operator

    def is_current(self, revision: int) -> bool:
        """Whether ``revision`` is still the executable one (stale results are dropped)."""
        with self._lock:
            return revision == self._revision

    def is_executable(self, revision: int) -> bool:
        """Whether an action produced under ``revision`` may still reach the robot."""
        with self._lock:
            return revision == self._revision and self._state is not MaskApplicationState.FAILED

    # ------------------------------------------------------------------
    # Intent (any thread)
    # ------------------------------------------------------------------

    def set_sink(self, sink: ViewMaskTransitionSink | None) -> None:
        """Attach the flush hook (the inference engine)."""
        with self._lock:
            self._sink = sink

    def toggle(self, canonical_view: str) -> bool:
        """Queue a latch toggle for ``canonical_view`` (key-down edge semantics).

        Returns ``True`` when the view is a known canonical view.  Latched rather
        than level-triggered: one discrete press adds it, the next removes it, and a
        held key's auto-repeat must be filtered by the input layer.
        """
        if canonical_view not in self._plan.canonical_views:
            logger.warning("Ignoring toggle for unknown view %r", canonical_view)
            return False
        with self._lock:
            self._pending.append(canonical_view)
        return True

    def request_views(self, views: frozenset[str], *, source: str = "operator") -> None:
        """Request an explicit mask set (used by ``/reset``); committed on the next tick."""
        unknown = sorted(views - set(self._plan.canonical_views))
        if unknown:
            raise ValueError(f"Unknown views requested for the mask: {unknown}")
        with self._lock:
            # An explicit request supersedes queued toggle edges: the operator asked
            # for a definite set, not for the set plus whatever keys were in flight.
            self._pending = []
            self._requested = frozenset(views)
            self._requested_source = source

    # ------------------------------------------------------------------
    # Control-loop side
    # ------------------------------------------------------------------

    def tick(self, *, runnable: bool, now: float | None = None) -> None:
        """Accumulate the transition timeout.  Control loop only.

        ``runnable`` is True only when the policy inference had a real opportunity to
        produce an action this tick (segment running, engine resumed, Control Source is
        Policy).  The budget is *runnable* time, not wall-clock: non-runnable ticks add
        nothing, and the deadline is recomputed from the accumulated total on the next
        runnable tick — so an idle stretch neither fails the mask nor silently resets
        the budget it had already consumed.
        """
        now = self._clock() if now is None else now
        with self._lock:
            if self._state is not MaskApplicationState.APPLYING:
                return
            if not runnable:
                # Idle / Human / Paused: freeze the budget rather than fail.
                self._last_runnable_at = None
                return
            if self._last_runnable_at is not None:
                self._runnable_elapsed += max(0.0, now - self._last_runnable_at)
            self._last_runnable_at = now
            if self._runnable_elapsed >= self._timeout_s:
                self._fail_locked(
                    f"no action with view mask revision {self._revision} arrived within "
                    f"{self._timeout_s:.1f}s of runnable policy inference"
                )

    def drain_transition(self, *, source: str = "hotkey") -> MaskTransition | None:
        """Commit at most one transition from all queued toggles.  Control loop only.

        Multiple keys received before this call are merged into a *single* revision
        and a single Inference Flush — the operator's intent for one tick is one
        state change, not one per key.  Returns ``None`` when nothing changed (and
        never bumps the revision in that case), or when the controller is FAILED.
        """
        with self._lock:
            pending, self._pending = self._pending, []
            requested, self._requested = self._requested, None
            if requested is not None:
                source = self._requested_source
            if self._state is MaskApplicationState.FAILED and requested is None:
                if pending:
                    logger.warning("Ignoring view-mask toggle: the mask is in FAILED state")
                return None

            if requested is not None:
                new_views = requested
            else:
                targets = set(self._target_views)
                for view in pending:
                    if view in targets:
                        targets.remove(view)
                    else:
                        targets.add(view)
                new_views = frozenset(targets)
            if new_views == self._target_views:
                return None  # no actual change: revision must not advance

            new_operator = self._make_operator(new_views)
            old_views, old_revision = self._target_views, self._revision
            old_operator = self._target_operator
            self._target_views = new_views
            self._target_operator = new_operator
            self._revision += 1
            self._state = MaskApplicationState.APPLYING
            self._failure_reason = None
            self._runnable_elapsed = 0.0
            self._last_runnable_at = None  # armed on the first runnable tick
            transition = MaskTransition(
                old_revision=old_revision,
                new_revision=self._revision,
                old_views=old_views,
                new_views=new_views,
                old_operator=old_operator,
                new_operator=new_operator,
                source=source,
                state=self._state,
            )
            sink = self._sink
        logger.info(
            "View mask transition: revision %d -> %d, masked %s -> %s (source=%s)",
            transition.old_revision,
            transition.new_revision,
            sorted(old_views) or ["(none)"],
            sorted(new_views) or ["(none)"],
            source,
        )
        if sink is not None:
            # Outside the lock: the flush touches engine/queue state and must not
            # block a keyboard toggle or a snapshot read behind it.
            sink.on_mask_transition(transition)
        return transition

    def restore_launch_mask(self) -> MaskTransition | None:
        """Explicit session reset: go back to the launch mask at a *new* revision.

        The revision never returns to 0 — a result produced under an older revision
        must stay recognisably stale, not become current again.
        """
        with self._lock:
            recovered = self._state is MaskApplicationState.FAILED
            if recovered:
                # A reset is the documented recovery from FAILED.
                self._failure_reason = None
            if self._target_views == self._launch_views:
                # Same set, but FAILED must still be cleared by an explicit reset.
                self._state = MaskApplicationState.ACTIVE
                self._runnable_elapsed = 0.0
                self._last_runnable_at = None
                logger.info(
                    "View mask reset: launch mask already in effect (revision %d)%s",
                    self._revision,
                    " — FAILED cleared" if recovered else "",
                )
                return None
        self.request_views(self._launch_views, source="reset")
        return self.drain_transition(source="reset")

    def mark_active(self, revision: int) -> bool:
        """Enter ACTIVE when a fresh action for ``revision`` is executable.

        Returns ``True`` when the state actually changed (so callers can log the
        edge exactly once).  A stale revision is ignored: a late result from an
        older transition must never advertise the newest mask as applied.
        """
        with self._lock:
            if revision != self._revision:
                return False
            if self._state is MaskApplicationState.ACTIVE:
                return False
            if self._state is MaskApplicationState.FAILED:
                return False
            self._state = MaskApplicationState.ACTIVE
            self._runnable_elapsed = 0.0
            self._last_runnable_at = None
            logger.info("View mask revision %d is ACTIVE (fresh action reached the policy)", revision)
            return True

    def mark_failed(self, reason: str) -> bool:
        """Latch FAILED.  Idempotent; returns True on the transition into FAILED."""
        with self._lock:
            if self._state is MaskApplicationState.FAILED:
                return False
            self._fail_locked(reason)
            return True

    def _fail_locked(self, reason: str) -> None:
        logger.error("View mask FAILED (revision %d): %s", self._revision, reason)
        self._state = MaskApplicationState.FAILED
        self._failure_reason = reason
        self._last_runnable_at = None

    def pending_toggles(self) -> int:
        """Number of queued but uncommitted toggles (telemetry/tests)."""
        with self._lock:
            return len(self._pending) + (1 if self._requested is not None else 0)
