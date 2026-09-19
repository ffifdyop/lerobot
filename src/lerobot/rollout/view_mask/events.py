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

"""JSONL sidecar for View Mask events.

Records the mask *timeline* — launch, every committed transition, every application
state change and every failure — so an experiment can be reconstructed even though
the dataset only holds training frames.  A transition that lands between two frames
has no frame to attach to, which is exactly why this is a separate file.

Fault policy: a broken writer disables the recorder and logs once, mirroring the
recording system's existing behaviour.  It must never invent robot-control
semantics, so a failure here never stops the rollout.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from threading import RLock
from typing import Any

from .controller import MaskApplicationState, MaskTransition, ViewMaskSnapshot

logger = logging.getLogger(__name__)

RUN_STARTED = "run_started"
TRANSITION = "transition"
STATE_CHANGED = "state_changed"
FAILED = "failed"

DEFAULT_EVENT_FILENAME = "view_mask_events.jsonl"


def default_event_log_path(dataset_root: str | Path | None) -> Path:
    """Resolve the sidecar path: dataset ``meta/`` when recording, else the rollout output dir.

    Recording runs keep their event log next to the data it explains, so moving or
    uploading the dataset carries the timeline with it.  A Base rollout has no
    dataset, so it gets the same timestamped ``outputs/rollout/`` convention its
    other artifacts use.
    """
    if dataset_root is not None:
        return Path(dataset_root) / "meta" / DEFAULT_EVENT_FILENAME
    return Path("outputs") / "rollout" / time.strftime("%Y%m%d_%H%M%S") / DEFAULT_EVENT_FILENAME


class ViewMaskEventRecorder:
    """Append-only JSONL writer for the View Mask timeline.

    Thread-safe: the control loop (transitions, state changes) and an inference
    callback may both append.  Events are flushed per line so a killed process
    still leaves a readable timeline — the whole point of the file is post-hoc
    debugging of a session that may not have ended cleanly.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        canonical_views: tuple[str, ...],
        initial_mask: ViewMaskSnapshot,
        source: str = "cli",
    ) -> None:
        self._path = Path(path)
        self._canonical_views = tuple(canonical_views)
        self._lock = RLock()
        self._disabled = False
        self._started = time.monotonic()
        self._fh = None
        self._failure_count = 0
        # Alignment hints, refreshed by the control loop each recorded frame.
        self._episode_index: int | None = None
        self._frame_index: int | None = None
        self._phase: str | None = None

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")
        except OSError as e:
            self._disable(f"could not open {self._path}: {e}")
            return

        self._write(
            {
                "event": RUN_STARTED,
                "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "canonical_views": list(self._canonical_views),
                "initial_mask": initial_mask.as_vector(),
                "initial_mask_operator": initial_mask.operator.to_metadata(),
                "initial_revision": initial_mask.revision,
                "source": source,
            }
        )

    # ------------------------------------------------------------------
    # Alignment hints (control loop only)
    # ------------------------------------------------------------------

    def note_frame(self, *, episode_index: int | None, frame_index: int | None) -> None:
        """Record where the dataset timeline currently is, for the next event."""
        with self._lock:
            self._episode_index = episode_index
            self._frame_index = frame_index

    def note_phase(self, phase: str | None) -> None:
        """Record the current Control Phase (the human-takeover vocabulary), when available."""
        with self._lock:
            self._phase = phase

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def record_transition(self, transition: MaskTransition) -> None:
        """Append a committed mask transition."""
        with self._lock:
            self._write(
                {
                    **self._alignment(),
                    "event": TRANSITION,
                    "revision": transition.new_revision,
                    "old_revision": transition.old_revision,
                    "mask": _vector(transition.new_views, self._canonical_views),
                    "old_mask": _vector(transition.old_views, self._canonical_views),
                    "masked_views": sorted(transition.new_views),
                    "old_masked_views": sorted(transition.old_views),
                    "mask_operator": transition.new_operator.to_metadata(),
                    "old_mask_operator": transition.old_operator.to_metadata(),
                    "state": transition.state.value,
                    "source": transition.source,
                }
            )

    def record_state(self, state: MaskApplicationState, *, revision: int, reason: str | None = None) -> None:
        """Append an application-state change (APPLYING → ACTIVE, or → FAILED)."""
        event = FAILED if state is MaskApplicationState.FAILED else STATE_CHANGED
        with self._lock:
            self._write(
                {
                    **self._alignment(),
                    "event": event,
                    "revision": revision,
                    "state": state.value,
                    "reason": reason,
                }
            )

    def close(self) -> None:
        """Flush and close. Idempotent."""
        with self._lock:
            if self._fh is not None:
                # A close that fails must not mask the failure that got us here (this also
                # runs from _disable, called while handling an unrelated write error).
                with contextlib.suppress(OSError):
                    self._fh.close()
                self._fh = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _alignment(self) -> dict[str, Any]:
        """Fields that let an event be lined up against the dataset/dataset timeline."""
        with self._lock:
            fields: dict[str, Any] = {"t": time.monotonic() - self._started}
            if self._episode_index is not None:
                fields["episode_index"] = self._episode_index
            if self._frame_index is not None:
                fields["nearest_frame_index"] = self._frame_index
            if self._phase is not None:
                fields["control_phase"] = self._phase
            return fields

    def _write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._disabled or self._fh is None:
                return
            try:
                self._fh.write(json.dumps(payload, default=str) + "\n")
                self._fh.flush()
            except (OSError, TypeError, ValueError) as e:
                self._failure_count += 1
                self._disable(f"write failed ({e})")
                if self._failure_count > 1:
                    logger.debug("View mask event recorder write failure #%d", self._failure_count)

    def _disable(self, reason: str) -> None:
        with self._lock:
            self._disabled = True
            self.close()
            logger.warning(
                "View mask event recording disabled: %s. The rollout continues; mask inference, "
                "hotkeys and dataset frames are unaffected.",
                reason,
            )


def _vector(views: frozenset[str], canonical_views: tuple[str, ...]) -> list[bool]:
    return [view in views for view in canonical_views]


def write_event(snapshot: ViewMaskSnapshot, **extra: Any) -> dict[str, Any]:
    """Build an event payload from a snapshot (used by callers without a transition)."""
    return {
        "revision": snapshot.revision,
        "mask": list(snapshot.as_vector()),
        "masked_views": sorted(snapshot.masked_views),
        "mask_operator": snapshot.operator.to_metadata(),
        **extra,
    }


def resolve_event_log_path(configured: str | None, dataset_root: str | Path | None) -> Path:
    """Explicit ``--view_mask.event_log_path`` wins; otherwise the conventional default."""
    if configured:
        return Path(os.path.expanduser(configured))
    return default_event_log_path(dataset_root)
