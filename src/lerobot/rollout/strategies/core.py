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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import contextlib
import logging
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

from ..inference import InferenceEngine
from ..view_mask.telemetry import log_status as log_view_mask_status

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.

    Lifecycle: ``setup()`` once, then ``run()``, then ``teardown()`` once.
    A strategy whose config declares ``supports_interactive = True`` is also
    driven by ``--interactive=true``, which calls ``run()`` once per
    start/stop segment.  Such a strategy must keep ``run()`` restartable:

    - never finalize the dataset in ``run()`` — that belongs in ``teardown()``;
      at most save a partial tail episode when a segment ends;
    - keep state that must survive a segment on the instance, not in ``run()``
      locals (the ``CycleTimer`` is deliberately the other way round, see ``run()``);
    - never bind keyboard/terminal listeners — stdin belongs to the command prompt;
    - call ``engine.pump_query(obs_processed)`` once at the end of every tick, see
      ``run()``.

    One-shot strategies (``supports_interactive = False``, the default) are
    free to finalize on ``run()`` exit, e.g. via ``VideoEncodingManager``.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None
        # View ablation is inert unless _attach_view_mask finds a runtime on the
        # context, so these stay None on an ordinary rollout.
        self._view_mask = None
        self._view_mask_dataset = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self.reset_control_state()
        self._engine.start()
        self._warmup_flushed = False
        logger.info("Inference engine started")

    def reset_control_state(self) -> None:
        """Clear episode-scoped control state so a paused session can restart cleanly.

        Resets the inference engine (policy hidden state, action queues), the action
        interpolator and the cached processed observation; pacing state is untouched.
        ``RolloutController`` calls it on its serve thread before each run segment.
        Only call while the control loop is not running: these resets are not synchronized
        against a live loop.  A caller that resets control state while a loop runs — or a
        strategy that hoists its timer onto the instance — must also call ``timer.restart()``.
        """
        if self._engine is not None:
            self._engine.reset()
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None

    # ------------------------------------------------------------------
    # View ablation
    # ------------------------------------------------------------------

    def _attach_view_mask(self, ctx: RolloutContext) -> None:
        """Bind the session's View Mask runtime and register the strategy-owned flush.

        The engine owns the policy/queue half of the Inference Flush; the action
        interpolator belongs to the strategy, so the strategy registers that half
        here.  A no-op when ``--view_mask.enabled=false``.
        """
        runtime = getattr(ctx.policy, "view_mask", None)
        if runtime is None:
            return
        self._view_mask = runtime
        runtime.controller.set_sink(self._engine)
        runtime.add_flush_hook(self._flush_view_mask_actions)

    def _flush_view_mask_actions(self, transition) -> None:
        """Discard interpolated steps that were derived from the previous mask.

        The robot keeps holding the runtime's ``hold_action`` — the last action it
        actually received — until a fresh one lands, rather than coasting along a
        trajectory planned for a scene that no longer applies.
        """
        if self._interpolator is None:
            return
        runtime = getattr(self._engine, "view_mask", None)
        if runtime is not None:
            runtime.hold_action = self._interpolator.last_sent()
        self._interpolator.flush()
        # The cached processed observation belongs to the old mask's tick; drop it so
        # the next policy tick re-processes the observation under the new mask.
        self._cached_obs_processed = None

    def _view_mask_tick(
        self,
        ctx: RolloutContext,
        *,
        runnable: bool,
        phase: str | None = None,
    ) -> bool:
        """One View Mask tick: charge the timeout, commit pending changes.

        Returns ``True`` when the caller must not trust the policy this tick — either
        because a transition just committed (``APPLYING``, holding the last action) or
        because the mask is latched ``FAILED`` (which the caller must surface).  A
        no-op returning ``False`` when view ablation is disabled.
        """
        runtime = self._view_mask
        if runtime is None:
            return False
        if self._engine is not None and self._engine.failed and not runtime.failed:
            # An inference failure while a transition is pending means no fresh action
            # is ever coming: latch FAILED rather than waiting out the whole timeout.
            runtime.mark_failed("policy inference failed while a view mask transition was pending")
        transition = runtime.poll(runnable=runnable, phase=phase)
        if transition is None:
            return runtime.blocked
        self._record_view_mask_transition(runtime, transition)
        return True

    def _view_mask_frame_fields(self, *, human: bool) -> dict | None:
        """Dataset mask fields for a frame about to be recorded, or ``None`` to skip it.

        A Policy frame is labelled from the *dispatched action's* provenance, never
        from the controller's live value: under RTC the action being recorded was
        generated up to a chunk ago, and the mask may already have moved on.  A Human
        frame has no action of its own, so it carries the live snapshot — the mask on
        the observation the shadow policy was conditioned on.

        Returns ``None`` during APPLYING/FAILED: those ticks only hold the last
        action, and recording them would teach the policy that a held action was the
        right response to the current observation.
        """
        runtime = self._view_mask
        if runtime is None:
            return None
        if runtime.blocked:
            return None
        snapshot = runtime.current if human else runtime.dispatched
        return runtime.frame_fields(snapshot)

    def _human_view_mask_fields(self) -> dict | None:
        """Mask fields for a Human frame, or ``None`` when view ablation is off.

        Human control may continue while the policy is blocked, but blocked ticks
        are not training samples.  The mask state is still available to telemetry
        and the sidecar; it is only omitted from the dataset frame.
        """
        runtime = self._view_mask
        if runtime is None or runtime.blocked:
            return None
        return runtime.frame_fields(runtime.current)

    def _view_mask_failed_should_stop(self, _ctx: RolloutContext) -> bool:
        """Whether the current segment should end after View Mask FAILED.

        Ending the segment in both modes lets the controller surface one failure event
        and return to its idle command loop.  Interactive recovery still works through
        ``/reset``; it no longer requires a failed strategy loop to spin forever.
        """
        return self._view_mask is not None and self._view_mask.failed

    def _note_recorded_frame(self, dataset) -> None:
        """Keep the sidecar's alignment hint pointing at the last frame actually recorded."""
        if self._view_mask is None:
            return
        self._view_mask.note_frame(
            episode_index=getattr(dataset, "num_episodes", None),
            frame_index=getattr(dataset, "num_frames", None),
        )

    def _record_view_mask_transition(self, runtime, transition) -> None:
        """Alignment hint: where the event landed relative to the dataset timeline.

        The counters read here are the ones from the last recorded frame, because the
        transition has no frame of its own — it happened *between* frames, and the
        sidecar's job is to say exactly that rather than to invent a timestamp.
        """
        dataset = self._view_mask_dataset
        if dataset is None:
            return
        self._note_recorded_frame(dataset)

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, timer: CycleTimer) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  Warmup ticks are paced through *timer* so the loop cadence
        stays anchored.  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            timer.wait()
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            timer.restart()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self.return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> bool:
        """Smoothly interpolate the robot back to its initial position.

        Returns ``True`` when the interpolation completed, ``False`` when it failed
        partway — the robot is then at an arbitrary pose, so callers must not report
        a completed reset on ``False``.
        """
        robot = hw.robot_wrapper
        target = hw.initial_position
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)
            return False
        return True

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to the visualization backend if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_visualization_data(
            cfg.display_mode,
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    def _log_view_mask(self, runtime_ctx: RuntimeContext) -> None:
        """Publish the View Mask status to the display backend, when one is active.

        Deliberately gated on ``display_data``: the display is a monitor, never a
        control surface, so turning it off must leave ablation fully functional.  The
        terminal log remains the authoritative record either way.
        """
        runtime = self._view_mask
        if runtime is None or not runtime_ctx.cfg.display_data:
            return
        try:
            log_view_mask_status(
                runtime_ctx.cfg.display_mode,
                observation=self._cached_obs_processed,
                **runtime.telemetry(),
            )
        except Exception:  # a display failure must never take the control loop down
            logger.exception("Failed to publish view mask status to the display backend")

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires.

        Implementations must call ``engine.resume()`` before entering their loop
        (async backends start paused, and the interactive controller pauses again at
        the end of every segment), and ``engine.pump_query(obs_processed)`` at the end
        of every tick — the text-query channel only advances through it, and a
        multi-second generation must not sit inside the action path.

        Each ``run()`` call builds its own ``CycleTimer`` and reports it through
        ``timer.log_run_summary()`` from its ``finally``: a fresh timer's start-up
        exemption is what absorbs the interpolator that ``reset_control_state()``
        re-primes at every ``/start``, and each segment gets its own cadence report.
        """

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: finalize dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    timer: CycleTimer | None = None,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    When *timer* is given, the engine pull and the robot send are timed as the
    ``infer`` and ``send`` steps of its cadence summary, and a tick with no action
    to send is counted there.  Note that on async backends ``infer`` is only a
    queue pull — inference runs off-thread, so its latency surfaces as starved
    ticks rather than as loop-body time.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    # ``nullcontext`` accepts (and ignores) the section name, so it stands in for
    # ``timer.section`` verbatim when no timer was passed.
    section = timer.section if timer is not None else contextlib.nullcontext

    if interpolator.needs_new_action():
        with section("infer"):
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        if timer is not None:
            timer.note_starved_tick()
        # A View Mask transition cleared the interpolated tail: hold the last action
        # the robot actually received until a fresh one is available.
        return hold_last_action(ctx, obs_raw, section)

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    with section("send"):
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict


def hold_last_action(ctx: RolloutContext, obs_raw: dict, section) -> dict | None:
    """Send a safe hold command while a mask transition has no fresh action.

    Only View Ablation uses this: without a transition the engine being starved is an
    ordinary condition that the strategies treat as "send nothing".  With one, the
    alternative to holding is leaving the robot uncommanded for the whole APPLYING
    window.  Position actions can be repeated safely; velocity and delta actions
    must be zeroed instead of replaying a stale command.  Unknown action semantics
    are left untouched (no command is sent) rather than guessed.

    Returns ``None`` — meaning "send nothing" — whenever view ablation is off or the
    mask is not blocking, so this is a no-op for every ordinary rollout.
    """
    runtime = getattr(ctx.policy, "view_mask", None)
    if runtime is None or not runtime.blocked or runtime.hold_action is None:
        return None
    hold = runtime.hold_action
    action_dict = {}
    for i, key in enumerate(ctx.data.ordered_action_keys):
        if key.endswith((".vel", ".velocity", ".delta")):
            action_dict[key] = 0.0
        elif key.endswith((".pos", ".position")):
            action_dict[key] = hold[i].item()
        else:
            logger.warning(
                "Cannot safely hold action key %r during View Mask transition; skipping hold command",
                key,
            )
            return None
    with section("send"):
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
