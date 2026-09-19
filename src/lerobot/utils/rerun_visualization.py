# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Rerun visualization backend.

Live control-loop streaming to the Rerun viewer (:func:`log_rerun_data`). Callers usually select a
backend at runtime through the dispatch in :mod:`lerobot.utils.visualization_utils` rather than
importing from here directly. Requires the ``viz`` extra (``pip install 'lerobot[viz]'``).
"""

import numbers
import os
from collections.abc import Iterable

import numpy as np

from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    log_rerun_data.blueprint = None  # Reset blueprint cache for new session
    _SALIENCY_PATHS.clear()  # saliency paths belong to the old session too
    _SALIENCY_BUDGET_SEGMENTS.clear()
    global _VIEW_MASK_LAST_STATUS
    _VIEW_MASK_LAST_STATUS = None

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _build_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]):
    """Build a Rerun blueprint laying out camera images, observation and action scalars in separate views.

    Camera images, observation and action scalars are arranged in a grid.
    """

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]
    # .copy() is an atomic C-level snapshot under the GIL: the recorder thread may extend
    # _SALIENCY_PATHS (via log_rerun_saliency) while the main thread iterates it here.
    views.extend(
        rrb.Spatial2DView(origin=f"saliency/{cam}", name=f"saliency/{cam}")
        for cam in sorted(_SALIENCY_PATHS.copy())
    )
    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))
    if _SALIENCY_BUDGET_SEGMENTS:
        views.append(
            rrb.TimeSeriesView(
                name="saliency/budget",
                contents=sorted(f"saliency/budget/{seg}" for seg in _SALIENCY_BUDGET_SEGMENTS.copy()),
            )
        )

    return rrb.Blueprint(rrb.Grid(*views))


# Saliency overlay entity paths seen so far; the blueprint is rebuilt (re-sent) whenever
# a new one arrives, so overlay views appear alongside the camera views without manual setup.
_SALIENCY_PATHS: set[str] = set()

# Attention-budget segment names seen so far (camera short names + "language"/"state"); same
# rebuild-on-first-arrival pattern as _SALIENCY_PATHS so the budget time series shows up unprompted.
_SALIENCY_BUDGET_SEGMENTS: set[str] = set()

# Last display state used to annotate camera entities.  The control loop may call
# the status logger every tick, but the text annotation is emitted only when the
# mask/revision/state actually changes.
_VIEW_MASK_LAST_STATUS: tuple | None = None


def _ensure_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]) -> None:
    """Build and send the blueprint once, from the first observation and action data."""
    if getattr(log_rerun_data, "blueprint", None) is not None:
        return

    if not (observation_paths or action_paths or image_paths):
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    log_rerun_data._last_paths = (observation_paths, action_paths, image_paths)
    blueprint = _build_blueprint(observation_paths, action_paths, image_paths)
    log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a single `rr.Scalars` batch under one entity path, so that every
      dimension shares the same view instead of being split across one view per element.
    - Multi-dimensional **action** arrays are flattened and logged as a single `rr.Scalars` batch.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    On the first call, a blueprint is built and sent so observation and action scalars get separate
    time-series views and each image gets its own spatial view.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                observation_paths.add(key)
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    rr.log(key, rr.Scalars(arr.astype(float)))
                    observation_paths.add(key)
                else:
                    if arr.shape[-1] == 1:
                        # At record time, the depth unit is inferred from the frame type.
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)
                    image_paths.add(key)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                action_paths.add(key)
            elif isinstance(v, np.ndarray):
                # Flatten any (incl. higher-dimensional) array into a single batched Scalars
                rr.log(key, rr.Scalars(v.reshape(-1).astype(float)))
                action_paths.add(key)

    _ensure_blueprint(observation_paths, action_paths, image_paths)


def log_rerun_view_mask(
    masked: dict[str, bool],
    *,
    revision: int,
    state: str,
    observation: RobotObservation | None = None,
    display_paths: dict[str, str] | None = None,
) -> None:
    """Publish the View Mask status: per-view 0/1, revision, and application state.

    Display-only.  Called on the control loop while the mask is enabled; a no-op when
    rerun is not the active backend (the caller checks ``display_data``/``display_mode``
    so viewing never becomes a prerequisite for ablation to work).
    """
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    global _VIEW_MASK_LAST_STATUS
    for view, is_masked in masked.items():
        rr.log(f"view_mask/{view.rsplit('.', 1)[-1]}", rr.Scalars(1.0 if is_masked else 0.0))
    rr.log("view_mask/revision", rr.Scalars(float(revision)))
    rr.log("view_mask/state", rr.TextLog(state))

    status = (tuple(sorted(masked.items())), revision, state)
    if status != _VIEW_MASK_LAST_STATUS:
        # Annotate the already logged camera entity.  This is a tiny metadata
        # event; it deliberately does not manufacture a second full-resolution
        # image for every camera/tick.
        for view, is_masked in masked.items():
            log_rerun_camera_masked(
                [view],
                label="MASKED" if is_masked else "VISIBLE",
                observation_keys=None if observation is None else observation.keys(),
                display_paths=display_paths,
            )
        _VIEW_MASK_LAST_STATUS = status


def log_rerun_camera_masked(
    cam_keys: Iterable[str],
    label: str = "MASKED",
    *,
    observation_keys: Iterable[str] | None = None,
    display_paths: dict[str, str] | None = None,
) -> None:
    """Annotate original camera entities without copying their pixels.

    The annotation is a Rerun ``Boxes2D`` component on the existing camera entity,
    so the original image remains the only full-resolution payload.  ``TextLog`` is
    kept as a fallback for older Rerun SDKs that do not expose ``Boxes2D``.
    """
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    keys = set(observation_keys or ())
    for cam_key in cam_keys:
        display_key = (display_paths or {}).get(cam_key, cam_key)
        short_name = display_key.removeprefix("observation.images.").removeprefix("observation.")
        candidates = (
            display_key,
            short_name,
            f"observation.{short_name}",
            f"observation.images.{short_name}",
        )
        path = next((candidate for candidate in candidates if candidate in keys), display_key)
        if path == short_name:
            path = f"observation.{short_name}"
        if hasattr(rr, "Boxes2D"):
            color = [220, 0, 0, 255] if label == "MASKED" else [0, 180, 0, 255]
            rr.log(
                path,
                rr.Boxes2D(
                    mins=[[64, 0]],  # Rerun centres the label on the box; keep it inside the image.
                    sizes=[[1, 1]],
                    colors=[color],
                    labels=[label],
                    show_labels=True,
                ),
                static=True,
            )
        else:
            rr.log(f"view_mask/tile/{short_name}/label", rr.TextLog(label))


def log_rerun_saliency(saliency_frames: dict[str, np.ndarray]) -> None:
    """Log saliency overlay frames (RGB uint8) under ``saliency/<camera>`` entities."""
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    for cam, frame in saliency_frames.items():
        rr.log(f"saliency/{cam}", rr.Image(frame), static=True)

    new_paths = set(saliency_frames) - _SALIENCY_PATHS
    _SALIENCY_PATHS.update(saliency_frames)
    if new_paths and getattr(log_rerun_data, "blueprint", None) is not None:
        # The observation blueprint already ran (before the first saliency frame landed);
        # rebuild it from the stashed path sets so the new overlay views are included.
        obs_paths, act_paths, img_paths = getattr(log_rerun_data, "_last_paths", (set(), set(), set()))
        blueprint = _build_blueprint(obs_paths, act_paths, img_paths)
        log_rerun_data.blueprint = blueprint
        rr.send_blueprint(blueprint)


def log_rerun_saliency_budget(budget: dict[str, float]) -> None:
    """Log attention-budget shares under ``saliency/budget/<segment>`` scalar entities.

    ``budget`` maps segment names (camera short names, "language", "state") to attention
    mass shares in [0, 1]. The blueprint gains a ``saliency/budget`` time-series view on
    the first call (same rebuild-on-arrival pattern as :func:`log_rerun_saliency`).
    """
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    for segment, share in budget.items():
        rr.log(f"saliency/budget/{segment}", rr.Scalars(float(share)))

    new_segments = set(budget) - _SALIENCY_BUDGET_SEGMENTS
    _SALIENCY_BUDGET_SEGMENTS.update(budget)
    if new_segments and getattr(log_rerun_data, "blueprint", None) is not None:
        obs_paths, act_paths, img_paths = getattr(log_rerun_data, "_last_paths", (set(), set(), set()))
        blueprint = _build_blueprint(obs_paths, act_paths, img_paths)
        log_rerun_data.blueprint = blueprint
        rr.send_blueprint(blueprint)
