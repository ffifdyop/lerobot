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

"""Display-only View Mask telemetry adapters."""

from __future__ import annotations

from lerobot.lerobot_types import RobotObservation
from lerobot.utils.import_utils import require_package


def log_status(
    display_mode: str,
    masked: dict[str, bool],
    *,
    revision: int,
    state: str,
    observation: RobotObservation | None = None,
    display_paths: dict[str, str] | None = None,
) -> None:
    """Publish mask state without making visualization a control-loop dependency."""
    if display_mode == "rerun":
        from lerobot.utils.rerun_visualization import log_rerun_view_mask

        log_rerun_view_mask(
            masked,
            revision=revision,
            state=state,
            observation=observation,
            display_paths=display_paths,
        )
    elif display_mode == "foxglove":
        log_foxglove_status(masked, revision=revision, state=state)
    else:
        raise ValueError(f"Unknown display_mode {display_mode!r}")


def log_foxglove_status(masked: dict[str, bool], *, revision: int, state: str) -> None:
    """Publish the scalar representation supported by Foxglove."""
    require_package("foxglove-sdk", extra="viz", import_name="foxglove")

    from lerobot.utils.foxglove_visualization import _log_foxglove_scalars, log_foxglove_data

    if getattr(log_foxglove_data, "server", None) is None:
        return
    channels = getattr(log_foxglove_data, "channels", {})
    values = {view.rsplit(".", 1)[-1]: 1.0 if is_masked else 0.0 for view, is_masked in masked.items()}
    values["revision"] = float(revision)
    _log_foxglove_scalars("/view_mask", values, channels=channels)
    if state:
        _log_foxglove_scalars("/view_mask_state", {state: 1.0}, channels=channels)
