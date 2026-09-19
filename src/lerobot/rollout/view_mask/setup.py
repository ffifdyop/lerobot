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

"""Build the View Mask runtime once the policy and robot are known.

Kept separate from :mod:`lerobot.rollout.context` so the wiring reads as one call:
resolve names, check hotkey ownership, attach the hotkey listener, open the sidecar.
"""

from __future__ import annotations

import logging

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.utils.keyboard_input import create_latching_key_listener

from .config import ViewMaskConfig, build_view_mask_plan, validate_hotkey_conflicts
from .controller import ViewMaskController
from .events import ViewMaskEventRecorder, resolve_event_log_path
from .runtime import ViewMaskRuntime

logger = logging.getLogger(__name__)

IMAGE_PREFIX = "observation.images."

# Keys already owned by rollout controls that always run, regardless of strategy:
# the interactive prompt reads whole lines from stdin, and Ctrl-C is process-level.
# Strategy- and teleop-owned keys are contributed by the caller.
_BASE_KEY_CLAIMS = {
    "esc": "the running strategy (stop / abort)",
}


def collect_hotkey_claims(strategy_claims: dict[str, str] | None) -> dict[str, str]:
    """Everything other than view ablation that could receive a raw keyboard key.

    The caller supplies strategy-specific claims (DAgger's pause/correction/upload,
    Highlight's save/push) plus any teleoperator-owned keys, so arbitration happens
    in one place with the full picture instead of per-listener registration order.
    """
    claims = dict(_BASE_KEY_CLAIMS)
    claims.update(strategy_claims or {})
    return claims


def build_view_mask_runtime(
    cfg: ViewMaskConfig,
    policy_config: PreTrainedConfig,
    *,
    robot_camera_keys: list[str],
    rename_map: dict[str, str],
    dataset_root: str | None,
    key_claims: dict[str, str] | None = None,
    interactive: bool = False,
) -> ViewMaskRuntime | None:
    """Resolve, validate and wire the View Mask runtime.  ``None`` when disabled.

    Raises:
        ValueError: for unknown/ambiguous views or a hotkey owned by another control.
            Both are startup errors: a silently dropped view or a silently stolen
            key would corrupt the experiment without any visible symptom.
    """
    if not cfg.enabled:
        return None

    policy_views = [
        key for key, feature in policy_config.input_features.items() if feature.type is FeatureType.VISUAL
    ]
    plan = build_view_mask_plan(
        cfg,
        policy_view_keys=policy_views,
        robot_camera_keys=robot_camera_keys,
        rename_map=rename_map,
    )
    validate_hotkey_conflicts(plan, collect_hotkey_claims(key_claims))

    controller = ViewMaskController(plan, timeout_s=cfg.transition_timeout_s)
    snapshot = controller.snapshot()
    recorder = ViewMaskEventRecorder(
        resolve_event_log_path(cfg.event_log_path, dataset_root),
        canonical_views=plan.canonical_views,
        initial_mask=snapshot,
    )

    def dispatch(key: str) -> None:
        canonical = plan.key_to_view.get(key)
        if canonical is None:
            return
        controller.toggle(canonical)

    listener = create_latching_key_listener(
        dispatch,
        controls_help=", ".join(f"{key}={view}" for key, view in sorted(plan.key_to_view.items())),
        # Interactive sessions have a line-oriented stdin command reader.  A raw
        # terminal listener would race it for bytes, so /mask is the safe fallback
        # there; global pynput remains available when the desktop backend works.
        terminal_enabled=not interactive,
    )
    if listener is None:
        logger.warning(
            "View ablation is enabled but no keyboard backend is available: the launch mask "
            "(%s) stays in effect and hotkeys are unavailable.",
            sorted(snapshot.masked_views) or ["(none)"],
        )

    return ViewMaskRuntime(plan, controller, recorder, listener=listener)


def strategy_hotkey_claims(strategy, teleop=None) -> dict[str, str]:
    """Keys a strategy registers for itself, as ``{key: "<feature> (what it does)"}``."""
    claims: dict[str, str] = {}
    save_key = getattr(strategy, "save_key", None)
    if save_key:
        claims[save_key] = "Highlight strategy (save episode)"
    push_key = getattr(strategy, "push_key", None)
    if push_key:
        claims[push_key] = "Highlight strategy (push to hub)"

    keyboard_cfg = getattr(strategy, "keyboard", None)
    if keyboard_cfg is not None:
        if getattr(keyboard_cfg, "pause_resume", None):
            claims[keyboard_cfg.pause_resume] = "DAgger (pause/resume policy)"
        if getattr(keyboard_cfg, "correction", None):
            claims[keyboard_cfg.correction] = "DAgger (start/stop human correction)"
        if getattr(keyboard_cfg, "upload", None):
            claims[keyboard_cfg.upload] = "DAgger (upload dataset)"
    # Keyboard teleoperators install their own global listener and consume every
    # key (the teleop config has no reserved-key declaration).  Treat the complete
    # keyboard namespace as claimed rather than allowing a custom mask key to race
    # the teleoperator silently.
    if teleop is not None and getattr(teleop, "type", "") in {
        "keyboard",
        "keyboard_ee",
        "keyboard_rover",
    }:
        claims["<keyboard>"] = "keyboard teleoperator (all keys)"
    return claims
