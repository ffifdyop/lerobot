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

"""View ablation configuration, startup validation, and name resolution.

Two layers:

* :func:`validate_view_mask` — flag-combination validation that runs in
  ``RolloutConfig.__post_init__``, before any hardware or policy I/O, so a
  half-configured ``--view_mask.*`` group is rejected without touching the robot.
* :func:`build_view_mask_plan` — resolution of operator-supplied view names
  (raw robot camera keys *or* canonical policy view keys) into the canonical
  view keys the policy actually consumes, run once the policy and robot are
  loaded so the real feature sets are available.

Both layers are pure stdlib: importing this module must not pull in torch, so
the config dataclass stays cheap to construct on every rollout start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Hardware camera keys are reported without the ``observation.images.`` prefix
# (``Robot.observation_features`` uses the short name), so operators naturally
# type either form.  Both are accepted; neither is fuzzy-matched.
IMAGE_PREFIX = "observation.images."

DEFAULT_TRANSITION_TIMEOUT_S = 10.0


@dataclass
class ViewMaskConfig:
    """View ablation settings (``--view_mask.*`` flags on ``lerobot-rollout``).

    Disabled by default and inert when disabled: no hotkeys are registered, no
    processor step is injected and no dataset field is added, so an ordinary
    rollout behaves exactly as before.
    """

    # Master switch. Every other field is a startup error while this is False.
    enabled: bool = False
    # Canonical or raw camera keys to start masked (revision 0).  Empty = no view masked.
    initial_views: list[str] = field(default_factory=list)
    # Per-view toggle keys: ``{"<view>": "<key name>"}``.  Empty = bind F1, F2, F3… to
    # the canonical views in the policy's declaration order.
    hotkeys: dict[str, str] = field(default_factory=dict)
    # Seconds of *runnable* policy inference allowed before a transition latches FAILED.
    # Time spent not inferring (idle, Human, Paused) does not count against it.
    transition_timeout_s: float = DEFAULT_TRANSITION_TIMEOUT_S
    # JSONL event sidecar. None = <dataset.root>/meta/view_mask_events.jsonl when a dataset
    # exists, else outputs/rollout/<timestamp>/view_mask_events.jsonl.
    event_log_path: str | None = None

    @property
    def is_default(self) -> bool:
        """True when no sub-config field was changed from its default.

        ``rollout.*`` cannot tell "unset" from "set to the default value", so an
        explicit ``--view_mask.enabled=false --view_mask.transition_timeout_s=10.0``
        is accepted: the value is indistinguishable from the default, and the
        effective configuration is identical.
        """
        return (
            not self.initial_views
            and not self.hotkeys
            and self.transition_timeout_s == DEFAULT_TRANSITION_TIMEOUT_S
            and self.event_log_path is None
        )


def validate_view_mask(view_mask: ViewMaskConfig) -> None:
    """Fail fast on flag combinations that cannot work. No-op when enabled or fully default.

    Called from ``RolloutConfig.__post_init__``.  Name resolution is deliberately
    *not* done here — the policy's visual features are only known after
    ``--policy.path`` is loaded, so that half happens in
    :func:`build_view_mask_plan`.
    """
    if view_mask.enabled:
        if view_mask.transition_timeout_s <= 0:
            raise ValueError(
                f"--view_mask.transition_timeout_s must be > 0, got {view_mask.transition_timeout_s}"
            )
        return
    if not view_mask.is_default:
        raise ValueError(
            "View ablation sub-options were set while --view_mask.enabled=false "
            f"(initial_views={view_mask.initial_views!r}, hotkeys={view_mask.hotkeys!r}, "
            f"transition_timeout_s={view_mask.transition_timeout_s!r}, "
            f"event_log_path={view_mask.event_log_path!r}). Enable the capability explicitly with "
            "--view_mask.enabled=true, or remove these flags."
        )


@dataclass(frozen=True)
class ViewMaskPlan:
    """Resolved, immutable view-ablation vocabulary for one rollout session.

    ``canonical_views`` preserves the policy's own declaration order — it is the
    slot identity of the model's visual inputs, so it must never be re-sorted.
    """

    canonical_views: tuple[str, ...]
    # Canonical key -> default position label, e.g. ("observation.images.wrist", "F3").
    key_to_view: dict[str, str]
    view_to_key: dict[str, str]
    initial_views: frozenset[str]
    # Canonical policy key -> raw robot camera key, used only to find the existing
    # display entity.  Inference and persistence continue to use canonical keys.
    canonical_to_raw: dict[str, str] = field(default_factory=dict)

    def default_key_for(self, canonical_view: str) -> str:
        """The hotkey bound to ``canonical_view`` (F1, F2, F3… by declaration order)."""
        index = self.canonical_views.index(canonical_view)
        return f"f{index + 1}"


def _canonical_candidates(
    name: str,
    canonical_to_index: dict[str, int],
    raw_to_canonical: dict[str, str],
) -> set[str]:
    """Every canonical view an operator-supplied ``name`` could denote (exact matches only)."""
    candidates: set[str] = set()
    if name in canonical_to_index:
        candidates.add(name)
    if name in raw_to_canonical:
        candidates.add(raw_to_canonical[name])
    prefixed = name if name.startswith(IMAGE_PREFIX) else f"{IMAGE_PREFIX}{name}"
    if prefixed in raw_to_canonical:
        candidates.add(raw_to_canonical[prefixed])
    if prefixed in canonical_to_index:
        candidates.add(prefixed)
    return candidates


def _describe_available(canonical_views: tuple[str, ...]) -> str:
    return ", ".join(repr(k) for k in canonical_views) or "(none)"


def build_view_mask_plan(
    view_mask: ViewMaskConfig,
    *,
    policy_view_keys: list[str],
    robot_camera_keys: list[str],
    rename_map: dict[str, str] | None = None,
) -> ViewMaskPlan:
    """Resolve the config's view names into canonical policy view keys.

    Args:
        view_mask: The (already :func:`validate_view_mask`-checked) config.
        policy_view_keys: Canonical view keys in the policy's declaration order.
        robot_camera_keys: Camera keys as the robot reports them (short hardware
            names), e.g. ``["top", "left_wrist"]``.
        rename_map: The rollout ``--rename_map``, mapping dataset observation keys
            to policy keys.

    Returns:
        The resolved plan.

    Raises:
        ValueError: If any configured name is unknown, non-visual, or ambiguous, or
            if the resolved hotkey set collides.  The whole configuration is
            rejected — a partially applied mask would silently ablate the wrong view.
    """
    rename_map = rename_map or {}
    canonical_views = tuple(dict.fromkeys(policy_view_keys))
    if not canonical_views:
        raise ValueError(
            "View ablation requires the policy to declare at least one visual input feature; "
            "no visual features were found in the policy config."
        )
    canonical_to_index = {key: i for i, key in enumerate(canonical_views)}

    # Dataset key -> canonical key: cameras arrive as ``observation.images.<short>`` in the
    # dataset frame and the preprocessor's rename step maps them onto the policy vocabulary.
    raw_to_canonical: dict[str, str] = {}
    for short in robot_camera_keys:
        dataset_key = short if short.startswith(IMAGE_PREFIX) else f"{IMAGE_PREFIX}{short}"
        raw_to_canonical[dataset_key] = rename_map.get(dataset_key, dataset_key)
    # A name may also be given as the bare hardware key.
    for short in robot_camera_keys:
        if not short.startswith(IMAGE_PREFIX):
            dataset_key = f"{IMAGE_PREFIX}{short}"
            raw_to_canonical.setdefault(short, raw_to_canonical.get(dataset_key, dataset_key))
    # Keys that already match canonically (no rename) resolve to themselves.
    for key in canonical_views:
        raw_to_canonical.setdefault(key, key)

    def resolve(name: str, *, purpose: str) -> str:
        candidates = _canonical_candidates(name, canonical_to_index, raw_to_canonical)
        if not candidates:
            raise ValueError(
                f"Unknown view {name!r} in {purpose}. Available canonical views: "
                f"{_describe_available(canonical_views)}."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous view {name!r} in {purpose}: it matches "
                f"{', '.join(repr(k) for k in sorted(candidates))}. Use the canonical key."
            )
        return next(iter(candidates))

    initial_views: set[str] = set()
    for name in view_mask.initial_views:
        initial_views.add(resolve(name, purpose="--view_mask.initial_views"))

    # Hotkeys: default F1, F2, F3… in the policy's declaration order, overridable per view.
    view_to_key: dict[str, str] = {}
    for canonical in canonical_views:
        index = canonical_to_index[canonical]
        view_to_key[canonical] = f"f{index + 1}"
    for name, key in view_mask.hotkeys.items():
        canonical = resolve(name, purpose="--view_mask.hotkeys")
        view_to_key[canonical] = key

    key_to_view: dict[str, str] = {}
    for canonical, key in view_to_key.items():
        if key in key_to_view:
            raise ValueError(
                f"View ablation hotkey {key!r} is bound to two views "
                f"({key_to_view[key]!r} and {canonical!r}). Each view needs its own key."
            )
        key_to_view[key] = canonical

    canonical_to_raw: dict[str, str] = {}
    for short in robot_camera_keys:
        dataset_key = short if short.startswith(IMAGE_PREFIX) else f"{IMAGE_PREFIX}{short}"
        canonical = raw_to_canonical.get(short, raw_to_canonical.get(dataset_key, dataset_key))
        canonical_to_raw.setdefault(canonical, short)
    for canonical in canonical_views:
        canonical_to_raw.setdefault(canonical, canonical)

    plan = ViewMaskPlan(
        canonical_views=canonical_views,
        key_to_view=key_to_view,
        view_to_key=view_to_key,
        initial_views=frozenset(initial_views),
        canonical_to_raw=canonical_to_raw,
    )
    logger.info(
        "View ablation resolved: canonical views %s, launch mask %s, hotkeys %s",
        list(canonical_views),
        sorted(plan.initial_views) or ["(none)"],
        {plan.default_key_for(v): v for v in canonical_views},
    )
    return plan


def validate_hotkey_conflicts(plan: ViewMaskPlan, claims: dict[str, str]) -> None:
    """Reject hotkeys that collide with another rollout control.

    Args:
        plan: The resolved view-mask plan.
        claims: ``{key name: "<feature> (<what it does>)"}`` for every *other*
            control listening on the same session, gathered by the caller from the
            active strategy and teleoperator.

    Raises:
        ValueError: Listing both sides of every conflict.  Registration order must
            never decide who wins a key.
    """
    all_keys_claim = claims.get("<all>") or claims.get("<keyboard>")
    conflicts = sorted(
        plan.key_to_view
        if all_keys_claim
        else (key for key in plan.key_to_view if key in claims)
    )
    if not conflicts:
        return
    details = "; ".join(
        f"{key!r} = View ablation (toggle {plan.key_to_view[key]}) vs "
        f"{all_keys_claim or claims[key]}"
        for key in conflicts
    )
    raise ValueError(
        f"View ablation hotkey conflict: {details}. "
        "Remap with --view_mask.hotkeys, or change the other control's key."
    )
