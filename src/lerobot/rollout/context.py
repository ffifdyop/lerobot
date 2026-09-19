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

"""Rollout context: shared state created once before strategy dispatch.

Grouped into five topical sub-contexts — :class:`RuntimeContext`,
:class:`HardwareContext`, :class:`PolicyContext`, :class:`ProcessorContext`,
and :class:`DatasetContext` — assembled into :class:`RolloutContext`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING

import torch

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.import_utils import _peft_available, require_package

from .configs import BaseStrategyConfig, DAggerStrategyConfig, RolloutConfig
from .inference import (
    InferenceEngine,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .inference.rtc import accepts_noise_injection, supports_rtc_inference
from .robot_wrapper import ThreadSafeRobot

if TYPE_CHECKING:
    from .saliency.manager import SaliencyManager

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel
else:
    PeftConfig = None
    PeftModel = None

logger = logging.getLogger(__name__)


def _wrap_predict_action_chunk_with_torch_compile(
    policy: PreTrainedPolicy,
    *,
    backend: str,
    mode: str,
) -> bool:
    """Install the JIT wrapper and report whether it was configured successfully.

    ``torch.compile`` compiles lazily on the first invocation, so success here
    does not guarantee that backend compilation will succeed during warm-up.
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available in this PyTorch build")
        return False

    try:
        policy.predict_action_chunk = torch.compile(
            policy.predict_action_chunk,
            backend=backend,
            mode=mode,
        )
    except Exception as exc:
        logger.warning("Failed to configure torch.compile: %s", exc)
        return False

    logger.info("torch.compile configured for predict_action_chunk")
    return True


def _validate_trained_rtc_rollout_config(policy_config, inference_config: RTCInferenceConfig) -> None:
    """Fail fast when rollout cannot retain every trained RTC prefix."""
    rtc = inference_config.rtc
    if not rtc.enabled or rtc.mode != "trained":
        return
    if policy_config.type != "pi05":
        raise ValueError(
            "--inference.rtc.mode=trained currently requires a PI05 checkpoint; "
            f"got policy type {policy_config.type!r}."
        )

    training_max_delay = int(getattr(policy_config, "rtc_training_max_delay", 0))
    if training_max_delay <= 0:
        raise ValueError(
            "--inference.rtc.mode=trained requires a checkpoint trained with "
            "--policy.rtc_training_max_delay > 0."
        )
    if rtc.execution_horizon < training_max_delay:
        raise ValueError(
            f"--inference.rtc.execution_horizon ({rtc.execution_horizon}) must be at least the "
            f"checkpoint's rtc_training_max_delay ({training_max_delay})."
        )
    if inference_config.queue_threshold < training_max_delay:
        raise ValueError(
            f"--inference.queue_threshold ({inference_config.queue_threshold}) must be at least the "
            f"checkpoint's rtc_training_max_delay ({training_max_delay})."
        )

    # RTC requires d <= s <= H - d (arXiv 2506.07339): an execution horizon past H - d would
    # commit actions the next chunk can no longer re-plan, so the overlap never closes.
    chunk_size = int(getattr(policy_config, "chunk_size", 0))
    if chunk_size and rtc.execution_horizon > chunk_size - training_max_delay:
        raise ValueError(
            f"--inference.rtc.execution_horizon ({rtc.execution_horizon}) must be at most "
            f"chunk_size - rtc_training_max_delay ({chunk_size} - {training_max_delay} = "
            f"{chunk_size - training_max_delay})."
        )


def _resolve_action_key_order(
    policy_action_names: list[str] | None, dataset_action_names: list[str]
) -> list[str]:
    """Choose action name ordering for mapping policy tensor outputs to robot action dicts."""
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        logger.warning(
            "policy.action_feature_names length (%d) != dataset action dim (%d); using dataset order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(dataset_action_names) != set(policy_action_names):
        logger.warning("policy.action_feature_names keys don't match dataset; using dataset order")
        return dataset_action_names
    return policy_action_names


def _align_state_feature_order(
    observation_features_hw: dict[str, type | tuple], policy_action_names: list[str] | None
) -> dict[str, type | tuple]:
    """Order scalar state features to match the checkpoint's joint order."""
    if not policy_action_names:
        return observation_features_hw

    scalar_names = [
        name for name, feature in observation_features_hw.items() if not isinstance(feature, tuple)
    ]
    if set(scalar_names) != set(policy_action_names) or scalar_names == policy_action_names:
        return observation_features_hw

    reordered = {name: observation_features_hw[name] for name in policy_action_names}
    reordered.update(
        {name: feature for name, feature in observation_features_hw.items() if name not in reordered}
    )
    logger.warning(
        "Robot state order %s differs from checkpoint joint order %s; reordering state",
        scalar_names,
        policy_action_names,
    )
    return reordered


# ---------------------------------------------------------------------------
# Sub-contexts
# ---------------------------------------------------------------------------


@dataclass
class RuntimeContext:
    """Runtime knobs shared with every strategy."""

    cfg: RolloutConfig
    shutdown_event: Event
    # Where the control loop's ``CycleTimer`` sends its cadence summaries; None
    # leaves them on ``logger.info``.  A strategy declaring ``supports_interactive``
    # must forward it to the timer it builds in ``run()``, since a session mutes
    # everything below ERROR.
    cadence_report: Callable[[str], None] | None = None


@dataclass
class HardwareContext:
    """Connected hardware.

    The raw robot is available via ``robot_wrapper.inner`` when needed
    (e.g. for disconnect); strategies should otherwise go through the
    thread-safe wrapper.

    ``initial_position`` stores the robot's joint positions at connect
    time.  Strategies use it to return the robot to a safe pose before
    shutting down.
    """

    robot_wrapper: ThreadSafeRobot
    teleop: Teleoperator | None
    initial_position: dict | None = None


@dataclass
class PolicyContext:
    """Loaded policy and its inference engine."""

    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    inference: InferenceEngine
    saliency: SaliencyManager | None = None
    # View ablation runtime; None when ``--view_mask.enabled=false``.
    view_mask: object | None = None


@dataclass
class ProcessorContext:
    """Robot-side pipelines (run outside the policy)."""

    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]


@dataclass
class DatasetContext:
    """Dataset and feature bookkeeping."""

    dataset: LeRobotDataset | None
    dataset_features: dict = field(default_factory=dict)
    hw_features: dict = field(default_factory=dict)
    ordered_action_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutContext:
    """Bundle of sub-contexts passed to every rollout strategy.

    Built once by :func:`build_rollout_context` before strategy dispatch.
    """

    runtime: RuntimeContext
    hardware: HardwareContext
    policy: PolicyContext
    processors: ProcessorContext
    data: DatasetContext


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _find_rename_insert_index(preprocessor: PolicyProcessorPipeline) -> int:
    """Index just past the last observation-rename step, i.e. where the mask step goes.

    The mask must run *after* the rename (so it targets the canonical keys the policy
    reads) and *before* every normalization and policy-specific image transform (so
    the zeroed image is normalized exactly like any other frame).  When a pipeline has
    no rename step at all the mask goes first, which is the same position for a
    pipeline whose keys are already canonical.
    """
    last_rename = -1
    for index, step in enumerate(preprocessor.steps):
        if isinstance(step, RenameObservationsProcessorStep):
            last_rename = index
    return last_rename + 1


#: Dataset feature names added by view ablation.  Absent entirely when disabled,
#: so an ablation-free dataset schema is bit-for-bit what it was before.
def _view_mask_dataset_features(canonical_views: tuple[str, ...]) -> dict:
    """The two per-frame mask fields, named by canonical view key."""
    return {
        "view_mask": {
            "dtype": "bool",
            "shape": (len(canonical_views),),
            # Names (not just a shape) so a resumed run can prove the vector means
            # the same thing — the same views, in the same slot order.
            "names": list(canonical_views),
        },
        "view_mask_revision": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
    }


def _validate_view_mask_dataset(dataset: LeRobotDataset, canonical_views: tuple[str, ...]) -> None:
    """Refuse to resume a dataset whose mask schema does not match this run.

    Migration is deliberately not attempted: silently appending differently-shaped
    or differently-named mask columns would produce a dataset whose frames cannot be
    compared, which is worse than asking the operator to start a new recording.
    """
    features = dataset.meta.features if hasattr(dataset.meta, "features") else {}
    problems: list[str] = []
    for name, expected in _view_mask_dataset_features(canonical_views).items():
        actual = features.get(name)
        if actual is None:
            problems.append(f"{name!r} is missing")
            continue
        if actual.get("dtype") != expected["dtype"]:
            problems.append(f"{name!r} dtype is {actual.get('dtype')!r}, expected {expected['dtype']!r}")
        elif tuple(actual.get("shape", ())) != tuple(expected["shape"]):
            problems.append(f"{name!r} shape is {actual.get('shape')}, expected {expected['shape']}")
        elif name == "view_mask" and list(actual.get("names") or []) != list(canonical_views):
            problems.append(
                f"{name!r} names are {actual.get('names')}, expected {list(canonical_views)}"
            )
    if problems:
        raise ValueError(
            "Cannot --resume this dataset with view ablation enabled: "
            + "; ".join(problems)
            + ". Start a new dataset (drop --resume) — the mask schema cannot be migrated."
        )


def _inject_view_mask_step(preprocessor: PolicyProcessorPipeline) -> None:
    """Splice the View Mask step into the checkpoint preprocessor, in memory only.

    Deliberately not written back to the checkpoint's processor config: the step is a
    rollout runtime concern, and persisting it would make every other consumer of the
    checkpoint (eval, record) silently mask the same views.
    """
    from .view_mask.processor import ViewMaskProcessorStep

    if any(isinstance(step, ViewMaskProcessorStep) for step in preprocessor.steps):
        return
    steps = list(preprocessor.steps)
    insert_at = _find_rename_insert_index(preprocessor)
    steps.insert(insert_at, ViewMaskProcessorStep())
    preprocessor.steps = steps
    logger.info(
        "View ablation: mask processor injected at preprocessor step %d of %d",
        insert_at,
        len(steps),
    )


def _load_pretrained_policy(policy_config: PreTrainedConfig) -> PreTrainedPolicy:
    """Load policy weights, keeping adapter and base-model revisions independent."""
    pretrained_revision = policy_config.pretrained_revision
    policy_class = get_policy_class(policy_config.type)

    if not policy_config.use_peft:
        return policy_class.from_pretrained(
            policy_config.pretrained_path,
            config=policy_config,
            revision=pretrained_revision,
        )

    require_package("peft", extra="peft")

    peft_path = policy_config.pretrained_path
    peft_config = PeftConfig.from_pretrained(peft_path, revision=pretrained_revision)
    policy = policy_class.from_pretrained(
        pretrained_name_or_path=peft_config.base_model_name_or_path,
        config=policy_config,
        revision=peft_config.revision,
    )
    return PeftModel.from_pretrained(
        policy,
        peft_path,
        config=peft_config,
        revision=pretrained_revision,
    )


def build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext:
    """Wire up policy, processors, hardware, dataset, and inference engine.

    The order is policy-first / hardware-last so a bad ``--policy.path``
    fails fast without touching the robot.
    """
    is_rtc = isinstance(cfg.inference, RTCInferenceConfig)

    # --- 1. Policy (heavy I/O, but no hardware yet) -------------------
    logger.info("Loading policy from '%s'...", cfg.policy.pretrained_path)
    policy_config = cfg.policy

    if is_rtc:
        _validate_trained_rtc_rollout_config(policy_config, cfg.inference)

    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = cfg.use_torch_compile

    if policy_config.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy = _load_pretrained_policy(policy_config)

    if is_rtc:
        if not supports_rtc_inference(policy):
            raise ValueError(
                f"RTC inference is not supported by policy type '{policy_config.type}': "
                "the policy must implement RTC semantics and predict_action_chunk must accept "
                "inference_delay and prev_chunk_left_over. Use '--inference.type=sync' instead."
            )
        policy.config.rtc_config = cfg.inference.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()
    logger.info("Policy loaded: type=%s, device=%s", policy_config.type, cfg.device)

    saliency_manager = None
    if cfg.saliency.enabled:
        # Local import inside the saliency-only branch: pulls the manager (and transitively
        # cv2 via the recorder) in only when saliency is enabled. The saliency package
        # __init__ stays import-light by design, so importing through it costs nothing here.
        from lerobot.rollout.saliency.manager import SaliencyManager, resolve_explainer_device

        # Resolve the explainer device up front and make it explicit on cfg.saliency:
        # from_pretrained loads onto config.device, so without the override the copy's
        # weights would land on the policy's GPU first and only then move, transiently
        # doubling policy-GPU residency. resolve_explainer_device gives an explicit
        # cfg_device the win, so the manager re-resolves to the same device.
        saliency_explainer_device = resolve_explainer_device(cfg.saliency.device, cfg.device or "cpu")
        cfg.saliency.device = saliency_explainer_device

        def _load_explainer_policy():
            # torch.compile and the load device must stay off the explainer's copy:
            # without the device override, from_pretrained would load onto the policy's
            # GPU first (config object is shared), transiently doubling policy-GPU
            # residency before .to(explainer_device) moves it.
            prev_compile = getattr(policy_config, "compile_model", None)
            prev_device = getattr(policy_config, "device", None)
            if prev_compile is not None:
                policy_config.compile_model = False
            if prev_device is not None:
                policy_config.device = saliency_explainer_device
            try:
                return _load_pretrained_policy(policy_config)
            finally:
                if prev_compile is not None:
                    policy_config.compile_model = prev_compile
                if prev_device is not None:
                    policy_config.device = prev_device

        logger.info("Saliency enabled — loading explainer model copy on %s...", saliency_explainer_device)
        saliency_manager = SaliencyManager(
            cfg.saliency,
            policy_config,
            policy_device=cfg.device or "cpu",
            display_mode=cfg.display_mode,
            load_policy=_load_explainer_policy,
        )

    view_mask_runtime = None
    try:
        torch_compile_active = cfg.use_torch_compile
        if cfg.use_torch_compile and policy.type not in ("pi0", "pi05"):
            torch_compile_active = _wrap_predict_action_chunk_with_torch_compile(
                policy,
                backend=cfg.torch_compile_backend,
                mode=cfg.torch_compile_mode,
            )

        if cfg.use_torch_compile and not torch_compile_active:
            # RolloutConfig.__post_init__ reloads the policy configuration, so avoid
            # dataclasses.replace when carrying the effective state downstream.
            cfg = copy(cfg)
            cfg.use_torch_compile = False

        # --- 2. Robot-side processors (user-supplied or defaults) --------
        if (
            teleop_action_processor is None
            or robot_action_processor is None
            or robot_observation_processor is None
        ):
            _t, _r, _o = make_default_processors()
            teleop_action_processor = teleop_action_processor or _t
            robot_action_processor = robot_action_processor or _r
            robot_observation_processor = robot_observation_processor or _o

        # --- 3. Hardware (heaviest side-effect, deferred) -----------------
        logger.info("Connecting robot (%s)...", cfg.robot.type if cfg.robot else "?")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        logger.info("Robot connected: %s", robot.name)

        # Store the initial joint positions so we can return to a safe pose on shutdown.
        initial_obs = robot.get_observation()
        initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
        logger.info("Captured initial robot position (%d keys)", len(initial_position))

        robot_wrapper = ThreadSafeRobot(robot)

        teleop = None
        if cfg.teleop is not None:
            logger.info("Connecting teleoperator (%s)...", cfg.teleop.type if cfg.teleop else "?")
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            logger.info("Teleoperator connected")

        # TODO(Steven): once Teleoperator motor-control methods are standardised
        # (``enable_torque`` / ``disable_torque`` / ``write_goal_positions``), gate
        # the DAgger strategy on their presence here and fail fast with a helpful
        # message instead of relying on the operator to pre-align the leader by
        # hand.  See :func:`DAggerStrategy._apply_transition` for the matching
        # disabled call sites.
        # if isinstance(cfg.strategy, DAggerStrategyConfig) and teleop is not None:
        #     required_teleop_methods = ("enable_torque", "disable_torque", "write_goal_positions")
        #     missing = [m for m in required_teleop_methods if not callable(getattr(teleop, m, None))]
        #     if missing:
        #         teleop.disconnect()
        #         raise ValueError(
        #             f"DAgger strategy requires a teleoperator with motor control methods "
        #             f"{required_teleop_methods}. '{type(teleop).__name__}' is missing: {missing}"
        #         )

        # --- 4. Features + action-key reconciliation ---------------------
        # TODO(Steven):Only ``.pos`` joint features are routed to the policy as state and as the
        # action target; velocity and torque channels (when present) are kept in
        # the raw observation but excluded from the policy-facing tensors.
        all_obs_features = robot.observation_features
        # ``observation_features`` values are either a tuple (camera shape) or the
        # ``float`` type itself used as a sentinel for scalar motor features —
        # see ``dict[str, type | tuple]`` annotation on ``Robot.observation_features``.
        # Keep cameras (tuple) plus both joint-position (.pos) and base-velocity (.vel)
        # scalar state features. LeKiwi's observation.state is 9-dim (6 arm .pos +
        # x/y/theta.vel) and the policy was trained/normalized on all 9; the old .pos-only
        # filter fed a 6-dim state into a 9-dim normalizer → RuntimeError (size 6 vs 9).
        # Pure-arm robots have no .vel state keys, so this is a no-op for them.
        observation_features_hw = {
            k: v
            for k, v in all_obs_features.items()
            if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))
        }
        policy_action_names = getattr(policy_config, "action_feature_names", None)
        observation_features_hw = _align_state_feature_order(
            observation_features_hw,
            list(policy_action_names) if policy_action_names else None,
        )
        # Keep both joint-position (.pos) and base-velocity (.vel) action features so
        # mobile manipulators command the base too (e.g. LeKiwi: 6 arm .pos +
        # x/y/theta.vel = 9-dim action). Pure-arm robots have no .vel keys, so this is
        # a no-op for them. Without the .vel keys the base velocities are silently
        # dropped from dataset_features[ACTION]/ordered_action_keys and the base never moves.
        action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith((".pos", ".vel"))}

        # The action side is always needed: sync inference reads action names from
        # ``dataset_features[ACTION]`` to map policy tensors back to robot actions.
        action_dataset_features = aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=cfg.dataset.video if cfg.dataset else True,
        )
        # Observation-side aggregation is needed because of build_dataset_frame
        observation_dataset_features = aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=cfg.dataset.video if cfg.dataset else True,
        )
        dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
        hw_features = hw_to_dataset_features(observation_features_hw, "observation")
        # The policy-facing action space is what teleop_action_processor produces (identity
        # for joint-space policies, ee keys for EE policies), so order actions by
        # dataset_features[ACTION] names rather than the raw hardware keys. With an identity
        # teleop processor the two coincide, keeping joint-space rollouts unchanged.
        policy_action_space_keys = list(dataset_features[ACTION].get("names") or action_features_hw.keys())
        ordered_action_keys = _resolve_action_key_order(
            list(policy_action_names) if policy_action_names else None,
            policy_action_space_keys,
        )

        # Validate visual features if no rename_map is active
        rename_map = cfg.rename_map
        if not rename_map:
            expected_visuals = {
                k for k, v in policy_config.input_features.items() if v.type == FeatureType.VISUAL
            }
            provided_visuals = {
                f"observation.images.{k}"
                for k, v in robot.observation_features.items()
                if isinstance(v, tuple)
            }
            policy_subset = expected_visuals.issubset(provided_visuals)
            hw_subset = provided_visuals.issubset(expected_visuals)
            if not (policy_subset or hw_subset):
                raise ValueError(
                    f"Visual feature mismatch between policy and robot hardware.\n"
                    f"Policy expects: {expected_visuals}\n"
                    f"Robot provides: {provided_visuals}\n"
                    f"Use --rename_map to map camera names, e.g. "
                    f"""--rename_map='{{"observation.images.top": "observation.images.cam0"}}'"""
                )

        # --- 5. Dataset -------------
        # View ablation's dataset fields must exist before create() or be verified
        # before resume(), so the plan is resolved here rather than after the engine.
        view_mask_plan = None
        if cfg.view_mask.enabled:
            from .view_mask.config import build_view_mask_plan

            view_mask_plan = build_view_mask_plan(
                cfg.view_mask,
                policy_view_keys=[
                    key
                    for key, feature in policy_config.input_features.items()
                    if feature.type is FeatureType.VISUAL
                ],
                robot_camera_keys=[
                    key for key, shape in all_obs_features.items() if isinstance(shape, tuple)
                ],
                rename_map=cfg.rename_map,
            )

        dataset = None
        if cfg.dataset is not None and not isinstance(cfg.strategy, BaseStrategyConfig):
            logger.info("Setting up dataset (repo_id=%s)...", cfg.dataset.repo_id)
            if view_mask_plan is not None:
                dataset_features.update(_view_mask_dataset_features(view_mask_plan.canonical_views))
            if cfg.resume:
                dataset = LeRobotDataset.resume(
                    cfg.dataset.repo_id,
                    root=cfg.dataset.root,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    rgb_encoder=cfg.dataset.rgb_encoder,
                    depth_encoder=cfg.dataset.depth_encoder,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    encoder_threads=cfg.dataset.encoder_threads,
                    image_writer_processes=cfg.dataset.num_image_writer_processes,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                    * len(robot.cameras if hasattr(robot, "cameras") else []),
                )
                if view_mask_plan is not None:
                    _validate_view_mask_dataset(dataset, view_mask_plan.canonical_views)
            else:
                if isinstance(cfg.strategy, DAggerStrategyConfig):
                    dataset_features["intervention"] = {
                        "dtype": "bool",
                        "shape": (1,),
                        "names": None,
                    }

                repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
                if not repo_name.startswith("rollout_"):
                    raise ValueError(
                        "Dataset names for rollout must start with 'rollout_'. "
                        "Use --dataset.repo_id=<user>/rollout_<name> for policy deployment datasets."
                    )
                cfg.dataset.stamp_repo_id()
                target_video_mb = getattr(cfg.strategy, "target_video_file_size_mb", None)
                dataset = LeRobotDataset.create(
                    cfg.dataset.repo_id,
                    cfg.dataset.fps,
                    root=cfg.dataset.root,
                    robot_type=robot.name,
                    features=dataset_features,
                    use_videos=cfg.dataset.video,
                    image_writer_processes=cfg.dataset.num_image_writer_processes,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                    * len(robot.cameras if hasattr(robot, "cameras") else []),
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    rgb_encoder=cfg.dataset.rgb_encoder,
                    depth_encoder=cfg.dataset.depth_encoder,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    encoder_threads=cfg.dataset.encoder_threads,
                    video_files_size_in_mb=target_video_mb,
                )

        if dataset is not None:
            logger.info("Dataset ready: %s (%d existing episodes)", dataset.repo_id, dataset.num_episodes)

        # --- 6. Policy pre/post processors (needs dataset stats if any) ---
        dataset_stats = None
        if dataset is not None:
            dataset_stats = rename_stats(
                dataset.meta.stats,
                cfg.rename_map,
            )

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=cfg.policy.pretrained_path,
            pretrained_revision=policy_config.pretrained_revision,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        relative_action_step = next(
            (
                step
                for step in getattr(preprocessor, "steps", ())
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        if isinstance(cfg.inference, SyncInferenceConfig) and relative_action_step is not None:
            raise NotImplementedError(
                "SyncInferenceEngine does not support policies with relative actions for now."
                "Use --inference.type=rtc or remove relative action processor steps from the policy pipeline."
            )

        # --- 6b. View ablation (needs policy features + robot cameras) -----
        # Built after the pre/post processors exist so the mask step can be spliced
        # into the checkpoint preprocessor, and before the engine so the engine can
        # register itself as the transition-flush sink.
        if view_mask_plan is not None:
            from .view_mask.setup import build_view_mask_runtime, strategy_hotkey_claims

            view_mask_runtime = build_view_mask_runtime(
                cfg.view_mask,
                policy_config,
                robot_camera_keys=[
                    key for key, shape in all_obs_features.items() if isinstance(shape, tuple)
                ],
                rename_map=cfg.rename_map,
                dataset_root=str(dataset.root) if dataset is not None else None,
                key_claims=strategy_hotkey_claims(cfg.strategy, cfg.teleop),
                interactive=cfg.interactive,
            )
            _inject_view_mask_step(preprocessor)

        # --- 7. Inference strategy (needs policy + pre/post + hardware) --
        logger.info(
            "Creating inference engine (type=%s)...",
            cfg.inference.type if hasattr(cfg.inference, "type") else "sync",
        )
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        inference_strategy = create_inference_engine(
            cfg.inference,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            hw_features=hw_features,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task_str,
            fps=cfg.fps,
            device=cfg.device,
            use_torch_compile=torch_compile_active,
            compile_warmup_inferences=cfg.compile_warmup_inferences,
            shutdown_event=shutdown_event,
        )
        if view_mask_runtime is not None:
            inference_strategy.attach_view_mask(view_mask_runtime)

        if saliency_manager is not None:
            # Validation guarantees cfg.inference is RTC when saliency is enabled. Probe the
            # noise kwarg too: supports_rtc_inference only binds inference_delay /
            # prev_chunk_left_over, so an RTC-capable policy without a noise parameter must
            # fail here at build time, not mid-loop inside the RTC thread.
            if not accepts_noise_injection(policy):
                raise ValueError(
                    f"Saliency requires {policy_config.type}.predict_action_chunk to accept a "
                    "'noise' parameter (the RTC engine injects the flow-matching noise so the "
                    "explainer can replay the executed chunk exactly). Disable saliency "
                    "(--saliency.enabled=false) or use a noise-injectable policy."
                )
            inference_strategy.saliency_sink = saliency_manager
            logger.info("Saliency snapshots enabled: RTC engine will inject noise and submit chunk snapshots")

        # --- 8. Assemble ---------------------------------------------------
        logger.info("Rollout context assembled successfully")
        return RolloutContext(
            runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
            hardware=HardwareContext(
                robot_wrapper=robot_wrapper, teleop=teleop, initial_position=initial_position
            ),
            policy=PolicyContext(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                inference=inference_strategy,
                saliency=saliency_manager,
                view_mask=view_mask_runtime,
            ),
            processors=ProcessorContext(
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            ),
            data=DatasetContext(
                dataset=dataset,
                dataset_features=dataset_features,
                hw_features=hw_features,
                ordered_action_keys=ordered_action_keys,
            ),
        )
    except Exception:
        # The recorder thread and its mp4 writers must not outlive a failed build
        # (half-open videos, and the recorder thread keeps the explainer copy alive).
        if saliency_manager is not None:
            logger.warning("Rollout context build failed; stopping saliency recorder")
            saliency_manager.stop()
        if view_mask_runtime is not None:
            logger.warning("Rollout context build failed; stopping View Mask runtime")
            view_mask_runtime.stop()
        raise
