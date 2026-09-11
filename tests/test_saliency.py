# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for rollout-time saliency (explainer, recorder, config plumbing)."""

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# Hoisted rather than re-imported inside each test: the module-scoped `importorskip`
# above already guards everything below it.
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# ---------------------------------------------------------------------------
# SaliencyConfig validation
# ---------------------------------------------------------------------------


def test_validate_saliency_requires_rtc_engine():
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    with pytest.raises(ValueError, match="inference.type=rtc"):
        validate_saliency(SaliencyConfig(enabled=True), SyncInferenceConfig(), "rerun", True)


def test_validate_saliency_rejects_unknown_method():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    cfg = SaliencyConfig(enabled=True, method="integrated")
    with pytest.raises(ValueError, match="grad_x_input"):
        validate_saliency(cfg, RTCInferenceConfig(), "rerun", True)


def test_validate_saliency_rejects_unknown_norm_scope():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    assert SaliencyConfig().norm_scope == "run"  # default: one online scale for the whole run
    cfg = SaliencyConfig(enabled=True, norm_scope="moment")
    with pytest.raises(ValueError, match="norm_scope"):
        validate_saliency(cfg, RTCInferenceConfig(), "rerun", True)


def test_validate_saliency_requires_rerun_for_display():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    with pytest.raises(ValueError, match="rerun"):
        validate_saliency(SaliencyConfig(enabled=True), RTCInferenceConfig(), "foxglove", True)


def test_validate_saliency_requires_display_data_for_display():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    # display_data=False but saliency.display=True → rerun never initialized, silent no-op
    with pytest.raises(ValueError, match="--display_data=true"):
        validate_saliency(SaliencyConfig(enabled=True), RTCInferenceConfig(), "rerun", False)


def test_validate_saliency_warns_on_torch_compile(caplog):
    import logging as _logging

    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    with caplog.at_level(_logging.WARNING):
        validate_saliency(
            SaliencyConfig(enabled=True), RTCInferenceConfig(), "rerun", True, use_torch_compile=True
        )
    assert any("torch.compile" in r.message for r in caplog.records)


def test_validate_saliency_passes_when_disabled():
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    validate_saliency(SaliencyConfig(enabled=False), SyncInferenceConfig(), "foxglove", False)


def test_rollout_config_wires_saliency_validation():
    from lerobot.rollout import RolloutConfig
    from lerobot.rollout.saliency.config import SaliencyConfig
    from tests.mocks.mock_robot import MockRobotConfig

    # default inference is sync → enabled saliency must fail fast at config construction
    with pytest.raises(ValueError, match="inference.type=rtc"):
        RolloutConfig(robot=MockRobotConfig(), saliency=SaliencyConfig(enabled=True))


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def test_normalize_map_min_max():
    from lerobot.rollout.saliency.overlay import normalize_map

    m = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    n = normalize_map(m)
    assert n.min() == 0.0 and n.max() == 1.0


def test_render_overlay_shape_and_dtype():
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.full((100, 180, 3), 128, dtype=np.uint8)
    sal = np.random.rand(100, 180).astype(np.float32)
    out = render_overlay(img, sal)
    assert out.shape == (100, 180, 3)
    assert out.dtype == np.uint8


def test_render_overlay_resizes_mismatched_map():
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.full((50, 60, 3), 200, dtype=np.uint8)
    sal = np.zeros((25, 30), dtype=np.float32)
    out = render_overlay(img, sal, blur_sigma=0.0)
    assert out.shape == (50, 60, 3)


def test_render_overlay_blur_flattens_singleton_map():
    """A uniform map must stay uniform after blur+normalize (constant map → colormap index 0, dark blue)."""
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    sal = np.full((32, 32), 7.0, dtype=np.float32)
    out = render_overlay(img, sal)
    # every pixel blended identically → rows identical
    assert (out == out[0]).all()


def test_render_overlay_alpha_zero_is_identity():
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    sal = np.random.rand(20, 30).astype(np.float32)
    out = render_overlay(img, sal, alpha=0.0, blur_sigma=0.0)
    assert (out == img).all()


def test_tensor_to_uint8_bgr_handles_batch_and_rgb():
    from lerobot.rollout.saliency.overlay import tensor_to_uint8_bgr

    t = torch.zeros(1, 3, 8, 6)
    t[:, 0] = 1.0  # RGB red
    arr = tensor_to_uint8_bgr(t)
    assert arr.shape == (8, 6, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0, 2] == 255  # red lands at BGR index 2


def test_tensor_to_uint8_bgr_takes_latest_of_queued_frames():
    from lerobot.rollout.saliency.overlay import tensor_to_uint8_bgr

    t = torch.zeros(1, 2, 3, 8, 6)  # (B, T, C, H, W): a policy with queued observations
    t[:, 1] = 1.0
    arr = tensor_to_uint8_bgr(t)
    assert arr.shape == (8, 6, 3)
    assert arr.dtype == np.uint8
    assert arr.min() == 255  # the latest (all-ones) frame wins


def test_shared_norm_range_spans_all_maps():
    from lerobot.rollout.saliency.overlay import shared_norm_range

    a = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    b = np.array([[0.0, 0.01]], dtype=np.float32)
    assert shared_norm_range([a, b]) == (0.0, 5.0)
    assert shared_norm_range(iter([a, b])) == (0.0, 5.0)  # must not iterate the input twice


def test_render_overlay_shared_range_keeps_weak_map_dark():
    """Two co-timed cameras, one contributing 1000× less than the other: with a shared
    norm range the weak map must stay cold (turbo low end is blue-purple, B > R), while a
    private per-camera min-max would stretch it to the same reds as the strong map."""
    from lerobot.rollout.saliency.overlay import render_overlay, shared_norm_range

    img = np.full((20, 30, 3), 100, dtype=np.uint8)
    strong = np.linspace(0, 1, 600, dtype=np.float32).reshape(20, 30)
    weak = strong * 1e-3

    norm_range = shared_norm_range([strong, weak])
    hot = render_overlay(img, strong, alpha=0.5, blur_sigma=0.0, norm_range=norm_range)
    cold = render_overlay(img, weak, alpha=0.5, blur_sigma=0.0, norm_range=norm_range)

    hot_diff = hot[..., 2].astype(int) - hot[..., 0].astype(int)  # R - B
    cold_diff = cold[..., 2].astype(int) - cold[..., 0].astype(int)
    assert hot_diff.max() > 30  # strong map reaches the red end of the colormap
    assert cold_diff.max() < 0  # weak map stays blue everywhere

    # Regression guard: a private per-camera min-max WOULD paint the weak map red.
    private = render_overlay(img, weak, alpha=0.5, blur_sigma=0.0)
    private_diff = private[..., 2].astype(int) - private[..., 0].astype(int)
    assert private_diff.max() > 30


# ---------------------------------------------------------------------------
# Explainer registry + SmolVLA grad-replay adapter (FakeSmolVLAPolicy stand-in)
# ---------------------------------------------------------------------------

CAM_KEY = "observation.images.cam1"
CAM_KEY_2 = "observation.images.cam2"


class _FakeFlowMatching(torch.nn.Module):
    """Tiny differentiable stand-in for VLAFlowMatching.sample_actions."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.head = torch.nn.Linear(8, action_dim)
        self.last_kwargs: dict | None = None

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        self.last_kwargs = {"noise": noise, **kwargs}
        x = self.conv(images[0]).mean(dim=(2, 3))
        first = self.head(x)  # (B, action_dim)
        return first.unsqueeze(1).repeat(1, 10, 1)  # (B, chunk, action_dim)


class _FakeSmolVLAPolicy(torch.nn.Module):
    """Stand-in exposing the SmolVLAPolicy internals the explainer touches.

    Must be an nn.Module: the adapter calls ``policy.eval()`` and derives the
    device from ``next(policy.parameters())``.

    Note: the real SmolVLAPolicy's ``_queues`` only ever contains ACTION, so its queued
    observation tensors never get stacked; this fake queues observation keys too, meaning
    the queue-stack path exercised here is a superset of real behavior (real batches stay
    4-dim — the adapter handles both).
    """

    def __init__(
        self,
        action_dim: int = 4,
        model: torch.nn.Module | None = None,
        image_keys: list[str] | None = None,
    ):
        super().__init__()
        image_keys = [CAM_KEY] if image_keys is None else image_keys
        self.config = SimpleNamespace(
            image_features=image_keys,
            action_feature=SimpleNamespace(shape=(action_dim,)),
            adapt_to_pi_aloha=False,
        )
        # submodule → parameters()/device work
        self.model = _FakeFlowMatching(action_dim) if model is None else model
        self._queues = {
            **{k: deque(maxlen=1) for k in image_keys},
            OBS_STATE: deque(maxlen=1),
            OBS_LANGUAGE_TOKENS: deque(maxlen=1),
            OBS_LANGUAGE_ATTENTION_MASK: deque(maxlen=1),
        }

    def _prepare_batch(self, batch):
        return batch

    def prepare_images(self, batch):
        images, masks = [], []
        for key in self.config.image_features:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            images.append(img)
            masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))
        return images, masks

    def prepare_state(self, batch):
        return batch[OBS_STATE]


class _FakeRTCGuidedFlowMatching(_FakeFlowMatching):
    """Wraps sample_actions with the RTC-guidance autograd pattern (detach + inner grad).

    Replicates the risky autograd interaction of ``RTCProcessor.denoise_step``
    (src/lerobot/policies/rtc/modeling_rtc.py:177-224): the noisy input is detached,
    the base velocity is recomputed under ``enable_grad`` from the (image-dependent)
    denoiser, and a correction is taken via an inner ``torch.autograd.grad(...,
    retain_graph=False)`` wrt the detached copy before being blended into the result.
    """

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        prev = kwargs.get("prev_chunk_left_over")
        if prev is None:
            return super().sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        time = 0.7  # denoising time in (0, 1); the guidance math runs regardless of its value
        x_t = noise.clone().detach()  # guidance operates on a detached copy, like the real wrapper
        with torch.enable_grad():
            v_t = super().sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=x_t)
            x_t.requires_grad_(True)
            x1_t = x_t - time * v_t
            err = prev - x1_t
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        self.last_kwargs = {"noise": noise, **kwargs}
        return v_t - 0.5 * correction


class _SecondCameraOnlyFlowMatching(_FakeFlowMatching):
    """Uses only ``images[1]`` — the first camera is inert (its outer grad is None)."""

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        self.last_kwargs = {"noise": noise, **kwargs}
        x = self.conv(images[1]).mean(dim=(2, 3))
        first = self.head(x)  # (B, action_dim)
        return first.unsqueeze(1).repeat(1, 10, 1)  # (B, chunk, action_dim)


def _make_fake_batch(h: int = 32, w: int = 32) -> dict:
    return {
        CAM_KEY: torch.rand(1, 3, h, w),
        OBS_STATE: torch.rand(1, 6),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.long),
    }


def test_get_explainer_cls_fails_fast_for_unregistered_policy():
    from lerobot.rollout.saliency.explainer import get_explainer_cls

    with pytest.raises(ValueError, match="No saliency explainer registered"):
        get_explainer_cls("definitely_not_registered")


def test_smolvla_explainer_is_registered():
    from lerobot.rollout.saliency import SmolVLAExplainer, get_explainer_cls

    assert get_explainer_cls("smolvla") is SmolVLAExplainer


def test_smolvla_explainer_produces_per_camera_maps():
    from lerobot.rollout.saliency import SmolVLAExplainer

    policy = _FakeSmolVLAPolicy()
    explainer = SmolVLAExplainer(policy)
    noise = torch.randn(1, 10, 4)
    chunk_kwargs = {"inference_delay": 2, "prev_chunk_left_over": None}

    maps = explainer.compute_saliency(_make_fake_batch(), noise, chunk_kwargs)

    assert set(maps) == {CAM_KEY}
    m = maps[CAM_KEY]
    assert m.shape == (32, 32)
    assert np.isfinite(m).all()
    assert (m >= 0).all()
    # chunk kwargs (incl. noise for exact replay) threaded through to the model
    assert policy.model.last_kwargs["inference_delay"] == 2
    assert policy.model.last_kwargs["noise"] is noise


def test_smolvla_explainer_map_depends_on_image():
    """Interior pixels of a uniform image yield a uniform map; a changed image changes it.

    The conv's zero padding makes the 1px border participate in fewer outputs, so
    uniformity is only asserted on the interior crop.
    """
    from lerobot.rollout.saliency import SmolVLAExplainer

    explainer = SmolVLAExplainer(_FakeSmolVLAPolicy())
    noise = torch.randn(1, 10, 4)
    kwargs = {"inference_delay": 0, "prev_chunk_left_over": None}

    uniform = _make_fake_batch()
    uniform[CAM_KEY] = torch.full((1, 3, 32, 32), 0.5)
    m0 = explainer.compute_saliency(uniform, noise, kwargs)[CAM_KEY]
    assert np.allclose(m0[2:-2, 2:-2], m0[2, 2])

    spotty = uniform.copy()
    spotty[CAM_KEY] = torch.rand(1, 3, 32, 32)
    m1 = explainer.compute_saliency(spotty, noise, kwargs)[CAM_KEY]
    assert not np.allclose(m1[2:-2, 2:-2], m1[2, 2])


def test_smolvla_explainer_rejects_missing_noise():
    """noise=None (snapshot from an engine without noise injection) is a contract error."""
    from lerobot.rollout.saliency import SmolVLAExplainer

    explainer = SmolVLAExplainer(_FakeSmolVLAPolicy())
    with pytest.raises(ValueError, match="injected noise"):
        explainer.compute_saliency(_make_fake_batch(), None, {"inference_delay": 0})


def test_smolvla_explainer_survives_rtc_guidance_pattern():
    """The image gradient survives an RTC-guidance-style inner detached-subgraph grad.

    RTCProcessor.denoise_step detaches its input and takes an internal
    ``torch.autograd.grad(..., retain_graph=False)``; if that inner backward freed or
    severed the outer image→action graph, every guided (steady-state) frame would lose
    its saliency map.
    """
    from lerobot.rollout.saliency import SmolVLAExplainer

    policy = _FakeSmolVLAPolicy(model=_FakeRTCGuidedFlowMatching(4))
    explainer = SmolVLAExplainer(policy)
    noise = torch.randn(1, 10, 4)
    prev = torch.rand(1, 10, 4)
    maps = explainer.compute_saliency(
        _make_fake_batch(), noise, {"inference_delay": 2, "prev_chunk_left_over": prev}
    )

    m = maps[CAM_KEY]
    assert m.shape == (32, 32)
    assert np.isfinite(m).all()
    assert m.sum() > 0  # image gradient survived the inner detached-subgraph grad


def test_smolvla_explainer_zero_map_for_inert_camera():
    """A camera the model ignores yields grad=None → all-zeros map of the right shape."""
    from lerobot.rollout.saliency import SmolVLAExplainer

    policy = _FakeSmolVLAPolicy(
        model=_SecondCameraOnlyFlowMatching(4),
        image_keys=[CAM_KEY, CAM_KEY_2],
    )
    explainer = SmolVLAExplainer(policy)
    noise = torch.randn(1, 10, 4)
    batch = _make_fake_batch()
    batch[CAM_KEY_2] = torch.rand(1, 3, 32, 32)

    maps = explainer.compute_saliency(batch, noise, {"inference_delay": 0, "prev_chunk_left_over": None})

    assert set(maps) == {CAM_KEY, CAM_KEY_2}
    inert = maps[CAM_KEY]
    assert inert.shape == (32, 32)
    assert (inert == 0).all()  # ignored camera → allow_unused grad=None → zeros fallback
    assert maps[CAM_KEY_2].sum() > 0


def test_smolvla_explainer_replays_real_policy_bitwise(monkeypatch):
    """Real (randomly initialized, tiny) SmolVLA: the explainer's grad replay must reproduce
    predict_action_chunk exactly given the same noise — including through the RTC guidance
    path — produce finite nonzero maps, and leave the caller's batch untouched."""
    pytest.importorskip("transformers", reason="SmolVLA requires transformers (install lerobot[smolvla])")

    import copy

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.rollout.saliency.smolvla import SmolVLAExplainer

    torch.manual_seed(0)
    action_dim, chunk = 7, 10
    config = SmolVLAConfig(
        max_action_dim=action_dim,
        chunk_size=chunk,
        n_action_steps=chunk,
        num_steps=3,  # fewer denoise steps for speed
        load_vlm_weights=False,
        num_vlm_layers=2,
        num_expert_layers=2,
        device="cpu",
    )
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=4,
        max_guidance_weight=5.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )
    cam = "observation.images.base_0_rgb"
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    policy = SmolVLAPolicy(config)
    policy.eval()

    # Batch shaped like the RTC engine's preprocessed output.
    batch = {
        cam: torch.rand(1, 3, 224, 224),
        OBS_STATE: torch.rand(1, 14),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (1, 8)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 8, dtype=torch.bool),
    }
    noise = torch.randn(1, chunk, action_dim)

    captured = {}
    orig_sample_actions = policy.model.sample_actions

    def capture_sample_actions(*args, **kwargs):
        out = orig_sample_actions(*args, **kwargs)
        captured["actions"] = out.detach().clone()
        return out

    monkeypatch.setattr(policy.model, "sample_actions", capture_sample_actions)
    explainer = SmolVLAExplainer(policy)

    for chunk_kwargs in (
        {"inference_delay": 0, "prev_chunk_left_over": None},
        {
            "inference_delay": 2,
            "prev_chunk_left_over": torch.randn(4, action_dim),
        },  # 2D, as the engine passes
    ):
        with torch.no_grad():
            ref = policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone(), **chunk_kwargs)
        maps = explainer.compute_saliency(copy.deepcopy(batch), noise.clone(), dict(chunk_kwargs))
        replayed = captured["actions"][:, :, :action_dim]
        assert (replayed - ref).abs().max().item() < 1e-5  # replay == executed chunk
        sal = maps[cam]
        assert sal.shape == (224, 224)
        assert np.isfinite(sal).all()
        assert sal.sum() > 0

    # Caller-batch integrity: compute_saliency must not mutate the snapshot it is given.
    snap = copy.deepcopy(batch)
    before = {k: v.clone() for k, v in snap.items()}
    explainer.compute_saliency(snap, noise.clone(), {"inference_delay": 0, "prev_chunk_left_over": None})
    for key, value in before.items():
        assert torch.equal(snap[key], value), f"compute_saliency mutated caller batch key {key}"


# ---------------------------------------------------------------------------
# SaliencyRecorder (side thread, drop-oldest queue, per-camera mp4s)
# ---------------------------------------------------------------------------


class _FixedExplainer:
    def __init__(self):
        self.calls = 0

    def compute_saliency(self, batch, noise, chunk_kwargs):
        self.calls += 1
        h, w = batch[CAM_KEY].shape[-2:]
        return {CAM_KEY: np.full((h, w), 0.5, dtype=np.float32)}


def test_recorder_drops_stale_snapshots_and_writes_video(tmp_path):
    import cv2

    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    explainer = _FixedExplainer()
    out = tmp_path / "saliency"
    recorder = SaliencyRecorder(
        explainer=explainer,
        output_dir=str(out),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    recorder.submit(snap)
    recorder.submit(snap)  # queue cap 1: first submission must be dropped
    recorder.start()
    recorder.close()

    assert explainer.calls == 1
    video = out / "cam1.mp4"  # filename = last path segment of the camera key
    assert video.exists() and video.stat().st_size > 0
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1
    cap.release()


def test_recorder_close_is_idempotent(tmp_path):
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    recorder = SaliencyRecorder(
        explainer=_FixedExplainer(),
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        display_mode="rerun",
    )
    recorder.start()
    recorder.close()
    recorder.close()  # must not raise, must not double-release writers


def test_recorder_survives_sentinel_steal_race(tmp_path):
    """A concurrent submit must not steal close()'s sentinel and strand the thread."""
    import threading

    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    class _SlowExplainer:
        def __init__(self):
            self.started = threading.Event()

        def compute_saliency(self, batch, noise, chunk_kwargs):
            self.started.set()
            time.sleep(0.3)
            h, w = batch[CAM_KEY].shape[-2:]
            return {CAM_KEY: np.zeros((h, w), dtype=np.float32)}

    explainer = _SlowExplainer()
    recorder = SaliencyRecorder(
        explainer=explainer,
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=0.0,
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    recorder.start()
    recorder.submit(snap)  # thread picks this up; the slow explainer keeps it busy
    assert explainer.started.wait(timeout=5.0)  # slow compute is running when close() lands

    closer = threading.Thread(target=recorder.close, daemon=True)
    closer.start()
    # Wait until close() has enqueued the sentinel (thread still busy → queue cap 1 is full).
    deadline = time.monotonic() + 5.0
    while not recorder._closed or recorder._queue.qsize() == 0:
        if time.monotonic() > deadline:
            raise AssertionError("close() never enqueued its sentinel")
        time.sleep(0.005)

    late = threading.Thread(target=recorder.submit, args=(snap,), daemon=True)
    late.start()  # racing submit against a queue already holding the sentinel
    late.join(timeout=5.0)
    assert not late.is_alive()
    closer.join(timeout=15.0)  # must return well under the 30s join timeout
    assert not closer.is_alive()
    assert not recorder._thread.is_alive()  # exited despite the (potentially) stolen sentinel


def test_recorder_drops_non_finite_saliency_and_warns_once(tmp_path, caplog):
    """NaN/Inf gradients must not flow into the mp4: the frame is dropped and the
    FP blowup is surfaced with a single warning per camera."""
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    class _NonFiniteExplainer:
        def __init__(self):
            self.calls = 0

        def compute_saliency(self, batch, noise, chunk_kwargs):
            self.calls += 1
            h, w = batch[CAM_KEY].shape[-2:]
            return {CAM_KEY: np.full((h, w), np.nan, dtype=np.float32)}

    explainer = _NonFiniteExplainer()
    out = tmp_path / "saliency"
    recorder = SaliencyRecorder(
        explainer=explainer,
        output_dir=str(out),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=0.0,
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    recorder.start()
    with caplog.at_level("WARNING", logger="lerobot.rollout.saliency.recorder"):
        for processed in (1, 2):
            recorder.submit(snap)
            deadline = time.monotonic() + 5.0
            while explainer.calls < processed and time.monotonic() < deadline:
                time.sleep(0.01)
    recorder.close()

    assert explainer.calls == 2
    assert list(out.iterdir()) == []  # every frame dropped → no writer was ever created
    warnings = [r for r in caplog.records if "Non-finite saliency" in r.message]
    assert len(warnings) == 1  # warn-once per camera, not per frame


def _norm_range_spy(monkeypatch) -> list:
    """Patch recorder.render_overlay with a spy recording each call's norm_range; writers → null."""
    from lerobot.rollout.saliency import recorder as recorder_mod
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    seen: list = []
    real_render_overlay = recorder_mod.render_overlay

    def spy_render_overlay(img_bgr, saliency, alpha=0.5, blur_sigma=2.0, norm_range=None):
        seen.append(norm_range)
        return real_render_overlay(img_bgr, saliency, alpha, blur_sigma, norm_range)

    class _NullWriter:
        def write(self, frame):
            return None

    monkeypatch.setattr(recorder_mod, "render_overlay", spy_render_overlay)
    monkeypatch.setattr(SaliencyRecorder, "_writer", lambda self, *args: _NullWriter())
    return seen


class _SequenceExplainer:
    """compute_saliency returns a uniform map per camera at a scripted magnitude per call
    (the last magnitude repeats once the list is exhausted)."""

    def __init__(self, magnitudes: list[float]):
        self._magnitudes = list(magnitudes)

    def compute_saliency(self, batch, noise, chunk_kwargs):
        magnitude = self._magnitudes.pop(0) if len(self._magnitudes) > 1 else self._magnitudes[0]
        h, w = batch[CAM_KEY].shape[-2:]
        return {CAM_KEY: np.full((h, w), magnitude, dtype=np.float32)}


def _make_recorder(tmp_path, explainer, **kwargs):
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    return SaliencyRecorder(
        explainer=explainer,
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=0.0,
        display=False,
        display_mode="rerun",
        **kwargs,
    )


def _snapshot() -> dict:
    return {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}


def test_recorder_shares_norm_range_across_cameras(tmp_path, monkeypatch):
    """All cameras of one snapshot must be rendered against ONE shared (min, max) — otherwise
    a camera with negligible contribution is stretched to full red by its private min-max."""
    seen = _norm_range_spy(monkeypatch)

    class _TwoScaleExplainer:
        def compute_saliency(self, batch, noise, chunk_kwargs):
            h, w = batch[CAM_KEY].shape[-2:]
            ramp = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))  # 0..1
            return {CAM_KEY: ramp, CAM_KEY_2: ramp * 1e-3}  # 0..0.001

    recorder = _make_recorder(tmp_path, _TwoScaleExplainer())
    batch = _make_fake_batch() | {CAM_KEY_2: torch.rand(1, 3, 32, 32)}
    recorder._process({"preprocessed": batch, "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}})

    assert len(seen) == 2  # one render call per camera
    assert seen[0] is seen[1]  # literally the same shared range object
    lo, hi = seen[0]
    assert lo == 0.0  # run scope: lo is the natural floor, not the weakest camera's min
    assert hi == pytest.approx(1.0, abs=1e-7)  # spans the STRONG camera, not each private max


def test_recorder_run_scope_shares_scale_across_time(tmp_path, monkeypatch):
    """norm_scope='run': a quiet chunk following a hot chunk keeps the hot chunk's (decayed)
    scale instead of stretching itself to full red."""
    seen = _norm_range_spy(monkeypatch)
    recorder = _make_recorder(tmp_path, _SequenceExplainer([1.0, 0.01]))

    recorder._process(_snapshot())  # hot chunk sets the scale
    recorder._process(_snapshot())  # quiet chunk must inherit it

    assert seen[0] == (0.0, pytest.approx(1.0))
    lo, hi = seen[1]
    assert lo == 0.0
    assert hi == pytest.approx(0.99)  # decayed running max, NOT the quiet chunk's own 0.01


def test_recorder_run_scope_decays_toward_current_levels(tmp_path, monkeypatch):
    """A one-off spike must not suppress the scale forever: the running max decays so the
    display recovers (~half-life 69 chunks at decay 0.99)."""
    seen = _norm_range_spy(monkeypatch)
    recorder = _make_recorder(tmp_path, _SequenceExplainer([1.0, 0.01]))

    recorder._process(_snapshot())  # the spike
    for _ in range(70):
        recorder._process(_snapshot())  # quiet chunks

    assert seen[-1][1] == pytest.approx(0.99**70, rel=1e-3)  # recovered about halfway to 0.01


def test_recorder_chunk_scope_rescales_per_chunk(tmp_path, monkeypatch):
    """norm_scope='chunk': every chunk re-normalizes to its own cross-camera range."""
    seen = _norm_range_spy(monkeypatch)
    recorder = _make_recorder(tmp_path, _SequenceExplainer([1.0, 0.01]), norm_scope="chunk")

    recorder._process(_snapshot())
    recorder._process(_snapshot())

    assert seen[0] == (pytest.approx(1.0), pytest.approx(1.0))  # uniform map: min == max
    assert seen[1][1] == pytest.approx(0.01)  # quiet chunk stretched to its own scale


# ---------------------------------------------------------------------------
# RTC engine: noise injection snapshot builder
# ---------------------------------------------------------------------------


def test_build_saliency_snapshot_clones_tensors():
    from lerobot.rollout.inference.rtc import _build_saliency_snapshot

    imgs = torch.rand(1, 3, 8, 8)
    noise = torch.rand(1, 10, 4)
    prev = torch.rand(4, 4)
    preprocessed = {CAM_KEY: imgs, "task": ["pick"]}

    snap = _build_saliency_snapshot(preprocessed, noise=noise, inference_delay=3, prev_chunk_left_over=prev)

    assert snap["preprocessed"][CAM_KEY] is not imgs
    assert snap["preprocessed"]["task"] == ["pick"]
    assert snap["preprocessed"]["task"] is preprocessed["task"]  # non-tensors pass by reference
    assert torch.equal(snap["noise"], noise)
    assert snap["chunk_kwargs"]["inference_delay"] == 3
    assert torch.equal(snap["chunk_kwargs"]["prev_chunk_left_over"], prev)
    snap["noise"].zero_()
    assert not torch.allclose(noise, torch.zeros_like(noise))  # deep copy, not a view


def test_build_saliency_snapshot_handles_none_optionals():
    from lerobot.rollout.inference.rtc import _build_saliency_snapshot

    snap = _build_saliency_snapshot({}, noise=None, inference_delay=0, prev_chunk_left_over=None)
    assert snap["noise"] is None
    assert snap["chunk_kwargs"]["prev_chunk_left_over"] is None
    assert snap["chunk_kwargs"]["inference_delay"] == 0


def test_accepts_noise_injection_probes_noise_kwarg():
    """The saliency build-time probe must reject RTC-capable policies whose
    predict_action_chunk lacks the noise parameter (they would TypeError mid-loop)."""
    from lerobot.rollout.inference.rtc import accepts_noise_injection

    class _NoiseCapable:
        def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None, noise=None):
            raise NotImplementedError

    class _NoNoise:
        def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None):
            raise NotImplementedError

    assert accepts_noise_injection(_NoiseCapable())
    assert not accepts_noise_injection(_NoNoise())


# ---------------------------------------------------------------------------
# SaliencyManager: device resolution + registry fail-fast
# ---------------------------------------------------------------------------


def test_resolve_explainer_device_prefers_second_gpu(monkeypatch):
    from lerobot.rollout.saliency.manager import resolve_explainer_device

    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    assert resolve_explainer_device(None, "cuda:0") == "cuda:1"
    assert resolve_explainer_device(None, "cpu") == "cuda:0"
    assert resolve_explainer_device(None, "cuda:1") == "cuda:0"
    assert resolve_explainer_device("cuda:1", "cuda:0") == "cuda:1"  # explicit wins
    # mps/xpu/... are left alone: cross-vendor defaults would be surprising.
    assert resolve_explainer_device(None, "mps") == "mps"

    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    assert resolve_explainer_device(None, "cuda:0") == "cuda:0"  # fallback: same device
    assert resolve_explainer_device(None, "cpu") == "cpu"
    assert resolve_explainer_device(None, "mps") == "mps"


def test_saliency_manager_fails_fast_on_unregistered_policy(tmp_path):
    from lerobot.rollout.saliency.manager import SaliencyManager

    def _load():
        raise AssertionError("loader must not run when the policy type is unregistered")

    cfg = SimpleNamespace(
        device=None, output_dir=str(tmp_path), video_fps=10.0, alpha=0.5, blur_sigma=2.0, display=False
    )
    policy_config = SimpleNamespace(type="definitely_not_registered")

    with pytest.raises(ValueError, match="No saliency explainer registered"):
        SaliencyManager(cfg, policy_config, policy_device="cpu", display_mode="rerun", load_policy=_load)
