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
# Patch-grid overlay + attention budget bar (method="attention")
# ---------------------------------------------------------------------------


def test_render_patch_overlay_blocky_and_grid_lines():
    """Square frame (no letterbox padding): the heat is piecewise-constant per patch and
    grid lines sit exactly on the uniform patch boundaries."""
    from lerobot.rollout.saliency.overlay import render_patch_overlay

    frame = np.zeros((80, 80, 3), dtype=np.uint8)  # 8×8 grid → 10px per patch, no padding
    patch_map = np.arange(64, dtype=np.float32).reshape(8, 8)

    overlay = render_patch_overlay(frame, patch_map, alpha=0.5, norm_range=(0.0, 63.0))

    # Heat piecewise-constant inside each 10×10 cell (frame is black → blend shows heat only).
    for r, c in ((0, 0), (3, 5), (7, 7)):
        cell = overlay[r * 10 + 1 : (r + 1) * 10 - 1, c * 10 + 1 : (c + 1) * 10 - 1]
        assert (cell == cell[0, 0]).all(), f"patch ({r},{c}) is not a solid block"
    assert (overlay[1:9, 1:9] != overlay[71:79, 71:79]).any()  # distinct patches differ

    # Grid lines are pure white at every uniform boundary, and only there.
    cold = render_patch_overlay(frame, np.zeros((8, 8), dtype=np.float32), alpha=0.5, norm_range=(0.0, 1.0))
    for y in (10, 20, 30, 40, 50, 60, 70):
        assert (cold[y] == 255).all(), f"missing horizontal grid line at y={y}"
    for x in (10, 70):
        assert (cold[:, x] == 255).all(), f"missing vertical grid line at x={x}"
    # Patch interior: no grid line, uniform blended cold color (TURBO(0) is dark purple, not
    # black — so a cold blend over a black frame is nonzero; assert uniformity instead).
    # Columns 1:9 stay inside cell (0,0) — a full row would cross the vertical grid lines.
    assert (cold[5, 1:9] == cold[5, 1]).all()
    assert (cold[5, 1] != 255).any()  # not a grid pixel


def test_render_patch_overlay_letterbox_top_band():
    """720p frame letterboxed to 512×512 (224px top padding, content bottom-aligned):
    patch row 3 renders only its bottom half as an 80px top band; grid lines land on the
    TRUE token boundaries (80/240/400/560), not on a naive uniform grid."""
    from lerobot.rollout.saliency.overlay import render_patch_overlay

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    patch_map = np.zeros((8, 8), dtype=np.float32)
    patch_map[3, :] = 1.0  # the patch row straddling the padding/content boundary

    overlay = render_patch_overlay(
        frame, patch_map, alpha=0.8, norm_range=(0.0, 1.0), letterbox_hw=(512, 512)
    )

    heat_col = overlay[:, 80, :]  # column 80 = cell center, away from vertical grid lines (160k)
    hot, cold = heat_col[40], heat_col[160]
    assert (hot != cold).any()
    assert (heat_col[:80] == hot).all()  # top band: bottom half of patch row 3, uniformly hot
    assert (heat_col[81:239] == cold).all()  # patch row 4 uniformly cold (only row 3 has mass)
    # True grid lines (content starts at letterbox y=224 → first visible boundary at 32px
    # of content = 80px on the frame; then every 160px).
    for y in (80, 240, 400, 560):
        assert (overlay[y] == 255).all(), f"missing grid line at y={y}"
    # Naive uniform grid would draw at y=90 and y=640: neither is a true boundary.
    assert (overlay[90, 80] == cold).all()
    assert (overlay[640, 80] == cold).all()


def test_patch_content_mapping_mirrors_resize_with_pad():
    """The mapping used for grid placement must reproduce resize_with_pad's left/top geometry."""
    from lerobot.rollout.saliency.overlay import patch_content_mapping

    pad_h, pad_w, resized_h, resized_w = patch_content_mapping((720, 1280), (512, 512))
    assert (pad_h, pad_w) == (224, 0)  # top-only padding, bottom-aligned content
    assert (resized_h, resized_w) == (288, 512)

    # Square input: no padding at all.
    assert patch_content_mapping((224, 224), (512, 512)) == (0, 0, 512, 512)


def test_draw_budget_bar_full_width_segments_and_labels():
    from lerobot.rollout.saliency.overlay import _BUDGET_CAM_PALETTE, _BUDGET_SPECIAL_COLORS, draw_budget_bar

    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    segments = [("cam1", 0.5), ("cam2", 0.3), ("language", 0.15), ("state", 0.05)]
    out = draw_budget_bar(frame, segments, bar_height=24)

    assert (out[5] != 0).any(axis=1).all()  # bar strip painted across the full width
    assert (out[50] == 0).all()  # frame below the bar untouched
    # Segment colors are kind-based (row 5 is above the text glyphs): cameras cycle the
    # palette in draw order (cam1 200px, cam2 120px), language/state keep their fixed
    # neutrals regardless of position (language 60px, state 20px).
    assert (out[5, 150] == _BUDGET_CAM_PALETTE[0]).all()
    assert (out[5, 300] == _BUDGET_CAM_PALETTE[1]).all()
    assert (out[5, 370] == _BUDGET_SPECIAL_COLORS["language"]).all()
    assert (out[5, 395] == _BUDGET_SPECIAL_COLORS["state"]).all()
    # Percentage label burned into a wide-enough segment (LINE_AA text → near-white pixels).
    assert out[8:18, 3:63].max() >= 200
    # State segment (20px) is too narrow for its label: bar painted, no text requirement.

    # Shares that don't sum to 1 exactly still span the full width (last segment absorbs).
    out2 = draw_budget_bar(frame, [("cam1", 0.333), ("language", 0.333)], bar_height=24)
    assert (out2[5] != 0).any(axis=1).all()


def test_budget_segments_orders_cameras_then_language_state():
    from lerobot.rollout.saliency.recorder import _budget_segments

    budget = {
        "observation.images.camera1": 0.1,
        "observation.images.camera2": 0.2,
        "observation.images.camera3": 0.3,
        "language": 0.35,
        "state": 0.05,
    }
    assert _budget_segments(budget) == [
        ("camera1", 0.1),
        ("camera2", 0.2),
        ("camera3", 0.3),
        ("language", 0.35),
        ("state", 0.05),
    ]


def test_recorder_patch_kind_writes_video_with_budget(tmp_path):
    """A patch-kind explainer (8×8 maps + last_budget) flows through the recorder end to end."""
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    class _PatchExplainer:
        map_kind = "patch"

        def __init__(self):
            self.last_budget = None
            self.calls = 0

        def compute_saliency(self, batch, noise, chunk_kwargs):
            self.calls += 1
            self.last_budget = {CAM_KEY: 0.5, "language": 0.4, "state": 0.1}
            return {CAM_KEY: np.full((8, 8), 0.25, dtype=np.float32)}

    explainer = _PatchExplainer()
    out = tmp_path / "saliency"
    recorder = SaliencyRecorder(
        explainer=explainer,
        output_dir=str(out),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,  # must be ignored for patch maps (blocky, never blurred)
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    recorder.submit(snap)
    recorder.start()
    recorder.close()

    assert explainer.calls == 1
    video = out / "cam1.mp4"
    assert video.exists() and video.stat().st_size > 0


def test_recorder_drops_nonfinite_budget_but_keeps_frame(tmp_path, caplog):
    """A NaN attention budget (FP blowup in language/state columns) must not cost the frame:
    the patch overlay is still written, only the bar/curves are dropped, with one warning."""
    import logging as _logging

    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    class _NanBudgetExplainer:
        map_kind = "patch"

        @property
        def last_budget(self):
            return {CAM_KEY: 0.5, "language": float("nan"), "state": 0.1}

        def compute_saliency(self, batch, noise, chunk_kwargs):
            return {CAM_KEY: np.full((8, 8), 0.25, dtype=np.float32)}

    out = tmp_path / "saliency"
    recorder = SaliencyRecorder(
        explainer=_NanBudgetExplainer(),
        output_dir=str(out),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    with caplog.at_level(_logging.WARNING):
        _run_recorder_snapshots(recorder, [snap, snap])  # two frames: warning must not repeat

    video = out / "cam1.mp4"
    assert video.exists() and video.stat().st_size > 0  # frames kept despite the NaN budget
    warnings = [r for r in caplog.records if "Non-finite attention budget" in r.message]
    assert len(warnings) == 1  # warn-once, not per frame


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
# Attention capture callback (SmolVLMWithExpertModel.attention_capture)
# ---------------------------------------------------------------------------


def _bare_expert_model(num_heads: int = 3, num_kv_heads: int = 1):
    """Bare SmolVLMWithExpertModel for exercising eager_attention_forward in isolation.

    Built via ``__new__`` (no weights, no transformers dependency): the attention math
    only touches ``num_attention_heads`` / ``num_key_value_heads`` / ``attention_capture``.
    """
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    model = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    model.num_attention_heads = num_heads
    model.num_key_value_heads = num_kv_heads
    model.attention_capture = None
    return model


def _attention_inputs(
    batch: int = 1, lq: int = 5, lk: int = 7, heads: int = 3, kv_heads: int = 1, dim: int = 4
):
    q = torch.randn(batch, lq, heads, dim)
    k = torch.randn(batch, lk, kv_heads, dim)  # GQA: K/V carry kv_heads, expanded inside
    v = torch.randn(batch, lk, kv_heads, dim)
    mask = torch.ones(batch, lq, lk, dtype=torch.bool)
    return mask, q, k, v


def test_attention_capture_default_off_runs_clean():
    model = _bare_expert_model()
    assert model.attention_capture is None
    mask, q, k, v = _attention_inputs()
    out = model.eager_attention_forward(mask, 1, 4, q, k, v, layer_idx=3, kind="cross")
    assert out.shape == (1, 5, 12)  # (B, Lq, heads*dim); no capture machinery engaged


def test_attention_capture_receives_float32_probs_layer_and_kind():
    model = _bare_expert_model()
    events = []
    model.attention_capture = lambda layer_idx, probs, kind: events.append((layer_idx, probs, kind))

    mask, q, k, v = _attention_inputs()
    mask[:, :, -2:] = False  # exercise the masked → zero-probability path
    model.eager_attention_forward(mask, 1, 4, q, k, v, layer_idx=7, kind="cross")

    assert len(events) == 1
    layer_idx, probs, kind = events[0]
    assert layer_idx == 7
    assert kind == "cross"
    assert probs.dtype == torch.float32  # tapped before the cast back to the value dtype
    assert probs.shape == (1, 3, 5, 7)  # (B, heads, Lq, Lk)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 3, 5), atol=1e-6)
    assert (probs[:, :, :, -2:] == 0).all()  # masked keys carry no mass


def test_attention_capture_is_per_instance():
    tapped, untapped = _bare_expert_model(), _bare_expert_model()
    events = []
    tapped.attention_capture = lambda *args: events.append(args)

    mask, q, k, v = _attention_inputs()
    untapped.eager_attention_forward(mask, 1, 4, q, k, v, layer_idx=0, kind="self")
    assert events == []  # the control-loop model can never fire another instance's callback

    tapped.eager_attention_forward(mask, 1, 4, q, k, v, layer_idx=0, kind="self")
    assert len(events) == 1


def test_attention_capture_reset_stops_events():
    model = _bare_expert_model()
    events = []
    model.attention_capture = lambda *args: events.append(args)
    mask, q, k, v = _attention_inputs()
    model.eager_attention_forward(mask, 1, 4, q, k, v, layer_idx=1, kind="cross")
    assert len(events) == 1

    model.attention_capture = None
    model.eager_attention_forward(mask, 1, 4, q, k, v, layer_idx=1, kind="cross")
    assert len(events) == 1


# ---------------------------------------------------------------------------
# SmolVLAAttentionExplainer (method="attention": patch maps + budget)
# ---------------------------------------------------------------------------


class _ScriptedVLM:
    """Stand-in for SmolVLMWithExpertModel: just an ``attention_capture`` attribute."""

    def __init__(self):
        self.attention_capture = None


class _ScriptedAttentionFlowMatching(_FakeFlowMatching):
    """Drives the vlm's attention_capture with a scripted event list inside sample_actions."""

    def __init__(self, action_dim: int, vlm: _ScriptedVLM, script: list):
        super().__init__(action_dim)
        self.vlm_with_expert = vlm
        self._script = script

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        for layer_idx, probs, kind in self._script:
            self.vlm_with_expert.attention_capture(layer_idx, probs, kind)
        return super().sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)


def _make_attention_policy(script: list, image_keys: list[str] | None = None, lang_len: int = 4):
    vlm = _ScriptedVLM()
    model = _ScriptedAttentionFlowMatching(4, vlm, script)
    policy = _FakeSmolVLAPolicy(action_dim=4, model=model, image_keys=image_keys or [CAM_KEY])
    policy.config.tokenizer_max_length = lang_len
    return policy, vlm


def _probs(a0_row: np.ndarray, heads: int = 2, queries: int = 3) -> torch.Tensor:
    """(1, heads, queries, Lk) probs whose a₀ query row is ``a0_row`` on every head."""
    lk = a0_row.shape[0]
    probs = torch.full((1, heads, queries, lk), 1.0 / lk)  # non-a₀ rows: arbitrary, must be ignored
    probs[:, :, 0, :] = torch.from_numpy(np.asarray(a0_row, dtype=np.float32))[None, :]
    return probs


_CHUNK_KWARGS = {"inference_delay": 0, "prev_chunk_left_over": None}


def test_attention_explainer_is_registered_for_attention_method():
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer, SmolVLAExplainer, get_explainer_cls

    assert get_explainer_cls("smolvla") is SmolVLAExplainer  # grad_x_input default unchanged
    assert get_explainer_cls("smolvla", "attention") is SmolVLAAttentionExplainer
    with pytest.raises(ValueError, match="integrated"):
        get_explainer_cls("smolvla", "integrated")


def test_validate_saliency_accepts_attention_method():
    from lerobot.rollout.inference import RTCInferenceConfig, SyncInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    validate_saliency(SaliencyConfig(enabled=True, method="attention"), RTCInferenceConfig(), "rerun", True)
    with pytest.raises(ValueError, match="inference.type=rtc"):
        validate_saliency(
            SaliencyConfig(enabled=True, method="attention"), SyncInferenceConfig(), "rerun", True
        )


def test_attention_explainer_column_attribution():
    """All mass on camera 2's 64 columns → only camera 2 lights up; budget follows."""
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    lang_len = 4
    lk = 64 * 2 + lang_len + 1
    a0 = np.zeros(lk, dtype=np.float32)
    a0[64:128] = 1.0 / 64
    policy, _ = _make_attention_policy(
        [(1, _probs(a0), "cross")], image_keys=[CAM_KEY, CAM_KEY_2], lang_len=lang_len
    )
    explainer = SmolVLAAttentionExplainer(policy)

    batch = _make_fake_batch()
    batch[CAM_KEY_2] = torch.rand(1, 3, 32, 32)
    maps = explainer.compute_saliency(batch, torch.randn(1, 10, 4), dict(_CHUNK_KWARGS))

    assert maps[CAM_KEY_2].shape == (8, 8)
    assert np.allclose(maps[CAM_KEY_2], 1.0 / 64)
    assert (maps[CAM_KEY] == 0).all()
    budget = explainer.last_budget
    assert budget[CAM_KEY_2] == pytest.approx(1.0)
    assert budget[CAM_KEY] == 0.0 and budget["language"] == 0.0 and budget["state"] == 0.0
    assert sum(budget.values()) == pytest.approx(1.0, abs=1e-4)


def test_attention_explainer_raw_shares_not_renormalized():
    """20% image mass must render as 20%, not stretched to full contrast."""
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    lang_len = 4
    lk = 64 + lang_len + 1
    a0 = np.zeros(lk, dtype=np.float32)
    a0[:64] = 0.2 / 64
    a0[64 : 64 + lang_len] = 0.8 / lang_len
    policy, _ = _make_attention_policy([(1, _probs(a0), "cross")], lang_len=lang_len)
    explainer = SmolVLAAttentionExplainer(policy)

    maps = explainer.compute_saliency(_make_fake_batch(), torch.randn(1, 10, 4), dict(_CHUNK_KWARGS))

    assert np.allclose(maps[CAM_KEY], 0.2 / 64)
    budget = explainer.last_budget
    assert budget[CAM_KEY] == pytest.approx(0.2, abs=1e-6)
    assert budget["language"] == pytest.approx(0.8, abs=1e-6)
    assert sum(budget.values()) == pytest.approx(1.0, abs=1e-4)


def test_attention_explainer_consumes_only_cross_events_and_averages():
    """Emulate the real firing pattern (prefill self + denoise self/cross): only cross counts,
    and the result is the mean over cross events (layers × denoise steps)."""
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    lang_len = 4
    lk = 64 + lang_len + 1
    a0_first = np.zeros(lk, dtype=np.float32)
    a0_first[:64] = 1.0 / 64  # step 1: all mass on the camera
    a0_second = np.zeros(lk, dtype=np.float32)
    a0_second[64:] = 1.0 / (lang_len + 1)  # step 2: all mass on language + state
    distractor = np.ones(lk, dtype=np.float32) / lk
    script = (
        [(i, _probs(distractor), "self") for i in range(2)]  # prefill: every layer is self
        + [(0, _probs(distractor), "self"), (1, _probs(a0_first), "cross")]  # denoise step 1
        + [(0, _probs(distractor), "self"), (1, _probs(a0_second), "cross")]  # denoise step 2
    )
    policy, _ = _make_attention_policy(script, lang_len=lang_len)
    explainer = SmolVLAAttentionExplainer(policy)

    maps = explainer.compute_saliency(_make_fake_batch(), torch.randn(1, 10, 4), dict(_CHUNK_KWARGS))

    assert explainer._events == 2  # 4 self events were filtered out
    assert np.allclose(maps[CAM_KEY], 0.5 / 64)  # mean of the two cross events
    budget = explainer.last_budget
    assert budget[CAM_KEY] == pytest.approx(0.5, abs=1e-6)
    assert budget["language"] == pytest.approx(0.5 * lang_len / (lang_len + 1), abs=1e-6)
    assert budget["state"] == pytest.approx(0.5 / (lang_len + 1), abs=1e-6)
    assert sum(budget.values()) == pytest.approx(1.0, abs=1e-4)


def test_attention_explainer_averages_over_heads():
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    lang_len = 4
    lk = 64 + lang_len + 1
    hot = np.zeros(lk, dtype=np.float32)
    hot[:64] = 1.0 / 64
    probs = torch.zeros(1, 2, 3, lk)
    probs[0, 0, 0, :] = torch.from_numpy(hot)  # head 0 sees the camera, head 1 sees nothing
    policy, _ = _make_attention_policy([(1, probs, "cross")], lang_len=lang_len)
    explainer = SmolVLAAttentionExplainer(policy)

    maps = explainer.compute_saliency(_make_fake_batch(), torch.randn(1, 10, 4), dict(_CHUNK_KWARGS))

    assert np.allclose(maps[CAM_KEY], 0.5 / 64)  # head-mean: (1/64 + 0) / 2


def test_attention_explainer_fails_fast_on_layout_mismatch():
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    lang_len = 4
    lk = 64 + lang_len + 1 + 1  # one column too many (e.g. pad_language_to changed)
    a0 = np.ones(lk, dtype=np.float32) / lk
    policy, _ = _make_attention_policy([(1, _probs(a0), "cross")], lang_len=lang_len)
    explainer = SmolVLAAttentionExplainer(policy)

    with pytest.raises(RuntimeError, match="Prefix layout mismatch"):
        explainer.compute_saliency(_make_fake_batch(), torch.randn(1, 10, 4), dict(_CHUNK_KWARGS))


def test_attention_explainer_raises_without_cross_events():
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    policy, _ = _make_attention_policy([])  # tap never fires
    explainer = SmolVLAAttentionExplainer(policy)
    with pytest.raises(RuntimeError, match="No cross-attention events"):
        explainer.compute_saliency(_make_fake_batch(), torch.randn(1, 10, 4), dict(_CHUNK_KWARGS))


def test_attention_explainer_rejects_missing_noise():
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    policy, _ = _make_attention_policy([])
    explainer = SmolVLAAttentionExplainer(policy)
    with pytest.raises(ValueError, match="injected noise"):
        explainer.compute_saliency(_make_fake_batch(), None, {"inference_delay": 0})


def test_attention_explainer_close_detaches_tap():
    from lerobot.rollout.saliency import SmolVLAAttentionExplainer

    policy, vlm = _make_attention_policy([])
    explainer = SmolVLAAttentionExplainer(policy)
    assert vlm.attention_capture == explainer._on_attention  # bound method: __eq__, not `is`
    explainer.close()
    assert vlm.attention_capture is None
    explainer.close()  # idempotent


def test_smolvla_attention_explainer_replays_real_policy():
    """Real (randomly initialized, tiny) SmolVLA: the attention replay must reproduce the
    direct forward's cross-attention on the same copy/device (<1e-5), fire the expected
    self/cross event counts, and leave the caller's batch untouched."""
    pytest.importorskip("transformers", reason="SmolVLA requires transformers (install lerobot[smolvla])")

    import copy

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.rollout.saliency.smolvla_attention import SmolVLAAttentionExplainer

    torch.manual_seed(0)
    action_dim, chunk = 7, 10
    config = SmolVLAConfig(
        max_action_dim=action_dim,
        chunk_size=chunk,
        n_action_steps=chunk,
        num_steps=3,  # fewer denoise steps for speed
        load_vlm_weights=False,
        num_vlm_layers=2,  # layer 0 self, layer 1 cross
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

    # Language pre-padded to tokenizer_max_length, exactly as the production preprocessor emits
    # it (the checkpoint sets pad_language_to="max_length"; default "longest" would fail the
    # explainer's layout assertion by design).
    lang_len = config.tokenizer_max_length
    batch = {
        cam: torch.rand(1, 3, 224, 224),
        OBS_STATE: torch.rand(1, 14),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (1, lang_len)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, lang_len, dtype=torch.bool),
    }
    noise = torch.randn(1, chunk, action_dim)

    explainer = SmolVLAAttentionExplainer(policy)
    counts = {"self": 0, "cross": 0}
    explainer_tap = policy.model.vlm_with_expert.attention_capture

    def counting_tap(layer_idx, probs, kind):
        counts[kind] += 1
        explainer_tap(layer_idx, probs, kind)

    policy.model.vlm_with_expert.attention_capture = counting_tap

    for chunk_kwargs in (
        {"inference_delay": 0, "prev_chunk_left_over": None},
        {"inference_delay": 2, "prev_chunk_left_over": torch.randn(4, action_dim)},
    ):
        counts["self"] = counts["cross"] = 0
        maps = explainer.compute_saliency(copy.deepcopy(batch), noise.clone(), dict(chunk_kwargs))

        # 2 prefill self + 3 denoise self (layer 0) + 3 denoise cross (layer 1)
        assert counts["self"] == 5
        assert counts["cross"] == 3
        sal = maps[cam]
        assert sal.shape == (8, 8)
        assert np.isfinite(sal).all()
        assert (sal >= 0).all()
        budget = explainer.last_budget
        assert set(budget) == {cam, "language", "state"}
        assert sum(budget.values()) == pytest.approx(1.0, abs=1e-4)
        assert budget[cam] == pytest.approx(float(sal.sum()), abs=1e-6)

        # Reference: the same tap logic driven by a direct predict_action_chunk on this copy.
        ref = {"acc": None, "events": 0}

        def ref_tap(layer_idx, probs, kind, ref=ref):  # default arg: bind loop variable (B023)
            if kind != "cross":
                return
            row = probs[0, :, 0, :].mean(dim=0).detach().to(torch.float32)
            ref["acc"] = row.clone() if ref["acc"] is None else ref["acc"] + row
            ref["events"] += 1

        policy.model.vlm_with_expert.attention_capture = ref_tap
        with torch.no_grad():
            policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone(), **chunk_kwargs)
        policy.model.vlm_with_expert.attention_capture = counting_tap
        ref_map = (ref["acc"] / ref["events"]).cpu().numpy()[:64].reshape(8, 8)
        assert np.abs(ref_map - sal).max() < 1e-5  # replay == executed chunk's attention

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


class _FlakyExplainer:
    """Fails the first ``fail_times`` calls, then behaves like _FixedExplainer."""

    def __init__(self, fail_times: int):
        self.calls = 0
        self._fail_times = fail_times

    def compute_saliency(self, batch, noise, chunk_kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("boom")
        h, w = batch[CAM_KEY].shape[-2:]
        return {CAM_KEY: np.full((h, w), 0.5, dtype=np.float32)}


def _run_recorder_snapshots(recorder, snapshots):
    recorder.start()
    for snap in snapshots:
        recorder.submit(snap)
        time.sleep(0.1)  # let the recorder thread consume each snapshot before the next
    recorder.close()


def test_recorder_survives_single_frame_failure(tmp_path):
    """One explainer failure drops that frame; recording continues (single failure ≠ disable)."""
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    explainer = _FlakyExplainer(fail_times=1)
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
    _run_recorder_snapshots(recorder, [snap, snap])

    assert explainer.calls == 2
    assert (out / "cam1.mp4").exists()  # the second snapshot still produced a video


def test_recorder_disables_after_consecutive_failures(tmp_path, caplog):
    """Two consecutive processing failures disable recording (and close() must not hang)."""
    import logging as _logging

    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    explainer = _FlakyExplainer(fail_times=10)
    recorder = SaliencyRecorder(
        explainer=explainer,
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    with caplog.at_level(_logging.ERROR):
        _run_recorder_snapshots(recorder, [snap, snap, snap])

    assert explainer.calls == 2  # the third snapshot was never processed
    assert any("Disabling saliency recording" in r.message for r in caplog.records)
    assert not recorder._thread.is_alive()


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
        device=None,
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        method="grad_x_input",
    )
    policy_config = SimpleNamespace(type="definitely_not_registered")

    with pytest.raises(ValueError, match="No saliency explainer registered"):
        SaliencyManager(cfg, policy_config, policy_device="cpu", display_mode="rerun", load_policy=_load)


def test_saliency_manager_swaps_letterbox_wh_for_attention(monkeypatch, tmp_path):
    """Config stores resize_imgs_with_padding as (width, height) (modeling_smolvla.py passes
    [1]/[0] to resize_with_pad); the recorder's overlay math wants (h, w). The manager is the
    single seam where the conventions meet."""
    import lerobot.rollout.saliency.manager as mgr

    captured = {}

    class _FakeExplainer:
        def __init__(self, policy):
            pass

    class _FakeRecorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(mgr, "get_explainer_cls", lambda *a, **k: _FakeExplainer)
    monkeypatch.setattr(mgr, "SaliencyRecorder", _FakeRecorder)

    cfg = SimpleNamespace(
        device=None,
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        method="attention",
        norm_scope="run",
    )
    policy_config = SimpleNamespace(type="smolvla", input_features={}, resize_imgs_with_padding=(640, 480))
    mgr.SaliencyManager(
        cfg,
        policy_config,
        policy_device="cpu",
        display_mode="rerun",
        load_policy=lambda: torch.nn.Linear(1, 1),
    )
    assert captured["letterbox_hw"] == (480, 640)  # (w, h) config → (h, w) overlay


def test_saliency_manager_attention_requires_letterboxed_images(monkeypatch, tmp_path):
    """resize_imgs_with_padding=None leaves the patch→pixel mapping undefined: fail fast at
    startup (before loading weights), don't crash on tuple(None) or silently misrender."""
    import lerobot.rollout.saliency.manager as mgr

    monkeypatch.setattr(mgr, "get_explainer_cls", lambda *a, **k: None)

    def _load():
        raise AssertionError("loader must not run: the letterbox check fails first")

    cfg = SimpleNamespace(
        device=None,
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        method="attention",
        norm_scope="run",
    )
    policy_config = SimpleNamespace(type="smolvla", input_features={}, resize_imgs_with_padding=None)
    with pytest.raises(ValueError, match="resize_imgs_with_padding"):
        mgr.SaliencyManager(cfg, policy_config, policy_device="cpu", display_mode="rerun", load_policy=_load)
