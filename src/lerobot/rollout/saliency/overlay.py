# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Overlay rendering: saliency map + raw camera frame → blended uint8 BGR frame."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
import torch


def tensor_to_uint8_bgr(img: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) / (B,T,C,H,W) / (C,H,W) image tensor (float [0,1] or uint8, RGB) → (H,W,3) uint8 BGR."""
    if img.ndim == 5:
        img = img[:, -1]  # queued observations: display the latest frame
    if img.ndim == 4:
        img = img[0]
    arr = img.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # CHW (RGB) → HWC (RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_blur_map(saliency: np.ndarray, target_hw: tuple[int, int], blur_sigma: float) -> np.ndarray:
    """Resize a raw saliency map to the frame size and Gaussian-blur it (no normalization)."""
    if saliency.shape != target_hw:
        saliency = cv2.resize(saliency, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    if blur_sigma and blur_sigma > 0:
        ksize = 2 * round(3 * blur_sigma) + 1
        saliency = cv2.GaussianBlur(saliency, (ksize, ksize), blur_sigma)
    return saliency


def shared_norm_range(maps: Iterable[np.ndarray]) -> tuple[float, float]:
    """One (min, max) spanning several saliency maps, so co-timed camera views share a color scale.

    Without this, per-camera min-max stretches a camera that contributed almost nothing to
    full red anyway — its internal maximum gets normalized to 1 regardless of magnitude.
    """
    ranges = [(float(m.min()), float(m.max())) for m in maps]  # materialize: `maps` may be a generator
    lo = min(lo for lo, _ in ranges)
    hi = max(hi for _, hi in ranges)
    return lo, hi


def normalize_map(m: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Min-max normalization to [0, 1]. Pass ``lo``/``hi`` to normalize against a shared scale."""
    lo = float(m.min()) if lo is None else lo
    hi = float(m.max()) if hi is None else hi
    return (m - lo) / (hi - lo + 1e-8)


def render_overlay(
    img_bgr: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.5,
    blur_sigma: float = 2.0,
    norm_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Blend a min-max-normalized turbo heatmap onto a uint8 BGR frame.

    ``norm_range`` pins the normalization to a shared (min, max) — e.g. spanning all cameras
    of one chunk — instead of this map's own extrema. Pass an already blurred/resized map
    together with ``blur_sigma=0`` when the range was computed over prepared maps.
    """
    saliency = resize_blur_map(saliency, img_bgr.shape[:2], blur_sigma)
    lo, hi = norm_range if norm_range is not None else (None, None)
    norm = normalize_map(saliency, lo, hi)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0.0)
