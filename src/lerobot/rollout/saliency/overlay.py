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


# ---------------------------------------------------------------------------
# Patch-grid overlay (method="attention": Patch Attention Map + Budget Bar)
# ---------------------------------------------------------------------------


def patch_content_mapping(
    frame_hw: tuple[int, int], letterbox_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Mirror ``resize_with_pad`` (policies/common/vla_utils.py, LEFT/TOP padding).

    Returns (pad_h, pad_w, resized_h, resized_w) in letterbox space for a camera frame of
    ``frame_hw`` letterboxed to ``letterbox_hw``. The displayed frame is exactly the content
    region of the letterboxed image, scaled by frame/resized.
    """
    fh, fw = frame_hw
    lh, lw = letterbox_hw
    ratio = max(fw / lw, fh / lh)
    resized_h, resized_w = int(fh / ratio), int(fw / ratio)
    return lh - resized_h, lw - resized_w, resized_h, resized_w


def render_patch_overlay(
    img_bgr: np.ndarray,
    patch_map: np.ndarray,
    alpha: float = 0.5,
    norm_range: tuple[float, float] | None = None,
    letterbox_hw: tuple[int, int] = (512, 512),
    grid_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Blocky (nearest-neighbor) overlay of a (G, G) patch-token map with true grid lines.

    The token grid spans the whole letterboxed image INCLUDING padding; the displayed frame
    is the content region only. Each patch's value is drawn solely over its intersection
    with the content region — pure-padding patches never render (thin partial-patch bands
    at the edges are correct behavior), and the grid is never resampled to fit the frame.
    No bilinear blur anywhere: the heat is piecewise-constant per patch by construction.
    """
    gh, gw = patch_map.shape
    fh, fw = img_bgr.shape[:2]
    lh, lw = letterbox_hw
    pad_h, pad_w, resized_h, resized_w = patch_content_mapping((fh, fw), letterbox_hw)

    # Blocky heat in letterbox space → crop the content region → scale to the frame. Both
    # resizes are INTER_NEAREST, so every patch stays a solid block; partial edge patches
    # land at their true (fractional-cell) position via the crop.
    heat_grid = cv2.resize(patch_map, (lw, lh), interpolation=cv2.INTER_NEAREST)
    content = heat_grid[pad_h:, pad_w:]
    heat = cv2.resize(content, (fw, fh), interpolation=cv2.INTER_NEAREST)

    lo, hi = norm_range if norm_range is not None else (None, None)
    norm = normalize_map(heat, lo, hi)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, color, alpha, 0.0)

    # Grid lines at TRUE token boundaries: letterbox edge r*lh/gh, shifted into content
    # space (-pad), scaled to frame coordinates; edges outside the content region are skipped.
    # LINE_8, not LINE_AA: anti-aliasing bleeds a thickness-1 line across two pixels.
    for r in range(1, gh):
        y = round((r * lh / gh - pad_h) * fh / resized_h)
        if 0 <= y < fh:
            cv2.line(blended, (0, y), (fw - 1, y), grid_color, 1, cv2.LINE_8)
    for c in range(1, gw):
        x = round((c * lw / gw - pad_w) * fw / resized_w)
        if 0 <= x < fw:
            cv2.line(blended, (x, 0), (x, fh - 1), grid_color, 1, cv2.LINE_8)
    return blended


# BGR colors for budget segments: cameras cycle the palette in draw order; language and
# state always get their fixed neutrals (a gray "language" segment means the same thing
# regardless of how many cameras the robot has).
_BUDGET_CAM_PALETTE: list[tuple[int, int, int]] = [
    (39, 127, 242),  # orange
    (87, 201, 87),  # green
    (242, 196, 39),  # cyan
]
_BUDGET_SPECIAL_COLORS: dict[str, tuple[int, int, int]] = {
    "language": (148, 148, 148),
    "state": (72, 72, 72),
}


def draw_budget_bar(
    img_bgr: np.ndarray,
    segments: list[tuple[str, float]],
    bar_height: int = 24,
) -> np.ndarray:
    """Burn a stacked attention-budget bar into the top strip of the frame.

    ``segments`` is [(label, share)] in draw order (cameras, then language, then state);
    shares are expected to sum to ~1 — the last segment absorbs the rounding remainder so
    the bar always spans the full frame width. Occludes the top ``bar_height`` rows by
    design (the video must be self-contained: readable without rerun).
    """
    out = img_bgr.copy()
    fh, fw = out.shape[:2]
    bh = min(bar_height, fh)
    x = 0
    cam_idx = 0
    for i, (label, share) in enumerate(segments):
        remaining = fw - x
        seg_w = remaining if i == len(segments) - 1 else max(0, min(round(share * fw), remaining))
        if label in _BUDGET_SPECIAL_COLORS:
            color = _BUDGET_SPECIAL_COLORS[label]
        else:
            color = _BUDGET_CAM_PALETTE[cam_idx % len(_BUDGET_CAM_PALETTE)]
            cam_idx += 1
        if seg_w > 0:
            cv2.rectangle(out, (x, 0), (x + seg_w - 1, bh - 1), color, thickness=-1)
            text = f"{label} {share * 100:.0f}%"
            if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] + 6 <= seg_w:
                # dark outline + light fill stays legible on every palette color
                cv2.putText(
                    out, text, (x + 3, bh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    out, text, (x + 3, bh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )
        x += seg_w
    if bh < fh:
        cv2.line(out, (0, bh - 1), (fw - 1, bh - 1), (32, 32, 32), 1)
    return out
