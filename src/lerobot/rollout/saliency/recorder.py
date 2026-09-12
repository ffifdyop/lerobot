# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SaliencyRecorder: side thread that turns chunk snapshots into overlay frames, videos, and rerun entities.

Never blocks the control loop: the queue holds at most one snapshot (latest wins),
frames that arrive during a slow backward pass are dropped — the mp4s are diagnostic
tools, not measurement instruments (see CONTEXT.md: Saliency Overlay Video).
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading

import cv2
import numpy as np

from lerobot.utils.rerun_visualization import log_rerun_saliency, log_rerun_saliency_budget

from .overlay import (
    draw_budget_bar,
    render_overlay,
    render_patch_overlay,
    resize_blur_map,
    shared_norm_range,
    tensor_to_uint8_bgr,
)

logger = logging.getLogger(__name__)

# Per-chunk decay of the running norm max in "run" scope: at ~1.3 chunks/s a one-off spike
# (e.g. an FP blowup in one chunk) stops suppressing the scale after ~1 minute.
_NORM_HI_DECAY = 0.99


def _budget_segments(budget: dict[str, float]) -> list[tuple[str, float]]:
    """Order a five-way attention budget for the stacked bar: cameras first (insertion order,
    short display names), then language, then state."""
    segments = [
        (key.rsplit(".", 1)[-1], share) for key, share in budget.items() if key not in ("language", "state")
    ]
    if "language" in budget:
        segments.append(("language", budget["language"]))
    if "state" in budget:
        segments.append(("state", budget["state"]))
    return segments


class SaliencyRecorder:
    """Side-thread pipeline: snapshot → saliency maps → overlay frame → mp4 + rerun."""

    def __init__(
        self,
        explainer,
        output_dir: str,
        video_fps: float,
        alpha: float,
        blur_sigma: float,
        display: bool,
        display_mode: str,
        norm_scope: str = "run",
        letterbox_hw: tuple[int, int] = (512, 512),
    ) -> None:
        self._explainer = explainer
        self._map_kind = getattr(explainer, "map_kind", "pixel")
        self._output_dir = output_dir
        self._video_fps = video_fps
        self._alpha = alpha
        self._blur_sigma = blur_sigma
        self._display = display
        self._display_mode = display_mode
        self._norm_scope = norm_scope
        self._letterbox_hw = letterbox_hw
        self._norm_hi = 0.0  # running max for "run" scope; 0.0 = not yet initialized
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._writers: dict[str, object] = {}
        self._writer_files: dict[str, str] = {}
        self._write_warned: set[str] = set()
        self._nonfinite_warned: set[str] = set()
        self._budget_nonfinite_warned = False
        self._closed = False
        os.makedirs(output_dir, exist_ok=True)

    def submit(self, snapshot: dict | None) -> None:
        """Non-blocking, drop-oldest push. ``None`` is the shutdown sentinel."""
        if self._closed and snapshot is not None:
            return  # closed: drop real snapshots; the ``None`` sentinel must still get through
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(snapshot)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="saliency-recorder", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the thread and flush the videos. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.submit(None)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                logger.warning("Saliency recorder thread did not exit within 30s; videos may be incomplete")

    def _run(self) -> None:
        consecutive_failures = 0
        try:
            while True:
                # Poll with a timeout instead of blocking forever: a racing submit can steal the
                # shutdown sentinel out of the cap-1 queue (drop-oldest), which would otherwise
                # strand this thread and stall close() for its full 30s join.
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._closed:
                        break
                    continue
                if item is None:
                    break
                try:
                    self._process(item)
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    logger.exception("Saliency frame dropped due to processing error")
                    if consecutive_failures >= 2:
                        # Every frame failing points at broken infrastructure (explainer, codec),
                        # not transient load: stop rather than silently producing an empty video.
                        logger.error("Disabling saliency recording after repeated frame failures")
                        break
        finally:
            # Writers are released here, on the recorder thread, so the main thread
            # never releases a writer the thread might still be writing to.
            for writer in self._writers.values():
                writer.release()

    def _process(self, item: dict) -> None:
        maps = self._explainer.compute_saliency(item["preprocessed"], item["noise"], item["chunk_kwargs"])
        # Patch explainers also expose a five-way attention budget (same thread — no race).
        budget = getattr(self._explainer, "last_budget", None) if self._map_kind == "patch" else None
        if budget is not None and not all(np.isfinite(v) for v in budget.values()):
            # A budget NaN (FP blowup in language/state columns) would raise inside
            # draw_budget_bar and cost the whole frame; drop just the bar/curves instead,
            # mirroring the per-camera NaN guard below.
            if not self._budget_nonfinite_warned:
                self._budget_nonfinite_warned = True
                logger.warning(
                    "Non-finite attention budget; dropping the budget bar/curves for affected frames"
                )
            budget = None
        # Pass 1: validate + convert frames + prepare maps for display. Pixel maps are blurred
        # before the shared min-max so bounds span the values actually displayed; patch maps
        # stay blocky (never blurred) and are normalized as-is.
        prepared: list[tuple[str, np.ndarray, np.ndarray]] = []  # (cam_key, frame_bgr, display map)
        for cam_key, sal in maps.items():
            if not np.isfinite(sal).all():
                # NaN/Inf gradients would flow silently through min-max normalization into
                # plausible-looking garbage: drop the frame and surface the FP blowup once.
                if cam_key not in self._nonfinite_warned:
                    self._nonfinite_warned.add(cam_key)
                    logger.warning(
                        "Non-finite saliency values for %s; dropping its frames (check for FP overflow)",
                        cam_key,
                    )
                continue
            frame = tensor_to_uint8_bgr(item["preprocessed"][cam_key])
            if self._map_kind == "patch":
                prepared.append((cam_key, frame, sal.astype(np.float32)))
            else:
                prepared.append((cam_key, frame, resize_blur_map(sal, frame.shape[:2], self._blur_sigma)))
        if not prepared:
            return
        if self._norm_scope == "run":
            # One online scale for the whole run: lo=0 (maps are non-negative by construction),
            # hi = decaying running max. Color depth ≈ absolute contribution, comparable across
            # cameras AND time — a quiet moment stays dark instead of being stretched to red.
            chunk_hi = max(float(m.max()) for _, _, m in prepared)
            self._norm_hi = max(chunk_hi, self._norm_hi * _NORM_HI_DECAY)
            norm_range = (0.0, self._norm_hi)
        else:
            # "chunk" scope: all cameras of one chunk share one color scale (per-chunk min-max);
            # a camera that contributed little stays dark instead of being stretched to full
            # red by a private per-camera min-max.
            norm_range = shared_norm_range(m for _, _, m in prepared)
        budget_segments = _budget_segments(budget) if budget is not None else None
        rerun_frames: dict[str, np.ndarray] = {}
        for cam_key, frame, display_map in prepared:
            if self._map_kind == "patch":
                overlay = render_patch_overlay(
                    frame, display_map, self._alpha, norm_range=norm_range, letterbox_hw=self._letterbox_hw
                )
                if budget_segments is not None:
                    overlay = draw_budget_bar(overlay, budget_segments)
            else:
                overlay = render_overlay(
                    frame, display_map, self._alpha, blur_sigma=0.0, norm_range=norm_range
                )
            # Some OpenCV builds return a bool from write(), None on others; a False means the
            # frame was silently dropped (e.g. resolution changed mid-video → frozen mp4).
            ret = self._writer(cam_key, overlay.shape[1], overlay.shape[0]).write(overlay)
            if ret is False and cam_key not in self._write_warned:
                self._write_warned.add(cam_key)
                logger.warning("Video writer rejected the frame for %s (resolution changed?)", cam_key)
            if self._display and self._display_mode == "rerun":
                rerun_frames[cam_key.rsplit(".", 1)[-1]] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        if rerun_frames:
            log_rerun_saliency(rerun_frames)
        if self._display and self._display_mode == "rerun" and budget is not None:
            log_rerun_saliency_budget({k.rsplit(".", 1)[-1]: v for k, v in budget.items()})

    def _writer(self, cam_key: str, width: int, height: int):
        if cam_key in self._writers:
            return self._writers[cam_key]
        path = os.path.join(self._output_dir, f"{cam_key.rsplit('.', 1)[-1]}.mp4")
        collide = next((k for k, p in self._writer_files.items() if p == path), None)
        if collide is not None:
            # Two camera keys sharing one file would interleave frames into a corrupt mp4.
            raise RuntimeError(f"Camera keys {collide!r} and {cam_key!r} both map to {path}")
        for fourcc in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), self._video_fps, (width, height))
            if writer.isOpened():
                self._writers[cam_key] = writer
                self._writer_files[cam_key] = path
                return writer
            writer.release()
        raise RuntimeError(f"Could not open video writer for {path}")
