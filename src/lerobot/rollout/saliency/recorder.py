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

from lerobot.utils.rerun_visualization import log_rerun_saliency

from .overlay import render_overlay, resize_blur_map, shared_norm_range, tensor_to_uint8_bgr

logger = logging.getLogger(__name__)

# Per-chunk decay of the running norm max in "run" scope: at ~1.3 chunks/s a one-off spike
# (e.g. an FP blowup in one chunk) stops suppressing the scale after ~1 minute.
_NORM_HI_DECAY = 0.99


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
    ) -> None:
        self._explainer = explainer
        self._output_dir = output_dir
        self._video_fps = video_fps
        self._alpha = alpha
        self._blur_sigma = blur_sigma
        self._display = display
        self._display_mode = display_mode
        self._norm_scope = norm_scope
        self._norm_hi = 0.0  # running max for "run" scope; 0.0 = not yet initialized
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._writers: dict[str, object] = {}
        self._writer_files: dict[str, str] = {}
        self._write_warned: set[str] = set()
        self._nonfinite_warned: set[str] = set()
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
        # Pass 1: validate + convert frames + blur maps (bounds must span the values actually
        # displayed, so blur happens before the shared min-max is taken).
        prepared: list[tuple[str, np.ndarray, np.ndarray]] = []  # (cam_key, frame_bgr, blurred map)
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
        rerun_frames: dict[str, np.ndarray] = {}
        for cam_key, frame, blurred in prepared:
            overlay = render_overlay(frame, blurred, self._alpha, blur_sigma=0.0, norm_range=norm_range)
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
