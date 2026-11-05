# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Optional W&B backend for the native training logger interface."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed import is_main_process
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import _wandb_available, require_package

if TYPE_CHECKING or _wandb_available:
    import wandb


class WandBLogger:
    """Log reduced native metrics on rank 0 and persist the run ID in checkpoints."""

    def __init__(self, cfg: TrainPipelineConfig) -> None:
        if not is_main_process():
            raise RuntimeError("Only rank 0 may initialize W&B")
        require_package("wandb", "wandb")
        if cfg.output_dir is None:
            raise ValueError("Resolve output_dir before initializing W&B")
        self.cfg = cfg.wandb_logging
        self.cfg.run_id = self.cfg.run_id or uuid.uuid4().hex[:12]
        self.cfg.run_name = self.cfg.run_name or cfg.job_name
        log_directory = cfg.output_dir / "wandb_logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.run_name,
            id=self.cfg.run_id,
            mode=self.cfg.mode,
            resume=("allow" if cfg.resume else "never") if self.cfg.mode == "online" else None,
            dir=str(log_directory),
            config=cfg.to_dict(),
        )
        self._run.define_metric("global_step")
        self._run.define_metric("*", step_metric="global_step")
        logging.info("W&B run: %s (mode=%s)", self._run.url or self.cfg.run_id, self.cfg.mode)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided")
        if custom_step_key is not None and custom_step_key not in d:
            raise ValueError(f"custom_step_key {custom_step_key!r} is missing")
        metrics = {f"{mode}/{key}": value for key, value in d.items() if isinstance(value, (int, float, str))}
        if not metrics:
            return
        if custom_step_key is None:
            assert step is not None
            metrics["global_step"] = step
            self._run.log(metrics, step=step)
        else:
            self._run.log(metrics)

    def log_policy(self, checkpoint_dir: Path) -> None:
        if self.cfg.disable_artifact:
            return
        try:
            artifact = wandb.Artifact(f"model-{self.cfg.run_id}-{checkpoint_dir.name}", type="model")
            artifact.add_dir(str(checkpoint_dir / PRETRAINED_MODEL_DIR))
            self._run.log_artifact(artifact)
        except Exception as error:  # noqa: BLE001
            logging.warning("Skipping W&B checkpoint artifact upload: %s", error)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        self._run.log({f"{mode}/video": wandb.Video(video_path), "global_step": step}, step=step)

    def finish(self) -> None:
        self._run.finish()
