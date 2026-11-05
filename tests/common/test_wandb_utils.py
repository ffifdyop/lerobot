# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import draccus
import pytest

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import DatasetConfig, TrackioConfig, WandBLoggingConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@pytest.fixture
def wandb(monkeypatch):
    module = MagicMock()
    monkeypatch.setattr("lerobot.common.wandb_utils.wandb", module, raising=False)
    monkeypatch.setattr("lerobot.common.wandb_utils.require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr("lerobot.common.wandb_utils.is_main_process", lambda: True)
    return module


def make_cfg(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "wandb_logging": WandBLoggingConfig(enable=True, project="so101", entity="team"),
        "output_dir": tmp_path,
        "job_name": "test-run",
        "resume": False,
        "to_dict": lambda: {"test": True},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_wandb_logs_native_losses_and_step_and_finishes(tmp_path, wandb):
    cfg = make_cfg(tmp_path)
    logger = WandBLogger(cfg)
    init = wandb.init.call_args.kwargs
    assert init["project"] == "so101" and init["entity"] == "team"
    assert init["id"] == cfg.wandb_logging.run_id
    assert init["resume"] == "never"
    logger.log_dict({"loss": 3.0, "loss_text": 0.4, "loss_fast": 0.6, "loss_flow": 0.2}, step=20)
    wandb.init.return_value.log.assert_called_once_with(
        {
            "train/loss": 3.0,
            "train/loss_text": 0.4,
            "train/loss_fast": 0.6,
            "train/loss_flow": 0.2,
            "global_step": 20,
        },
        step=20,
    )
    logger.finish()
    wandb.init.return_value.finish.assert_called_once()


def test_wandb_resume_keeps_saved_run_id(tmp_path, wandb):
    cfg = make_cfg(tmp_path, resume=True)
    cfg.wandb_logging.run_id = "saved-run"
    WandBLogger(cfg)
    assert wandb.init.call_args.kwargs["id"] == "saved-run"
    assert wandb.init.call_args.kwargs["resume"] == "allow"


def test_wandb_offline_does_not_claim_server_resume(tmp_path, wandb):
    cfg = make_cfg(tmp_path, resume=True)
    cfg.wandb_logging.mode = "offline"
    WandBLogger(cfg)
    assert wandb.init.call_args.kwargs["mode"] == "offline"
    assert wandb.init.call_args.kwargs["resume"] is None


def test_wandb_only_rank_zero_can_create_run(tmp_path, wandb, monkeypatch):
    monkeypatch.setattr("lerobot.common.wandb_utils.is_main_process", lambda: False)
    with pytest.raises(RuntimeError, match="rank 0"):
        WandBLogger(make_cfg(tmp_path))
    wandb.init.assert_not_called()


def test_wandb_checkpoint_upload_is_opt_in(tmp_path, wandb):
    logger = WandBLogger(make_cfg(tmp_path))
    logger.log_policy(tmp_path / "checkpoint")
    wandb.Artifact.assert_not_called()


def test_wandb_logging_config_survives_native_serialization(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/data"),
        policy=ACTConfig(device="cpu", push_to_hub=False),
        wandb_logging=WandBLoggingConfig(enable=True, project="so101", run_id="saved-run", mode="offline"),
        output_dir=tmp_path / "train",
    )
    encoded = draccus.encode(cfg)
    decoded = draccus.decode(TrainPipelineConfig, encoded)
    decoded.validate()
    assert decoded.wandb_logging == cfg.wandb_logging
    assert not decoded.trackio.enable


def test_train_rejects_two_logging_backends(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/data"),
        policy=ACTConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "train",
        trackio=TrackioConfig(enable=True),
        wandb_logging=WandBLoggingConfig(enable=True),
    )
    with pytest.raises(ValueError, match="only one training logger"):
        cfg.validate()


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_wandb_config_accepts_modes(mode):
    assert WandBLoggingConfig(mode=mode).mode == mode


def test_wandb_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="wandb_logging.mode"):
        WandBLoggingConfig(mode="auto")
