# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import os
import signal
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train


@pytest.mark.parametrize("exit_kind", ["normal", "error", "interrupt"])
def test_train_owns_cleanup_around_body(monkeypatch, exit_kind):
    calls = []
    accelerator = SimpleNamespace(device=torch.device("cpu"), end_training=lambda: calls.append("end"))
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/synthetic"))
    monkeypatch.setattr(cfg, "validate", lambda: None)
    monkeypatch.setattr(lerobot_train, "make_accelerator", lambda cfg: accelerator)

    def body(cfg, accelerator, shutdown):
        calls.append("train")
        if exit_kind == "error":
            raise RuntimeError("training failed")
        if exit_kind == "interrupt":
            os.kill(os.getpid(), signal.SIGINT)
            shutdown.raise_if_requested(accelerator.device)

    monkeypatch.setattr(lerobot_train, "_train", body)
    if exit_kind == "error":
        with pytest.raises(RuntimeError, match="training failed"):
            lerobot_train.train(cfg)
    elif exit_kind == "interrupt":
        with pytest.raises(SystemExit) as exc:
            lerobot_train.train(cfg)
        assert exc.value.code == 130
    else:
        lerobot_train.train(cfg)
    assert calls == ["train", "end"]
