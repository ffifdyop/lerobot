# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler

pytest.importorskip("datasets")
pytest.importorskip("av")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from accelerate.data_loader import BatchSamplerShard
from accelerate.utils.fsdp_utils import fsdp2_prepare_auto_wrap_policy
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

from examples.training import train_fineart_so101 as recipe
from lerobot.configs.accelerator import _accelerate_parallelism_config
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import factory

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("frames", [1, 8, 9, 17281])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "backend,shard,groups", [("fsdp", 2, 1), ("fsdp", 2, 2), ("fsdp", 4, 2), ("ddp", 1, 2)]
)
def test_epoch_budget_matches_actual_data_parallel_batch_sharding(frames, batch_size, backend, shard, groups):
    args = recipe.parse_args(
        [
            "check",
            "--epochs",
            "5",
            "--batch-size",
            str(batch_size),
            "--distributed-backend",
            backend,
            "--dp-shard",
            str(shard),
            "--dp-replicate",
            str(groups),
        ]
    )
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    recipe.configure_distributed(cfg, args)
    cfg.parallelism.resolve(shard * groups)
    assert cfg.parallelism.dp_world_size == args.num_gpus == shard * groups
    batches = BatchSampler(SequentialSampler(range(frames)), batch_size=batch_size, drop_last=False)
    per_rank = BatchSamplerShard(
        batches, num_processes=cfg.parallelism.dp_world_size, process_index=0, even_batches=True
    )
    steps, steps_per_epoch = recipe.training_schedule(args, frames)
    assert steps_per_epoch == len(per_rank)
    assert steps == 5 * len(per_rank)


@pytest.mark.parametrize("ids", ["0,1,2,3", "GPU-a,GPU-b,GPU-c,GPU-d"])
def test_hsdp_selects_all_four_workers_and_matches_native_mesh(monkeypatch, ids):
    args = recipe.parse_args(["check", "--dp-shard", "2", "--dp-replicate", "2"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ids)
    assert recipe.visible_gpu_ids(args) == ids.split(",")
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    recipe.configure_distributed(cfg, args)
    cfg._validate_distributed()
    native = _accelerate_parallelism_config(cfg.parallelism)
    assert native.dp_shard_size == native.dp_replicate_size == 2
    assert cfg.parallelism.is_sharded and args.num_gpus == 4


@pytest.mark.parametrize("ids", ["0,2", "0,1,2,2", "0,1,2,", ""])
def test_gpu_count_and_duplicates_fail_before_data_loading(monkeypatch, ids):
    args = recipe.parse_args(["check", "--dp-shard", "2", "--dp-replicate", "2"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ids)
    with pytest.raises(ValueError, match="4 distinct GPUs"):
        recipe.visible_gpu_ids(args)


@pytest.mark.parametrize("reshard,cpu_offload", [(True, False), (False, True)])
def test_native_fsdp2_wrap_policy_leaves_joint_decoder_in_root(reshard, cpu_offload):
    args = recipe.parse_args(
        [
            "check",
            "--fsdp-reshard-after-forward",
            str(reshard),
            "--fsdp-cpu-offload",
            str(cpu_offload),
        ]
    )
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    recipe.configure_distributed(cfg, args)
    cfg.parallelism.resolve(2)
    cfg._validate_distributed()
    assert cfg.parallelism.is_sharded
    assert (cfg.parallelism.dp_replicate, cfg.parallelism.dp_shard) == (1, 2)
    plugin = cfg.accelerator.fsdp.build_plugin()
    assert plugin.fsdp_version == 2 and plugin.reshard_after_forward == reshard
    # Exercise Accelerate's actual class lookup with a small real SigLIP layer. Decoder
    # parameters must remain owned by the root; directly accessed children cannot be units.
    encoder = SiglipEncoderLayer(
        SiglipVisionConfig(hidden_size=8, intermediate_size=16, num_attention_heads=2)
    )
    decoder = torch.nn.Linear(8, 8)
    model = torch.nn.ModuleDict({"encoder": encoder, "decoder": decoder})
    wrap = fsdp2_prepare_auto_wrap_policy(plugin, model)
    assert wrap(encoder)
    assert not wrap(model) and not wrap(decoder)
    assert not wrap(encoder.self_attn.q_proj)


def test_explicit_steps_override_epochs_and_smoke_keeps_small_budget():
    args = recipe.parse_args(["train", "--epochs", "5", "--steps", "123"])
    assert recipe.training_schedule(args, 17281) == (123, 2161)
    smoke = recipe.parse_args(["smoke"])
    assert recipe.training_schedule(smoke, 17281)[0] == 20


@pytest.mark.parametrize(
    "flags",
    [
        ["--train-ratio", "0"],
        ["--train-ratio", "1.1"],
        ["--epochs", "nan"],
        ["--batch-size", "0"],
        ["--min-lr", "0.01"],
        ["--gradient-checkpointing", "invalid"],
        ["--lora-r", "0"],
        ["--distributed-backend", "fsdp1"],
        ["--dp-shard", "0"],
        ["--dp-replicate", "0"],
        ["--dp-shard", "1"],
        ["--distributed-backend", "ddp", "--dp-shard", "2"],
    ],
)
def test_invalid_training_settings_fail_before_data_loading(flags):
    with pytest.raises(SystemExit):
        recipe.parse_args(["check", *flags])


@pytest.mark.parametrize("backend", ["fsdp", "ddp"])
def test_native_holdout_controls_step_count_and_fast_fit_scope(monkeypatch, tmp_path, backend):
    class Dataset:
        fps = 30
        root = tmp_path
        lengths = [100, 200, 300, 401, 500, 600, 700, 801]
        features = {
            key: {"dtype": "float32", "shape": (12,), "names": [f"joint_{i}" for i in range(12)]}
            for key in ("action", "observation.state")
        } | {
            f"observation.images.{view}": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            }
            for view in ("top", "left_wrist", "right_wrist")
        }

        def __init__(self, *args, episodes=None, **kwargs):
            self.episodes = episodes
            self.meta = SimpleNamespace(
                features=self.features,
                fps=30,
                has_language_columns=True,
                camera_keys=[key for key in self.features if "images" in key],
                episodes={"tasks": [["insert"]] * 4 + [["remove"]] * 4},
                stats={"action": {"sentinel": "source statistics"}},
            )

        def __len__(self):
            return sum(self.lengths[i] for i in (self.episodes if self.episodes is not None else range(8)))

        @property
        def num_episodes(self):
            return len(self.episodes) if self.episodes is not None else 8

    monkeypatch.setattr(recipe, "LeRobotDataset", Dataset)
    monkeypatch.setattr(factory, "make_dataset", lambda cfg: Dataset())
    monkeypatch.setattr(factory, "LeRobotDataset", Dataset)
    policy = recipe.FineARTVLAConfig(device="cpu", push_to_hub=False, fast_skip_tokens=128)
    monkeypatch.setattr(recipe.FineARTVLAConfig, "from_pretrained", lambda *args, **kwargs: policy)
    args = recipe.parse_args(
        [
            "check",
            "--distributed-backend",
            backend,
            "--dataset-root",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--train-ratio",
            "0.75",
            "--batch-size",
            "4",
            "--lr",
            "2e-5",
            "--eval-freq",
            "250",
            "--save-freq",
            "500",
            "--warmup-steps",
            "50",
        ]
    )
    cfg, full, train, evaluation = recipe.build_config(args, tmp_path)
    cfg.parallelism.resolve(2)
    assert cfg.parallelism.is_sharded == (backend == "fsdp")
    assert cfg.parallelism.dp_world_size == 2
    assert train.episodes == [0, 1, 2, 4, 5, 6]
    assert evaluation.episodes == [3, 7]
    assert set(train.episodes).isdisjoint(evaluation.episodes)
    assert len(train) + len(evaluation) == len(full)
    assert len(train) == 2400
    assert cfg.steps == 1500  # 5 * ceil(2400 / 8), rather than using the full dataset.
    assert cfg.policy.scheduler_decay_steps == cfg.steps
    assert cfg.policy.optimizer_lr == 2e-5
    assert cfg.policy.scheduler_warmup_steps == 50
    assert cfg.eval_steps == 250 and cfg.save_freq == 500
    assert cfg.dataset.eval_split == 0.25 and cfg.wandb_logging.disable_artifact
    resolver = Mock(return_value=str(tmp_path / "fitted_fast"))
    monkeypatch.setattr(recipe, "resolve_fast_tokenizer", resolver)
    recipe.prepare_fast_tokenizer(cfg, train)
    assert resolver.call_args.kwargs["episodes"] == train.episodes
    assert set(resolver.call_args.kwargs["episodes"]).isdisjoint(evaluation.episodes)
    assert not cfg.policy.auto_fit_fast_tokenizer
    assert cfg.policy.action_tokenizer_name == str(tmp_path / "fitted_fast")


@pytest.mark.parametrize("freeze_vlm", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_finetuning_controls_preserve_native_loss_and_adapter_contract(freeze_vlm, use_lora, checkpointing):
    args = recipe.parse_args(
        [
            "check",
            "--freeze-vlm",
            str(freeze_vlm),
            "--use-lora",
            str(use_lora),
            "--gradient-checkpointing",
            str(checkpointing),
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
        ]
    )
    policy = recipe.FineARTVLAConfig(device="cpu", push_to_hub=False)
    peft = recipe.configure_finetuning(policy, args)
    policy.__post_init__()  # Native FineART rejects frozen VLM + text/FAST supervision.
    assert policy.gradient_checkpointing == checkpointing
    assert policy.train_expert_only == freeze_vlm
    assert policy.enable_fast_action_loss == (not freeze_vlm)
    assert policy.auto_fit_fast_tokenizer == (not freeze_vlm)
    assert policy.text_loss_weight == (0 if freeze_vlm else 1)
    if freeze_vlm:
        assert list(policy.recipe["blend"]) == ["low_level_execution"]
        assert policy.recipe["blend"]["low_level_execution"]["weight"] == 1
    if use_lora:
        assert peft.r == 8 and peft.lora_alpha == 16
        assert re.fullmatch(peft.target_modules, "model.action_out_proj")
        language_layer = (
            "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj"
        )
        assert bool(re.fullmatch(peft.target_modules, language_layer)) == (not freeze_vlm)
        assert not re.fullmatch(
            peft.target_modules,
            "model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.0.self_attn.q_proj",
        )
    else:
        assert peft is None


def test_freeze_vision_only_keeps_text_and_fast_and_frozen_vlm_skips_fast_fit(monkeypatch):
    args = recipe.parse_args(["check", "--freeze-vision-encoder", "true"])
    policy = recipe.FineARTVLAConfig(device="cpu", push_to_hub=False)
    recipe.configure_finetuning(policy, args)
    assert policy.freeze_vision_encoder and not policy.train_expert_only
    assert policy.enable_fast_action_loss and policy.text_loss_weight > 0
    args.freeze_vlm = True
    recipe.configure_finetuning(policy, args)
    resolver = Mock()
    monkeypatch.setattr(recipe, "resolve_fast_tokenizer", resolver)
    recipe.prepare_fast_tokenizer(SimpleNamespace(policy=policy), SimpleNamespace())
    resolver.assert_not_called()


@pytest.mark.parametrize("warmup", [0, 100])
def test_native_lr_schedule_honors_peak_warmup_and_final_rate(warmup):
    args = recipe.parse_args(["check", "--max-lr", "2e-5", "--final-lr", "2e-6"])
    policy = recipe.FineARTVLAConfig(
        device="cpu",
        optimizer_lr=args.lr,
        scheduler_decay_lr=args.min_lr,
        scheduler_warmup_steps=warmup,
        scheduler_decay_steps=1000,
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=args.lr)
    scheduler = policy.get_scheduler_preset().build(optimizer, num_training_steps=1000)
    initial = optimizer.param_groups[0]["lr"]
    assert initial == pytest.approx(args.lr / (warmup + 1))
    rates = [initial]
    for _ in range(1000):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert max(rates) <= args.lr
    if warmup:
        assert rates[warmup - 1] > initial
    assert rates[-1] == pytest.approx(args.min_lr)
