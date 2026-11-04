"""PI05 coarse FSDP, lazy optimizer resume, legacy keys and gradient accumulation."""

import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from lerobot.configs.default import DatasetConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed import finalize_sharded_policy, set_fsdp_wrap_modules
from lerobot.distributed.checkpoint import (
    load_sharded_model,
    load_sharded_optimizer,
    save_sharded_model,
    save_sharded_optimizer,
)
from lerobot.distributed.parallel_dims import ParallelDims
from lerobot.utils.checkpoint_compat import canonical_pi05_key, canonical_pi05_state_dict
from tests.distributed.test_multigpu_training import (
    _assert_tree_equal,
    _gather_full,
    _init_worker_env,
    _spawn,
)
from tests.fixtures.pi05_fsdp import deterministic_flow, pi05_batch, tiny_pi05


def _native_worker(rank, world_size, port, precision, checkpointing, directory):
    _init_worker_env(rank, world_size, port)
    torch.set_num_threads(1)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "precision": precision,
        "checkpointing": checkpointing,
        "wrap_modules": ["SiglipEncoderLayer"],
        "phases": {},
    }
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/synthetic"),
        parallelism=ParallelismConfig(dp_shard=2),
        batch_size=2,
    )
    cfg.parallelism.resolve(world_size)
    cfg.accelerator.mixed_precision = precision
    cfg.accelerator.fsdp.wrap_modules = ["SiglipEncoderLayer"]
    accelerator = cfg.accelerator.build(cfg.parallelism)
    report["device"] = torch.cuda.get_device_name(accelerator.device)

    def record(phase, fn):
        error = None
        try:
            fn()
        except BaseException:
            error = traceback.format_exc()
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        report["phases"][phase] = {"status": "failed" if any(errors) else "passed", "rank_errors": errors}
        if rank == 0:
            (out / "report.json").write_text(json.dumps(report, indent=2))
            print(
                f"CASE {precision} checkpoint={checkpointing} PHASE {phase}: {report['phases'][phase]['status']}",
                flush=True,
            )
        return not any(errors)

    def prepare():
        policy = tiny_pi05(checkpointing=checkpointing, device=str(accelerator.device))
        deterministic_flow(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        set_fsdp_wrap_modules(accelerator, policy)
        policy, optimizer = accelerator.prepare(policy, optimizer)
        finalize_sharded_policy(policy, ParallelDims.from_config(cfg.parallelism, world_size, "cuda"))
        assert isinstance(policy, FSDPModule)
        assert not hasattr(policy.model.paligemma_with_expert, "joint_layers")
        assert all(isinstance(p, DTensor) for p in policy.parameters())
        siglip = [m for m in policy.modules() if "SiglipEncoderLayer" in type(m).__name__]
        assert len(siglip) == 2 and all(isinstance(m, FSDPModule) for m in siglip)
        return policy, optimizer

    def step(policy, optimizer, batch):
        policy.train()
        with accelerator.autocast():
            loss = policy(batch)[0]
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(loss)
        return loss.detach()

    try:
        torch.manual_seed(7)
        reference = tiny_pi05(checkpointing=checkpointing, device=str(accelerator.device))
        deterministic_flow(reference)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        torch.manual_seed(7)
        policy, optimizer = prepare()
        report["sharded_units"] = [
            name or "<root>" for name, m in policy.named_modules() if isinstance(m, FSDPModule)
        ]
        report["phases"]["prepare"] = {"status": "passed"}

        def train():
            losses = []
            for index in range(2):
                batch = pi05_batch(batch_size=4, seed=20 + index, device=accelerator.device)
                local = {key: value[2 * rank : 2 * rank + 2] for key, value in batch.items()}
                reference.train()
                with accelerator.autocast():
                    ref_loss = reference(batch)[0]
                ref_loss.backward()
                policy.train()
                with accelerator.autocast():
                    loss = policy(local)[0]
                accelerator.backward(loss)
                mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
                tol = {"atol": 1e-5, "rtol": 1e-4} if precision == "no" else {"atol": 1e-2, "rtol": 3e-2}
                torch.testing.assert_close(mean_loss.float(), ref_loss.detach().float(), **tol)
                if precision == "no":
                    gradients = {
                        name: None if p.grad is None else p.grad.full_tensor()
                        for name, p in policy.named_parameters()
                    }
                    for name, p in reference.named_parameters():
                        torch.testing.assert_close(gradients[name], p.grad, atol=1e-5, rtol=2e-3, msg=name)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                reference_optimizer.step()
                optimizer.step()
                reference_optimizer.zero_grad()
                optimizer.zero_grad()
                losses.append({"sharded": mean_loss.item(), "reference": ref_loss.item()})
            report["losses"] = losses
            report["peak_allocated_bytes_tiny_with_reference"] = torch.cuda.max_memory_allocated(
                accelerator.device
            )

        if not record("two_training_steps", train):
            return

        expected_model, expected_optimizer = _gather_full(policy, optimizer)

        def export():
            policy.save_pretrained(out / "export")

        export_ok = record("ordinary_weight_export", export)
        if export_ok:

            def load_plain():
                if rank == 0:
                    plain = tiny_pi05(device="cpu")
                    exported = load_file(out / "export/model.safetensors")
                    plain.load_state_dict(exported, strict=True)
                    torch.testing.assert_close(plain.state_dict(), expected_model, rtol=0, atol=0)
                    assert torch.isfinite(plain.predict_action_chunk(pi05_batch())).all()

            record("ordinary_weight_load_and_inference", load_plain)

        def save_dcp():
            save_sharded_model(accelerator, policy, out / "dcp")
            save_sharded_optimizer(accelerator, optimizer, policy, out / "dcp")

        if not record("dcp_save", save_dcp):
            return
        fresh, fresh_optimizer = prepare()
        if not record("dcp_model_load", lambda: load_sharded_model(accelerator, fresh, out / "dcp")):
            return
        if not record(
            "dcp_optimizer_load",
            lambda: load_sharded_optimizer(accelerator, fresh_optimizer, fresh, out / "dcp"),
        ):
            return
        actual_model, actual_optimizer = _gather_full(fresh, fresh_optimizer)

        def state_parity():
            if rank == 0:
                _assert_tree_equal(expected_model, actual_model, "model")
                _assert_tree_equal(expected_optimizer, actual_optimizer, "optimizer")

        if not record("restored_state_parity", state_parity):
            return

        def continue_step():
            batch = pi05_batch(seed=40 + rank, device=accelerator.device)
            loss_a = step(policy, optimizer, batch)
            loss_b = step(fresh, fresh_optimizer, batch)
            torch.testing.assert_close(loss_a, loss_b, atol=0, rtol=0)
            model_a, optimizer_a = _gather_full(policy, optimizer)
            model_b, optimizer_b = _gather_full(fresh, fresh_optimizer)
            if rank == 0:
                _assert_tree_equal(model_a, model_b, "continued model")
                _assert_tree_equal(optimizer_a, optimizer_b, "continued optimizer")

        record("continued_next_step_parity", continue_step)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def setup(rank, world_size, port, precision, sync_each_batch=False, accumulation=1):
    _init_worker_env(rank, world_size, port)
    torch.set_num_threads(1)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/synthetic"),
        parallelism=ParallelismConfig(dp_shard=2),
        batch_size=2,
    )
    cfg.parallelism.resolve(world_size)
    cfg.accelerator.mixed_precision = precision
    cfg.accelerator.fsdp.wrap_modules = ["SiglipEncoderLayer"]
    cfg.accelerator.gradient_accumulation.steps = accumulation
    cfg.accelerator.gradient_accumulation.sync_each_batch = sync_each_batch
    accelerator = cfg.accelerator.build(cfg.parallelism)
    return cfg, accelerator


def prepare(cfg, accelerator):
    policy = tiny_pi05(checkpointing=True, device=str(accelerator.device))
    deterministic_flow(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    set_fsdp_wrap_modules(accelerator, policy)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    finalize_sharded_policy(policy, ParallelDims.from_config(cfg.parallelism, 2, "cuda"))
    return policy, optimizer


def step(accelerator, policy, optimizer, batch):
    policy.train()
    with accelerator.autocast():
        loss = policy(batch)[0]
    accelerator.backward(loss)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach()


def legacy_worker(rank, world_size, port, precision, directory):
    cfg, accelerator = setup(rank, world_size, port, precision)
    directory = Path(directory)
    try:
        fresh, fresh_optimizer = prepare(cfg, accelerator)
        print(f"legacy rank {rank}: prepared", flush=True)
        load_sharded_model(accelerator, fresh, directory)
        print(f"legacy rank {rank}: model loaded", flush=True)
        load_sharded_optimizer(accelerator, fresh_optimizer, fresh, directory)
        print(f"legacy rank {rank}: optimizer loaded", flush=True)
        reference, reference_optimizer = prepare(cfg, accelerator)
        print(f"legacy rank {rank}: reference prepared", flush=True)
        model_state = canonical_pi05_state_dict(load_file(directory / "legacy_joint.safetensors"))
        optimizer_state = torch.load(directory / "optimizer_expected.pt", weights_only=True)
        optimizer_state["state"] = canonical_pi05_state_dict(optimizer_state["state"])
        for group in optimizer_state["param_groups"]:
            group["params"] = [canonical_pi05_key(name) for name in group["params"]]
        options = StateDictOptions(full_state_dict=True, strict=False)
        set_model_state_dict(reference, model_state, options=StateDictOptions(full_state_dict=True))
        set_optimizer_state_dict(reference, reference_optimizer.optimizer, optimizer_state, options=options)
        # Torch's full-state helper materializes zero moments for absent entries;
        # remove those dummy entries in the independent reference only.
        saved_params = {p for name, p in reference.named_parameters() if name in optimizer_state["state"]}
        for p in list(reference_optimizer.optimizer.state):
            if p not in saved_params:
                del reference_optimizer.optimizer.state[p]
        # The full-state helper also flattens tuples (e.g. Adam betas) to lists.
        # Keep the independent reference's saved optimizer settings verbatim.
        for current_group, saved_group in zip(
            reference_optimizer.param_groups, optimizer_state["param_groups"], strict=True
        ):
            current_group.update({key: value for key, value in saved_group.items() if key != "params"})
        print(f"legacy rank {rank}: reference state set", flush=True)
        fresh_model, fresh_state = _gather_full(fresh, fresh_optimizer)
        ref_model, ref_state = _gather_full(reference, reference_optimizer)
        print(f"legacy rank {rank}: states gathered", flush=True)
        errors = [None]
        if rank == 0:
            try:
                _assert_tree_equal(ref_model, fresh_model, "legacy restored model")
                _assert_tree_equal(ref_state, fresh_state, "legacy restored optimizer")
            except AssertionError as error:
                errors[0] = str(error)
        dist.broadcast_object_list(errors, src=0)
        assert errors[0] is None, errors[0]
        batch = pi05_batch(seed=77 + rank, device=accelerator.device)
        a = step(accelerator, reference, reference_optimizer, batch)
        b = step(accelerator, fresh, fresh_optimizer, batch)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        fresh_model, fresh_state = _gather_full(fresh, fresh_optimizer)
        ref_model, ref_state = _gather_full(reference, reference_optimizer)
        if rank == 0:
            _assert_tree_equal(ref_model, fresh_model, "legacy continued model")
            _assert_tree_equal(ref_state, fresh_state, "legacy continued optimizer")
            (Path(directory) / f"legacy_{precision}.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "precision": precision,
                        "restored_state": "exact",
                        "next_step_state": "exact",
                        "source": "legacy checkpoint key schema fixture",
                    },
                    indent=2,
                )
            )
    except BaseException:
        print(f"rank {rank} legacy failure:\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        dist.destroy_process_group()


def accumulation_worker(rank, world_size, port, precision, sync_each_batch, directory):
    cfg, accelerator = setup(rank, world_size, port, precision, sync_each_batch, accumulation=2)
    try:
        torch.manual_seed(7)
        reference = tiny_pi05(checkpointing=True, device=str(accelerator.device))
        deterministic_flow(reference)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        torch.manual_seed(7)
        policy, optimizer = prepare(cfg, accelerator)
        boundaries = []
        losses = []
        for update in range(2):
            batch = pi05_batch(batch_size=8, seed=37 + update, device=accelerator.device)
            reference.train()
            with accelerator.autocast():
                ref_loss = reference(batch)[0]
            ref_loss.backward()
            local_loss = torch.zeros((), device=accelerator.device)
            for micro in range(2):
                start = rank * 4 + micro * 2
                local = {name: value[start : start + 2] for name, value in batch.items()}
                with accelerator.accumulate(policy):
                    with accelerator.autocast():
                        loss = policy(local)[0]
                    local_loss += loss.detach() / 2
                    accelerator.backward(loss)
                    boundaries.append(accelerator.sync_gradients)
                    if accelerator.sync_gradients:
                        if precision == "no":
                            gradients = {
                                name: None if p.grad is None else p.grad.full_tensor()
                                for name, p in policy.named_parameters()
                            }
                            for name, p in reference.named_parameters():
                                torch.testing.assert_close(
                                    gradients[name], p.grad, atol=2e-5, rtol=3e-3, msg=name
                                )
                        accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            mean_loss = accelerator.reduce(local_loss, reduction="mean")
            tolerance = {"atol": 1e-5, "rtol": 1e-4} if precision == "no" else {"atol": 1e-2, "rtol": 3e-2}
            torch.testing.assert_close(mean_loss.float(), ref_loss.detach().float(), **tolerance)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            losses.append({"accumulated": mean_loss.item(), "reference": ref_loss.item()})
        assert boundaries == [False, True, False, True]
        final_model, final_optimizer = _gather_full(policy, optimizer)
        if rank == 0:
            assert all(state["step"].item() == 2 for state in final_optimizer["state"].values())
            if precision == "no":
                # Near-zero softmax key-bias gradients amplify FP32 summation
                # differences through Adam's epsilon. Gradients above are checked
                # before the optimizer, and this bounds the resulting update.
                torch.testing.assert_close(
                    final_model,
                    {name: p.cpu() for name, p in reference.state_dict().items()},
                    atol=1e-4,
                    rtol=3e-3,
                )
            (Path(directory) / f"accumulation_{precision}_{sync_each_batch}.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "precision": precision,
                        "sync_each_batch": sync_each_batch,
                        "optimizer_steps": 2,
                        "microbatches": 4,
                        "boundaries": boundaries,
                        "losses": losses,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize(
    "precision,checkpointing", [("no", False), ("no", True), ("bf16", False), ("bf16", True)]
)
def test_pi05_fsdp_training_export_and_resume(tmp_path, precision, checkpointing):
    _spawn(2, _native_worker, precision, checkpointing, str(tmp_path), timeout_s=150)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["phases"]["continued_next_step_parity"]["status"] == "passed", report
    assert all(phase["status"] == "passed" for phase in report["phases"].values()), report


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("precision", ["no", "bf16"])
@pytest.mark.parametrize("sync_each_batch", [False, True])
def test_pi05_fsdp_gradient_accumulation(tmp_path, precision, sync_each_batch):
    _spawn(2, accumulation_worker, precision, sync_each_batch, str(tmp_path), timeout_s=150)
    assert (
        json.loads((tmp_path / f"accumulation_{precision}_{sync_each_batch}.json").read_text())["status"]
        == "passed"
    )


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_pi05_fsdp_legacy_checkpoint_resume(tmp_path, precision):
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    from tests.distributed.test_pi05_checkpoint_migration import _legacy_key

    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = tiny_pi05(checkpointing=True)
    deterministic_flow(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(pi05_batch())[0].backward()
    optimizer.step()
    optimizer.zero_grad()
    weights = {_legacy_key(name): tensor for name, tensor in model.state_dict().items()}
    state = get_optimizer_state_dict(model, optimizer)
    state["state"] = {_legacy_key(name): value for name, value in state["state"].items()}
    for group in state["param_groups"]:
        group["params"] = [_legacy_key(name) for name in group["params"]]
    dcp.save({"model": weights}, checkpoint_id=tmp_path / "pytorch_model_fsdp_0")
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path / "optimizer_0")
    save_file(weights, tmp_path / "legacy_joint.safetensors")
    torch.save(state, tmp_path / "optimizer_expected.pt")
    _spawn(2, legacy_worker, precision, str(tmp_path), timeout_s=150)
    assert json.loads((tmp_path / f"legacy_{precision}.json").read_text())["status"] == "passed"
