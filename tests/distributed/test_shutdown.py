# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import multiprocessing
import os
import signal
from types import SimpleNamespace

import pytest
import torch

import lerobot.distributed.shutdown as shutdown_module
from lerobot.distributed.shutdown import TrainingInterrupted, TrainingShutdown, init_training_data_worker


@pytest.fixture
def fake_group(monkeypatch):
    state = SimpleNamespace(initialized=True, destroys=0)

    def destroy():
        state.destroys += 1
        state.initialized = False

    monkeypatch.setattr(shutdown_module.dist, "is_initialized", lambda: state.initialized)
    monkeypatch.setattr(shutdown_module.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(shutdown_module.dist, "all_reduce", lambda *args, **kwargs: None)
    return state


@pytest.mark.parametrize("failure", [None, RuntimeError("training failed"), KeyboardInterrupt()])
def test_cleanup_and_handlers_restored_on_every_exit(fake_group, failure):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    calls = []
    try:
        with TrainingShutdown() as shutdown:
            shutdown.accelerator = SimpleNamespace(end_training=lambda: calls.append("end"))
            if failure is not None:
                raise failure
    except BaseException as exc:
        assert exc is failure
    assert calls == ["end"]
    assert fake_group.destroys == 1
    assert not fake_group.initialized
    assert {sig: signal.getsignal(sig) for sig in original} == original


def test_failed_accelerator_construction_still_destroys_group(fake_group):
    with pytest.raises(RuntimeError, match="construction"), TrainingShutdown():
        raise RuntimeError("construction failed after process group initialization")
    assert fake_group.destroys == 1


def test_tracker_failure_does_not_skip_group_cleanup_or_mask_body_error(fake_group):
    def failing_end():
        raise RuntimeError("tracker failed")

    with pytest.raises(ValueError, match="original"), TrainingShutdown() as shutdown:
        shutdown.accelerator = SimpleNamespace(end_training=failing_end)
        raise ValueError("original training error")
    assert fake_group.destroys == 1


def test_cleanup_error_is_reported_on_normal_exit(fake_group):
    def failing_end():
        raise RuntimeError("tracker failed")

    with pytest.raises(RuntimeError, match="tracker failed"), TrainingShutdown() as shutdown:
        shutdown.accelerator = SimpleNamespace(end_training=failing_end)
    assert fake_group.destroys == 1


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_waits_for_boundary_and_repeated_signals_do_not_abort_cleanup(fake_group, signum):
    calls = []

    def end():
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        calls.append("cleanup completed")

    with pytest.raises(TrainingInterrupted) as raised, TrainingShutdown() as shutdown:
        shutdown.accelerator = SimpleNamespace(end_training=end)
        os.kill(os.getpid(), signum)
        calls.append("batch completed")
        shutdown.raise_if_requested(torch.device("cpu"))
    assert raised.value.signum == signum
    assert calls == ["batch completed", "cleanup completed"]
    assert fake_group.destroys == 1


def test_signal_arriving_during_collective_is_preserved_for_next_boundary(fake_group, monkeypatch):
    def late_signal(tensor, **kwargs):
        assert tensor.item() == 0
        os.kill(os.getpid(), signal.SIGINT)

    with TrainingShutdown() as shutdown:
        monkeypatch.setattr(shutdown_module.dist, "all_reduce", late_signal)
        shutdown.raise_if_requested(torch.device("cpu"))  # peers agreed on zero, so do not stop yet
        assert shutdown.signum == signal.SIGINT
        monkeypatch.setattr(shutdown_module.dist, "all_reduce", lambda *args, **kwargs: None)
        with pytest.raises(TrainingInterrupted):
            shutdown.raise_if_requested(torch.device("cpu"))


def test_remote_rank_request_stops_this_rank(fake_group, monkeypatch):
    monkeypatch.setattr(
        shutdown_module.dist, "all_reduce", lambda tensor, **kwargs: tensor.fill_(signal.SIGINT)
    )
    with TrainingShutdown() as shutdown:
        assert shutdown.signum == 0
        with pytest.raises(TrainingInterrupted):
            shutdown.raise_if_requested(torch.device("cpu"))


def _terminate_forked_worker():
    os.kill(os.getpid(), signal.SIGTERM)
    raise AssertionError("Inherited training handler swallowed worker termination")


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_forked_dataloader_worker_does_not_swallow_termination(fake_group):
    with TrainingShutdown():
        process = multiprocessing.get_context("fork").Process(target=_terminate_forked_worker)
        process.start()
        try:
            process.join(timeout=5)
            assert process.exitcode == -signal.SIGTERM
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def test_end_training_destroy_is_not_repeated(fake_group):
    with TrainingShutdown() as shutdown:
        shutdown.accelerator = SimpleNamespace(end_training=shutdown_module.dist.destroy_process_group)
    assert fake_group.destroys == 1


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")
def test_data_worker_keeps_serving_after_ctrl_c_and_can_still_terminate(fake_group):
    fake_group.initialized = False  # real DataLoader, no actual process group in this unit test
    with TrainingShutdown():
        loader = torch.utils.data.DataLoader(
            torch.arange(20),
            batch_size=1,
            num_workers=1,
            multiprocessing_context="spawn",
            worker_init_fn=init_training_data_worker,
            timeout=15,
        )
        iterator = iter(loader)
        worker = iterator._workers[0]
        try:
            assert next(iterator).item() == 0  # worker initialization completed
            os.kill(worker.pid, signal.SIGINT)
            assert next(iterator).item() == 1
            assert worker.is_alive()
        finally:
            iterator._shutdown_workers()
        assert not worker.is_alive()
