# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Exercise actual torchrun signal forwarding and two-rank CUDA cleanup."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("transformers")


@pytest.mark.multigpu
@pytest.mark.skipif(os.name != "posix" or torch.cuda.device_count() < 2, reason="requires POSIX and two GPUs")
@pytest.mark.parametrize("layout,delivery", [("ddp", "terminal"), ("fsdp", "terminal"), ("fsdp", "rank0")])
def test_torchrun_training_signal_cleans_all_ranks(tmp_path, layout, delivery):
    log_path = tmp_path / "torchrun.log"
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "tests.fixtures.distributed_shutdown_worker",
        "--output",
        str(tmp_path),
        "--layout",
        layout,
    ]
    workers = []
    with log_path.open("w") as log:
        launcher = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 90
            while not all((tmp_path / f"ready-{rank}.json").exists() for rank in range(2)):
                if launcher.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"Workers did not become ready:\n{log_path.read_text()}")
                time.sleep(0.05)
            workers = [json.loads((tmp_path / f"ready-{rank}.json").read_text())["pid"] for rank in range(2)]
            if delivery == "terminal":
                # A separate session makes this equivalent to terminal Ctrl+C
                # without sending signals to pytest or the user's own shell.
                os.killpg(launcher.pid, signal.SIGINT)
                expected_signal = signal.SIGINT
            else:
                os.kill(workers[0], signal.SIGTERM)
                expected_signal = signal.SIGTERM
            launcher.wait(timeout=40)
            assert launcher.returncode != 0
            stopped = []
            for rank in range(2):
                report = tmp_path / f"stopped-{rank}.json"
                assert report.exists(), log_path.read_text()
                stopped.append(json.loads(report.read_text()))
            assert all(report["signum"] == expected_signal for report in stopped)
            assert all(not report["group_initialized"] for report in stopped)
            assert stopped[0]["step"] == stopped[1]["step"] >= 2
            assert "process group has NOT been destroyed" not in log_path.read_text()
        finally:
            if launcher.poll() is None:
                # Only this test's isolated launcher session and recorded workers.
                with suppress(ProcessLookupError):
                    os.killpg(launcher.pid, signal.SIGTERM)
                try:
                    launcher.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(launcher.pid, signal.SIGKILL)
                    for pid in workers:
                        with suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGKILL)
                    launcher.wait(timeout=10)
