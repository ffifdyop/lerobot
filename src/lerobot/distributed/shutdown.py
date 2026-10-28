# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Cooperative training interrupts and unconditional distributed cleanup.

Signals request a stop; they never interrupt an in-flight collective. Every rank
must call ``raise_if_requested`` at the same training boundaries. SIGKILL, a stuck
rank, and torchrun's eventual forced-termination timeout cannot be handled here.
"""

import logging
import os
import signal
import threading
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from accelerate import Accelerator


def init_training_data_worker(worker_id: int) -> None:
    """Let the trainer handle Ctrl+C while keeping worker termination available.

    torchrun forwards terminal signals to each worker process group, which also
    contains its DataLoader children. They must keep serving the current batch
    until the trainer reaches its coordinated stop boundary. SIGTERM retains its
    default meaning for DataLoader's forced worker cleanup (including after fork).

    Args:
        worker_id: Worker index supplied by DataLoader.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


class TrainingInterrupted(KeyboardInterrupt):
    """All ranks have agreed to stop at a safe training boundary."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"Training interrupted by {signal.Signals(signum).name}")


class TrainingShutdown:
    """Own signal handlers and cleanup for one local training invocation.

    Enter before constructing the Accelerator, then assign it to ``accelerator``.
    Cleanup also handles partial distributed initialization. No shutdown barrier
    is added: peers may already have failed when unwinding an ordinary exception.
    """

    def __init__(self) -> None:
        self.accelerator: Accelerator | None = None
        self.signum = 0
        self._owner_pid = os.getpid()
        self._original_handlers: dict[int, Any] = {}
        self._request: torch.Tensor | None = None

    def __enter__(self) -> "TrainingShutdown":
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("TrainingShutdown must run on the training process's main thread.")
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        return self

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        if os.getpid() != self._owner_pid:
            # A forked DataLoader worker must not swallow its termination signal.
            previous = self._original_handlers[signum]
            signal.signal(signum, previous)
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                os.kill(os.getpid(), signum)
            return
        # Repeated signals (including torchrun forwarding the terminal SIGINT)
        # leave cleanup and ongoing CUDA/NCCL operations intact.
        if not self.signum:
            self.signum = signum

    def raise_if_requested(self, device: torch.device) -> None:
        """Collectively agree on a stop; all ranks must call in the same order.

        Args:
            device: The Accelerator device; NCCL requires a CUDA request tensor.

        Raises:
            TrainingInterrupted: Any rank has requested SIGINT/SIGTERM shutdown.
        """
        if dist.is_initialized():
            if self._request is None:
                self._request = torch.zeros((), dtype=torch.int32, device=device)
            self._request.fill_(self.signum)
            dist.all_reduce(self._request, op=dist.ReduceOp.MAX)
            agreed_signal = int(self._request.item())
            if not agreed_signal:
                # A signal can arrive AFTER fill_ or during the collective/item.
                # Preserve it for the next boundary; never stop only this rank.
                return
            self.signum = agreed_signal
        if self.signum:
            raise TrainingInterrupted(self.signum)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            try:
                if self.accelerator is not None:
                    self.accelerator.end_training()
            finally:
                # end_training may fail while finishing a tracker, or construction
                # may have initialized a process group before returning an object.
                if dist.is_initialized():
                    dist.destroy_process_group()
        except Exception:
            if exc_type is None:
                raise
            logging.warning("Distributed cleanup failed while handling a training exit", exc_info=True)
        finally:
            for signum, handler in self._original_handlers.items():
                signal.signal(signum, handler)
