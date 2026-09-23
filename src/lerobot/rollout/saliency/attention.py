"""Single-group streaming attention reduction, independent of model and display code."""

from collections.abc import Iterable

import torch


class AttentionAccumulator:
    """Consume head-mean rows; retain one float32 sum, not the attention matrices.

    A capture callback can send the same row to several independent accumulators
    in a future multi-group implementation without running the model again.
    """

    def __init__(self, layers: Iterable[int] | None = None, steps: Iterable[int] | None = None):
        self.layers = None if layers is None else set(layers)
        self.steps = None if steps is None else set(steps)
        self.seen: set[tuple[int, int]] = set()
        self.total: torch.Tensor | None = None

    def add(self, layer: int, step: int | None, row: torch.Tensor) -> None:
        if step is None:
            raise RuntimeError("Attention step identity is unavailable")
        if self.layers is not None and layer not in self.layers:
            return
        if self.steps is not None and step not in self.steps:
            return
        event = (layer, step)
        if event in self.seen:
            raise RuntimeError(f"Duplicate attention event for layer={layer}, step={step}")
        if row.ndim != 1:
            raise RuntimeError("Attention event must contain a single head-mean query row")
        if self.total is None:
            self.total = row.detach().float().clone()
        else:
            if self.total.shape != row.shape:
                raise RuntimeError("Attention key layout changed between events")
            self.total += row.detach().float()
        self.seen.add(event)

    def mean(self, valid_layers: Iterable[int], num_steps: int) -> torch.Tensor:
        if self.total is None:
            raise RuntimeError("No selected attention events were captured during replay")
        layers = set(valid_layers) if self.layers is None else self.layers
        steps = set(range(num_steps)) if self.steps is None else self.steps
        expected = {(layer, step) for layer in layers for step in steps}
        if self.seen != expected:
            missing = sorted(expected - self.seen)
            extra = sorted(self.seen - expected)
            raise RuntimeError(f"Attention event set mismatch; missing={missing[:4]}, extra={extra[:4]}")
        return self.total / len(self.seen)
