"""Loss functions and optimizers."""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """Label smoothing via KL divergence (from the Annotated Transformer)."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class LossComputer:
    """Forward pass loss with optional token accuracy."""

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        logits = self.generator(x)
        kl_loss = self.criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            y.contiguous().view(-1),
        )
        loss = kl_loss / norm

        with torch.no_grad():
            preds = logits.argmax(dim=-1).view(-1)
            accuracy = (preds == y.contiguous().view(-1)).float().mean().item()

        return loss.data * norm, loss, accuracy


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        return None

    def zero_grad(self, set_to_none: bool = False):
        return None


class DummyScheduler:
    def step(self):
        return None
