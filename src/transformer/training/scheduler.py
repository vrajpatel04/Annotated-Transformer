"""Learning rate schedule from Attention Is All You Need."""

from __future__ import annotations


def learning_rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
