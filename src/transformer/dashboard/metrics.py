"""Training metrics collection and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_perplexity: float = 0.0
    val_perplexity: float = 0.0
    train_kl: float = 0.0
    val_kl: float = 0.0
    corpus_bleu: float | None = None
    corpus_perplexity: float | None = None
    gpu_temp: float | None = None
    learning_rate: float = 0.0


@dataclass
class MetricsTracker:
    """Accumulate per-epoch metrics for dashboard rendering."""

    history: list[EpochMetrics] = field(default_factory=list)
    batch_losses: list[float] = field(default_factory=list)
    batch_steps: list[int] = field(default_factory=list)

    def record_batch(self, step: int, loss: float) -> None:
        self.batch_losses.append(loss)
        self.batch_steps.append(step)

    def record_epoch(self, metrics: EpochMetrics) -> None:
        self.history.append(metrics)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "epochs": [asdict(m) for m in self.history],
            "batch_losses": self.batch_losses,
            "batch_steps": self.batch_steps,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> MetricsTracker:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        tracker = cls()
        tracker.batch_losses = payload.get("batch_losses", [])
        tracker.batch_steps = payload.get("batch_steps", [])
        tracker.history = [EpochMetrics(**m) for m in payload.get("epochs", [])]
        return tracker

    @staticmethod
    def loss_to_perplexity(loss: float) -> float:
        import math

        return math.exp(min(loss, 20.0))
