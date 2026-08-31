"""Training and model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Configurable hyperparameters for Multi30k transformer training."""

    # Training schedule
    num_epochs: int = 8
    batch_size: int = 32
    accum_iter: int = 10
    base_lr: float = 1.0
    warmup: int = 3000

    # Model architecture
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Data
    max_padding: int = 72
    vocab_min_freq: int = 2
    language_pair: tuple[str, str] = ("de", "en")

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_prefix: str = "multi30k_model_"
    vocab_path: Path = field(default_factory=lambda: Path("outputs/vocab.pt"))

    # Runtime
    device: str = "auto"
    distributed: bool = False
    seed: int = 42

    # Evaluation & dashboard
    eval_every_epoch: bool = True
    max_decode_len: int = 72
    bleu_max_samples: int | None = None  # None = full validation corpus
    dashboard_path: Path = field(
        default_factory=lambda: Path("outputs/dashboard.html")
    )
    metrics_path: Path = field(
        default_factory=lambda: Path("outputs/metrics.json")
    )

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.vocab_path = Path(self.vocab_path)
        self.dashboard_path = Path(self.dashboard_path)
        self.metrics_path = Path(self.metrics_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "language_pair" in data and isinstance(data["language_pair"], list):
            data["language_pair"] = tuple(data["language_pair"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result

    def checkpoint_path(self, epoch: int | None = None) -> Path:
        if epoch is None:
            return self.output_dir / f"{self.checkpoint_prefix}final.pt"
        return self.output_dir / f"{self.checkpoint_prefix}{epoch:02d}.pt"
