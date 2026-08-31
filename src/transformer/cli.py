"""CLI entry point for Multi30k transformer training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformer.config import TrainingConfig
from transformer.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Annotated Transformer on Multi30k (de-en)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Run full training or evaluation-only on a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for evaluation mode.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.base_lr = args.lr
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device is not None:
        config.device = args.device
    return config


def main() -> None:
    args = parse_args()

    if args.config.exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    config = apply_overrides(config, args)
    trainer = Trainer(config)

    if args.mode == "train":
        trainer.train()
    else:
        results = trainer.evaluate_only(
            checkpoint=str(args.checkpoint) if args.checkpoint else None
        )
        print("\nEvaluation results:")
        for key, value in results.items():
            print(
                f"  {key}: {value:.4f}"
                if isinstance(value, float)
                else f"  {key}: {value}"
            )


if __name__ == "__main__":
    main()
