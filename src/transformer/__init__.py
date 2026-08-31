"""Modular OOP implementation of the Annotated Transformer."""

from transformer.config import TrainingConfig

__all__ = ["TrainingConfig", "Trainer"]


def __getattr__(name: str):
    if name == "Trainer":
        from transformer.training.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
