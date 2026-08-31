"""Batch container and training state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from transformer.model.transformer import subsequent_mask


@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


class Batch:
    """Hold a batch with source/target masks for training."""

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor | None = None, pad: int = 2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = int((self.tgt_y != pad).data.sum())
        else:
            self.tgt = None
            self.tgt_y = None
            self.tgt_mask = None
            self.ntokens = 0

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
