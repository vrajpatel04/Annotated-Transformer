"""Transformer factory and decoding utilities."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from transformer.config import TrainingConfig
from transformer.model.attention import MultiHeadedAttention
from transformer.model.layers import (
    Decoder,
    DecoderLayer,
    Embeddings,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    Generator,
    PositionalEncoding,
    PositionwiseFeedForward,
)


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~mask


class TransformerFactory:
    """Build encoder-decoder transformers from configuration."""

    @staticmethod
    def build(
        src_vocab: int,
        tgt_vocab: int,
        config: TrainingConfig | None = None,
    ) -> EncoderDecoder:
        if config is None:
            config = TrainingConfig()
        return make_model(
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            n=config.num_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            h=config.num_heads,
            dropout=config.dropout,
        )


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def greedy_decode(
    model: EncoderDecoder,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
) -> torch.Tensor:
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory,
            src_mask,
            ys,
            subsequent_mask(ys.size(1)).type_as(src.data),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
