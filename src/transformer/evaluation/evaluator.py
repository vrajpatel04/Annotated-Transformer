"""Corpus-level evaluation: BLEU and perplexity."""

from __future__ import annotations

import math

import torch
from sacrebleu.metrics import BLEU

from transformer.data.multi30k import Multi30kDataModule
from transformer.model.transformer import greedy_decode
from transformer.training.batch import Batch
from transformer.training.loss import LossComputer


class Evaluator:
    """Evaluate a trained transformer on Multi30k."""

    def __init__(
        self,
        model: torch.nn.Module,
        data_module: Multi30kDataModule,
        loss_computer: LossComputer,
        device: torch.device,
        max_decode_len: int = 72,
    ):
        self.model = model
        self.data_module = data_module
        self.loss_computer = loss_computer
        self.device = device
        self.max_decode_len = max_decode_len
        self.pad_idx = data_module.pad_idx
        self.bleu = BLEU()

    @torch.no_grad()
    def corpus_perplexity(self, dataloader) -> tuple[float, float, float, float]:
        """
        Compute corpus-level perplexity on a dataloader.

        Returns (avg_loss, avg_kl, perplexity, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        total_kl = 0.0
        total_tokens = 0
        total_accuracy = 0.0
        n_batches = 0

        for src, tgt in dataloader:
            batch = Batch(src, tgt, self.pad_idx)
            out = self.model(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            loss_val, _, accuracy = self.loss_computer(
                out, batch.tgt_y, batch.ntokens
            )
            kl_val = loss_val.item()
            total_kl += kl_val
            total_loss += kl_val / batch.ntokens
            total_tokens += batch.ntokens
            total_accuracy += accuracy
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_kl = total_kl / max(total_tokens, 1)
        avg_accuracy = total_accuracy / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, avg_kl, perplexity, avg_accuracy

    @torch.no_grad()
    def corpus_bleu(self, dataloader, max_samples: int | None = None) -> float:
        """Compute corpus BLEU with greedy decoding."""
        self.model.eval()
        hypotheses: list[str] = []
        references: list[list[str]] = []
        start_symbol = 0
        eos = "</s>"
        count = 0

        for src, tgt in dataloader:
            for i in range(src.size(0)):
                if max_samples is not None and count >= max_samples:
                    break

                src_i = src[i : i + 1]
                tgt_i = tgt[i : i + 1]
                src_mask = (src_i != self.pad_idx).unsqueeze(-2)

                decoded = greedy_decode(
                    self.model,
                    src_i,
                    src_mask,
                    self.max_decode_len,
                    start_symbol,
                )[0]
                hyp = self.data_module.decode_prediction(decoded, eos=eos)
                ref_tokens = self.data_module.ids_to_tokens(tgt_i[0], side="tgt")
                ref = " ".join(ref_tokens).split(eos, 1)[0] + eos

                hypotheses.append(hyp)
                references.append([ref])
                count += 1

            if max_samples is not None and count >= max_samples:
                break

        if not hypotheses:
            return 0.0
        return float(self.bleu.corpus_score(hypotheses, references).score)

    def evaluate(self, dataloader, max_bleu_samples: int | None = None) -> dict:
        loss, kl, perplexity, accuracy = self.corpus_perplexity(dataloader)
        bleu = self.corpus_bleu(dataloader, max_samples=max_bleu_samples)
        return {
            "loss": loss,
            "kl": kl,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "corpus_bleu": bleu,
            "corpus_perplexity": perplexity,
        }
